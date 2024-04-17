use std::collections::HashMap;
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    AdamW, Dropout, Embedding, LayerNorm, LayerNormConfig, Linear, Module, Optimizer, VarBuilder,
    VarMap,
};
use rand::{distributions::Distribution, rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};

const DROPOUT: f32 = 0.2;

// Note: pulled from my other Candle project ... consider pulling in the sampler from Lantern.
fn sample_multinomial(rng: &mut rand::rngs::StdRng, prs: &Vec<f32>) -> Result<u32> {
    let distr = rand::distributions::WeightedIndex::new(prs).map_err(anyhow::Error::msg)?;
    let next_token = distr.sample(rng) as u32;
    Ok(next_token)
}

pub struct DatasetBatcher {
    rng: StdRng,
    total_tokens: HashMap<String, usize>,
    data: HashMap<String, Tensor>,
}
impl DatasetBatcher {
    // creates a new DatasetBatcher, ready to have datasets inserted
    pub fn new(rng: &StdRng) -> Self {
        let data = HashMap::new();
        let total_tokens = HashMap::new();
        Self {
            rng: rng.clone(),
            total_tokens,
            data,
        }
    }

    // adds a new dataset to the batcher, indexed by name
    pub fn add(&mut self, name: String, data: Tensor, total_tokens: usize) {
        self.total_tokens.insert(name.clone(), total_tokens);
        self.data.insert(name, data);
    }

    fn get_batch(
        &mut self,
        name: &String,
        num_of_batches: usize,
        block_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let total_tokens = self
            .total_tokens
            .get(name)
            .ok_or_else(|| anyhow!("Failed to get total_tokens for dataset \"{}\".", name))?;
        let data = self
            .data
            .get(name)
            .ok_or_else(|| anyhow!("Failed to get data for dataset \"{}\".", name))?;

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _batch_i in 0..num_of_batches {
            let random_start: usize = self.rng.gen_range(0..(total_tokens - 1 - block_size));
            let x = data.i(random_start..random_start + block_size)?;
            let y = data.i(random_start + 1..random_start + 1 + block_size)?;
            xs.push(x);
            ys.push(y);
        }
        let batched_x = Tensor::stack(xs.as_slice(), 0)?;
        let batched_y = Tensor::stack(ys.as_slice(), 0)?;
        Ok((batched_x, batched_y))
    }
}

struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout: Dropout,
}
impl MultiHeadAttention {
    fn new(vb: VarBuilder, head_count: usize, head_size: usize, n_embed: usize) -> Result<Self> {
        let mut heads = Vec::new();
        for i in 0..head_count {
            heads.push(Head::new(
                vb.push_prefix(format!("Multihead_{}", i)),
                head_size,
                n_embed,
            )?);
        }
        let proj = create_linear(
            n_embed,
            n_embed,
            vb.push_prefix("residual_projection"),
            0.0,
            0.2,
            true,
        )?;
        let dropout = Dropout::new(DROPOUT);

        Ok(Self {
            heads,
            proj,
            dropout,
        })
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Result<Tensor> {
        let mut results = Vec::new();
        for head in &mut self.heads {
            results.push(head.forward(x, training)?);
        }
        let result = Tensor::cat(results.as_slice(), 2)?.contiguous()?;
        let out = self.proj.forward(&result)?;
        let dropped = self.dropout.forward(&out, training)?;
        Ok(dropped)
    }
}

struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    head_size: usize,
    dropout: Dropout,
}
impl Head {
    fn new(vb: VarBuilder, head_size: usize, n_embed: usize) -> Result<Self> {
        let key = create_linear(
            n_embed,
            head_size,
            vb.push_prefix("head_key"),
            0.0,
            0.2,
            false,
        )?;
        let query = create_linear(
            n_embed,
            head_size,
            vb.push_prefix("head_query"),
            0.0,
            0.2,
            false,
        )?;
        let ln_value = create_linear(
            n_embed,
            head_size,
            vb.push_prefix("head_value"),
            0.0,
            0.2,
            false,
        )?;
        let dropout = Dropout::new(DROPOUT);

        Ok(Self {
            key,
            query,
            value: ln_value,
            head_size,
            dropout,
        })
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Result<Tensor> {
        let (_b, time_x, _c) = x.shape().dims3()?;
        let k = self.key.forward(x)?; // (B,T,C)
        let q = self.query.forward(x)?; // (B,T,C)

        let k_t = k.transpose(D::Minus2, D::Minus1)?; // (B,C,T)
        let mut weights = q.matmul(&k_t)?; // (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = (weights * (1.0 / (self.head_size as f64).sqrt()))?;
        let (batch, _, _) = weights.shape().dims3()?;
        let tril = Tensor::tril2(time_x, DType::U8, &x.device())?.broadcast_left(batch)?;
        let neginf = Tensor::try_from(f32::NEG_INFINITY)?
            .to_device(&x.device())?
            .broadcast_as(tril.shape())?;
        let weights_masked = tril.where_cond(&weights, &neginf)?; // (B,T,T)
        let weights_softmaxed = candle_nn::ops::softmax(&weights_masked, 2).unwrap(); // (B,T,T)
        let dropped = self.dropout.forward(&weights_softmaxed, training)?;
        let v = self.value.forward(x)?; // (B,T,C)
        let out = dropped.matmul(&v)?; // (B,T,T) @ (B,T,C) -> (B,T,C)
        Ok(out)
    }
}

struct FeedForward {
    net: Linear,
    proj: Linear,
    dropout: Dropout,
}
impl FeedForward {
    fn new(vb: VarBuilder, n_embed: usize) -> Result<Self> {
        let linear = create_linear(
            n_embed,
            4 * n_embed,
            vb.push_prefix("ffwd_linear"),
            0.0,
            0.2,
            true,
        )?;
        let proj = create_linear(
            4 * n_embed,
            n_embed,
            vb.push_prefix("ffwd_projection"),
            0.0,
            0.2,
            true,
        )?;
        let dropout = Dropout::new(DROPOUT);

        Ok(Self {
            net: linear,
            proj,
            dropout,
        })
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Result<Tensor> {
        let result = self.net.forward(x)?;
        let rectified = result.relu()?;
        let proj = self.proj.forward(&rectified)?;
        let dropped = self.dropout.forward(&proj, training)?;
        Ok(dropped)
    }
}

struct Block {
    sa: MultiHeadAttention,
    ffwd: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}
impl Block {
    // remember to keep e_embed divisible by num_of_heads.
    fn new(num_of_heads: usize, n_embed: usize, vb: VarBuilder) -> Result<Self> {
        let head_size = n_embed / num_of_heads;
        assert_eq!(n_embed % num_of_heads, 0);

        let sa = MultiHeadAttention::new(
            vb.push_prefix("Multiattn Heads"),
            num_of_heads,
            head_size,
            n_embed,
        )?;
        let ffwd = FeedForward::new(vb.push_prefix("ffwd"), n_embed)?;

        let ln_config = LayerNormConfig::default();
        let ln1 = candle_nn::layer_norm(n_embed, ln_config, vb.push_prefix("ln1"))?;
        let ln2 = candle_nn::layer_norm(n_embed, ln_config, vb.push_prefix("ln2"))?;

        Ok(Self { sa, ffwd, ln1, ln2 })
    }

    fn forward(&mut self, x: &Tensor, training: bool) -> Result<Tensor> {
        let sa_out = (self.sa.forward(&self.ln1.forward(x)?, training)? + x)?;
        let x = (self.ffwd.forward(&self.ln2.forward(&sa_out)?, training)? + sa_out)?;
        Ok(x)
    }
}

// a custom Embedding creation function to allow for custom `mean` and `stdev` values,
// since the default ones don't match Karpathy's nanogpt implementation.
pub fn create_embedding(
    in_size: usize,
    out_size: usize,
    vb: VarBuilder,
    mean: f64,
    stdev: f64,
) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        candle_nn::Init::Randn { mean, stdev },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}

// a custom Linear creation function to allow for custom `mean` and `stdev` values,
// since the default ones don't match Karpathy's nanogpt implementation.
pub fn create_linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    mean: f64,
    stdev: f64,
    bias: bool,
) -> Result<Linear> {
    let ws = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        candle_nn::Init::Randn { mean, stdev },
    )?;
    let mut bs = None;
    if bias {
        bs = Some(vb.get_with_hints(out_dim, "bias", candle_nn::Init::Const(0.0))?);
    }
    Ok(Linear::new(ws, bs))
}

// Specifies the required sizes for the components of the model.
// (mocking the llama json file naming scheme here...)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NanoGptModelConfig {
    // number of heads in the transformer per layer block
    pub num_attention_heads: usize,

    // number of layer blocks in the model
    pub num_hidden_layers: usize,

    // number of tokens possible in the vocabulary
    pub vocab_size: usize,

    // maximum context length for the model
    pub block_size: usize,

    // size of the embeddings for each token
    pub hidden_size: usize,

    // learning rate used to train the model
    pub learning_rate: f64,
}

pub struct NanoGptModel {
    pub varmap: VarMap,
    token_embedding_table: candle_nn::Embedding,
    position_embedding_table: candle_nn::Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: candle_nn::Linear,
    dropout: candle_nn::Dropout,
    config: NanoGptModelConfig,
    losses: Vec<f32>,
    optimizer: AdamW,
}
impl NanoGptModel {
    // creates a new model with the following parameters:
    //   * `num_of_heads`: number of attention heads per layer block
    //   * `num_of_hidden_layers`: number of layer blocks in the model
    //   * `vocab_size`: should be the number of tokens for the model's vocabulary
    //   * `block_size`: the maximum context window for the model
    //   * `n_embed`: the size of the internal embeddings for each token
    //   * `learning_rate`: the learning rate to use for training th emodel
    pub fn new(
        num_of_heads: usize,
        num_of_hidden_layers: usize,
        vocab_size: usize,
        block_size: usize,
        n_embed: usize,
        learning_rate: f64,
        device: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let token_embedding_table = create_embedding(
            vocab_size,
            n_embed,
            vb.push_prefix("token_embedding"),
            0.0,
            0.2,
        )?;
        let position_embedding_table = create_embedding(
            block_size,
            n_embed,
            vb.push_prefix("position_embedding"),
            0.0,
            0.2,
        )?;

        let mut blocks = Vec::new();
        for i in 0..num_of_hidden_layers {
            blocks.push(Block::new(
                num_of_heads,
                n_embed,
                vb.push_prefix(format!("block_{}", i)),
            )?);
        }

        let dropout = Dropout::new(DROPOUT);
        let ln_config = LayerNormConfig::default();
        let ln_f = candle_nn::layer_norm(n_embed, ln_config, vb.push_prefix("ln_f"))?;
        let lm_head = create_linear(
            n_embed,
            vocab_size,
            vb.push_prefix("linear_output"),
            0.0,
            0.2,
            true,
        )?;

        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

        let config = NanoGptModelConfig {
            num_attention_heads: num_of_heads,
            num_hidden_layers: num_of_hidden_layers,
            vocab_size,
            block_size,
            hidden_size: n_embed,
            learning_rate,
        };

        Ok(Self {
            varmap: varmap,
            token_embedding_table,
            position_embedding_table,
            blocks,
            ln_f,
            lm_head,
            dropout,
            config,
            losses: Vec::new(),
            optimizer,
        })
    }

    // loads a model from the 'file stem' specified. for example, if file_stem
    // is "model_step_15000", then this function will attempt to load the model
    // from "model_step_15000.safetensors" and use the parameters from the file
    // "model_step_15000.config.json".
    pub fn load_from_safetensors(file_stem: &str) -> Result<Self> {
        // load the model configuration
        let f = std::fs::File::open(format!("{}.config.json", file_stem))?;
        let bf = std::io::BufReader::new(f);
        let config: NanoGptModelConfig = serde_json::from_reader(bf)?;

        // setup some Candle stuff to setup the device to use
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).unwrap();
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0).unwrap();
        #[cfg(not(feature = "cuda"))]
        #[cfg(not(feature = "metal"))]
        let device = Device::Cpu;

        let mut model = NanoGptModel::new(
            config.num_attention_heads,
            config.num_hidden_layers,
            config.vocab_size,
            config.block_size,
            config.hidden_size,
            config.learning_rate,
            &device,
        )?;
        model.varmap.load(format!("{}.safetensors", file_stem))?;
        Ok(model)
    }

    pub fn save_to_file(&mut self, file_stem: &str) -> Result<()> {
        // save the model safetensors
        self.varmap.save(format!("{}.safetensors", file_stem))?;

        // save the model config json
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(format!("{}.config.json", file_stem), config_json)?;

        Ok(())
    }

    fn forward(
        &mut self,
        idx: &Tensor,
        maybe_targets: Option<&Tensor>,
        training: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (_batch_size, time_size) = idx.shape().dims2()?; // idx and targets are both shape (B,T)
        let tok_emb = self
            .token_embedding_table
            .forward(idx)
            .map_err(anyhow::Error::msg)?; // (B,T,n_embed)
        let positionals = Tensor::arange(0, time_size as u32, idx.device())?;
        let pos_emb = self
            .position_embedding_table
            .forward(&positionals)
            .map_err(anyhow::Error::msg)?; // (T,n_embed)
        let mut x = tok_emb.broadcast_add(&pos_emb)?; // pos brodcasted, result is (B,T,C)
        x = self.dropout.forward(&x, training)?;

        for block in &mut self.blocks {
            x = block.forward(&x, training)?;
        }
        x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x).map_err(anyhow::Error::msg)?;

        let mut loss = None;
        // if targets are supplied, then calculate a loss
        if let Some(targets) = maybe_targets {
            // just like pytorch, doing cross_entropy loss requires manipulation of the
            // shape of the tensor to reduce it to the dimensionality it wants in its view.
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let logits_ce = logits.reshape((batch_size * time_size, channel_size))?;
            let (batch_size, time_size, _channel_size) = logits.shape().dims3()?;
            let targets_ce = targets.reshape(batch_size * time_size)?;

            let loss_tensor = candle_nn::loss::cross_entropy(&logits_ce, &targets_ce)?;
            loss = Some(loss_tensor);
        }
        Ok((logits, loss))
    }

    // Generates `max_new_tokens` from the model while keeping the context window
    // set to be `block_size` in dimension at most. The nanogpt video starts off
    // the generation with a tensor containing the newline character, which can be
    // supplied via the `idx` parameter. `temperature` affects token sampling.
    pub fn generate(
        &mut self,
        rng: &mut rand::rngs::StdRng,
        idx: Tensor,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Tensor> {
        let mut idx = idx;
        for _i in 0..max_new_tokens {
            let (_, time_size) = idx.shape().dims2()?;
            let current_idx = idx.i((.., time_size.saturating_sub(self.config.block_size)..))?;
            // get the predictions only, we're not training and tracking loss
            // logits should be of shape (Batch, Time, Channel)
            let (logits, _) = self.forward(&current_idx, None, false)?;

            // focus only on the last 'time step'; the last part of a batch
            // this should reduce it to shape (Batch, Channel)
            let (_batch_size, time_size, _channel_size) = logits.shape().dims3()?;
            let mut logits = logits.i((.., time_size - 1, ..))?;

            // basic sampling addition
            logits = (logits / temperature as f64)?;

            // now we apply softmax to get the probabilities into a more usable distribution
            // between 0 and 1, with the sum adding up to 1. probs is shape (Batch, Channel)
            let probs = candle_nn::ops::softmax(&logits, 1)?;

            // FIXME: probably screwing up batching here.
            let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
            let idx_next = sample_multinomial(rng, &probs_vec)?;

            // Note: our sample_multinomial here doesn't deal with batching so it just returns
            // the index. To concat it to the way this model is written in the YT video
            // we're going to unsqueeze it to give it another dimension.
            let idx_next_tensor = Tensor::new(&[idx_next], &current_idx.device())?.unsqueeze(0)?;
            idx = Tensor::cat(&[idx, idx_next_tensor], 1)?;
        }

        Ok(idx)
    }

    // calculates the loss when attempting to predict the next token for a given number of batchs
    // of data from the batcher using the dataset_name specified.
    pub fn eval_loss(
        &mut self,
        dataset_name: &String,
        num_of_batchs: usize,
        batcher: &mut DatasetBatcher,
    ) -> Result<f32> {
        let (xb, yb) = batcher.get_batch(dataset_name, num_of_batchs, self.config.block_size)?;
        let (_logits, loss) = self.forward(&xb, Some(&yb), false)?;
        let loss_f = loss.unwrap().to_scalar::<f32>()?;
        Ok(loss_f)
    }

    // runs a training loop, pulling the specified number of batches from the batcher using the
    // dataset_name specified and runs the loop for the count specified in `num_epochs`. With
    // each step the loss is propegated backwards by the AdamW optimizer.
    pub fn train(
        &mut self,
        dataset_name: &String,
        num_epochs: usize,
        num_of_batchs: usize,
        batcher: &mut DatasetBatcher,
    ) -> Result<f32> {
        for _epoch in 0..num_epochs {
            let (xb, yb) =
                batcher.get_batch(dataset_name, num_of_batchs, self.config.block_size)?;
            let (_logits, loss) = self.forward(&xb, Some(&yb), true)?;
            let loss = loss.as_ref().unwrap();
            self.optimizer.backward_step(loss)?;

            let loss_f = loss.to_scalar::<f32>()?;
            self.losses.push(loss_f);
        }
        let loss_value = self.losses.last().unwrap();
        Ok(*loss_value)
    }
}