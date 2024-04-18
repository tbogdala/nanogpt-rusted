use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use clap::{ArgAction, Parser};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use log::{error, info};
use model::{DatasetBatcher, NanoGptModel};
use rand::SeedableRng;
use ratatui::{
    prelude::*,
    widgets::{Axis, Borders, Chart, Clear, Dataset, GraphType, Paragraph, Wrap},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{read_dir, File},
    io::{stdout, BufReader, Read, Write},
    path::PathBuf,
    process::exit,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

mod model;

// Clap provides an easy to use way of parsing command-line options by just
// creating a struct and labelling the members with attributes to control
// the parsing behavior.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        value_name = "Text-Dataset-File",
        help = "Prepares the source text by tokenizing it all at character-level."
    )]
    pub prepare_dataset: Option<String>,

    #[arg(
        long,
        value_name = "Text-Dataset-File",
        help = "Prepares the source text by tokenizing it all with the gpt2 tokenizer."
    )]
    pub prepare_dataset_gpt2: Option<String>,

    #[arg(long, action=ArgAction::SetTrue, help = "Train a model based on the dataset provided.")]
    pub train: bool,

    #[arg(
        long,
        value_name = "Training-Dataset-File",
        help = "The prepared binary training dataset to use for training."
    )]
    pub training_dataset: Option<String>,

    #[arg(
        long,
        value_name = "Validation-Dataset-File",
        help = "The prepared binary training dataset to use for validation."
    )]
    pub validation_dataset: Option<String>,

    #[arg(
        long,
        value_name = "Vocabulary-Metadata-File",
        help = "The prepared vocabulary metadata file."
    )]
    pub vocab_metadata: Option<String>,

    #[arg(
        long,
        default_value_t = 1337,
        help = "The seed to use for RNG; if 0, a random one will be used."
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Total training steps to perform."
    )]
    pub steps: usize,

    #[arg(
        long,
        default_value_t = 32,
        help = "The size of the baches sent into each training step."
    )]
    pub batch_size: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "The maximum context size for the transformer."
    )]
    pub block_size: usize,

    #[arg(long, default_value_t = 32, help = "The embedding size of the tokens.")]
    pub embedding_size: usize,

    #[arg(
        long,
        default_value_t = 4,
        help = "The number of transformer heads for each layer block."
    )]
    pub head_count: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "The number of layer blocks for the model."
    )]
    pub layer_count: usize,

    #[arg(
        long,
        default_value_t = 0.001,
        help = "The learning rate to start the training at in the AdamW optimizer."
    )]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value_t = 200,
        help = "The number of steps until a validation loss is calculated."
    )]
    pub validation_interval: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "The batch_size for the valdiation loss evaluations."
    )]
    pub validation_batch: usize,

    #[arg(
        long,
        value_name = "Model-File",
        help = "Generate text from the model provided."
    )]
    pub generate: Option<String>,

    #[arg(
        long,
        default_value_t = 200,
        help = "The number of new tokens to generate with --generate or during training."
    )]
    pub tokens_to_generate: usize,

    #[arg(
        long,
        value_name = "Prompt-Text",
        help = "The starting prompt for the text generation."
    )]
    pub prompt: Option<String>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Temperature for the sampler when generating text."
    )]
    pub temperature: f32,
}

fn main() {
    // setup our logging interface
    let mut builder = env_logger::Builder::new();
    builder.filter(None, log::LevelFilter::Info);
    builder.init();

    // parse in the command line options all derived fromt he Args struct.
    let args = Args::parse();

    // if this option is provided, then prepare the binary datasets
    // based on the text source file provided.
    if args.prepare_dataset.is_some() {
        if let Err(err) = prepare_datase_charlevel(&args) {
            error!("{err}");
            exit(1);
        }
    // we can also prepare datasets with the gpt2 tokenizer
    // though training and inference aren't quite supported yet
    } else if args.prepare_dataset_gpt2.is_some() {
        if let Err(err) = prepare_datase_gpt2(&args) {
            error!("{err}");
            exit(1);
        }
    } else if args.train {
        if args.training_dataset.is_none()
            || args.validation_dataset.is_none()
            || args.vocab_metadata.is_none()
        {
            error!("For training, it is required to pass the --training-dataset, --validation-dataset and --vocab-metadata parameters.");
            exit(1);
        }

        if let Err(err) = run_training_ui(&args) {
            error!("{err}");
            exit(1);
        }
    } else if args.generate.is_some() {
        if args.vocab_metadata.is_none() {
            error!("For text generation, it is required to pass the --vocab-metadata parameter.");
            exit(1);
        }

        if let Err(err) = run_generation(&args) {
            error!("{err}");
            exit(1);
        }
    }
}

fn run_generation(args: &Args) -> Result<()> {
    let vocab_meta = load_vocab_meta(args.vocab_metadata.as_ref().unwrap().as_str()).unwrap();

    // setup some Candle stuff to setup the device to use
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();
    #[cfg(not(feature = "cuda"))]
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    let mut model_filepath = args.generate.as_ref().unwrap().clone();
    if model_filepath.ends_with(".safetensors") {
        if let Some(dot_pos) = model_filepath.rfind('.') {
            model_filepath.truncate(dot_pos);
        }
    }
    let mut model = NanoGptModel::load_from_safetensors(model_filepath.as_str())?;
    info!("Parameter count: {}", model.parameter_count());

    // setup a vector with just the newline token to be used for text generation tensors
    // unless a prompt is specified on the command line.
    let prompt_tokens: Vec<u32> = if let Some(prompt) = &args.prompt {
        vocab_meta.encode_string(prompt.as_str())?
    } else {
        vocab_meta.encode_string("\n")?
    };
    let prompt_length = prompt_tokens.len();

    let mut rng = if args.seed > 0 {
        rand::rngs::StdRng::seed_from_u64(args.seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    let new_text_tensor = Tensor::from_vec(prompt_tokens, (1, prompt_length), &device)?;
    let new_text_tensor = model.generate(
        &mut rng,
        new_text_tensor,
        args.tokens_to_generate,
        args.temperature,
    )?;
    let new_text_tokens = new_text_tensor
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    let generated_text = vocab_meta.decode_tokens(new_text_tokens)?;

    println!("Generated text:\n{}", generated_text);
    Ok(())
}

fn run_training_ui(args: &Args) -> Result<()> {
    // load our prepared datasets with the generated vocabulary data
    let vocab_meta = load_vocab_meta(args.vocab_metadata.as_ref().unwrap().as_str()).unwrap();
    let training_tokens =
        load_training_tokens(args.training_dataset.as_ref().unwrap().as_str()).unwrap();
    let validation_tokens =
        load_training_tokens(args.validation_dataset.as_ref().unwrap().as_str()).unwrap();
    let mut rng = if args.seed > 0 {
        rand::rngs::StdRng::seed_from_u64(args.seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // setup some Candle stuff to setup the device to use
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();
    #[cfg(not(feature = "cuda"))]
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    // load the datasets into tensors and put them into the batcher
    let mut training_batcher = DatasetBatcher::new(&rng);
    let total_tokens = training_tokens.len();
    let train_ds_name = "train".to_string();
    let training_tensor = Tensor::from_vec(training_tokens, total_tokens, &device).unwrap();
    training_batcher.add(train_ds_name.clone(), training_tensor, total_tokens);
    let valid_ds_name = "valid".to_string();
    let total_validation_tokens = validation_tokens.len();
    let validation_tensor =
        Tensor::from_vec(validation_tokens, total_validation_tokens, &device).unwrap();
    training_batcher.add(
        valid_ds_name.clone(),
        validation_tensor,
        total_validation_tokens,
    );

    // setup a vector with just the newline token to be used for text generation tensors
    let mut newline_token: Vec<u32> = Vec::new();
    newline_token.push(vocab_meta.encode_char('\n')?);

    // pull all the training parameters out of the args for convenience
    let epochs = args.steps;
    let num_of_batches = args.batch_size;
    let block_size = args.block_size;
    let n_embed = args.embedding_size;
    let num_of_hidden_layers = args.layer_count;
    let num_of_heads = args.head_count;
    let learning_rate: f64 = args.learning_rate;
    let validation_interval = args.validation_interval;
    let validation_batch = args.validation_batch;

    // create the bigram model
    let mut model = NanoGptModel::new(
        num_of_heads,
        num_of_hidden_layers,
        vocab_meta.vocab_size,
        block_size,
        n_embed,
        learning_rate,
        &device,
    )?;
    info!("Datasets are loaded up and the model has been created.");
    info!("Parameter count: {}", model.parameter_count());

    // this will keep track of our training data over time
    let mut results: Vec<TrainingStepData> = Vec::new();
    let mut ui_loss_data: Vec<(f64, f64)> = Vec::new();
    let mut ui_val_loss_data: Vec<(f64, f64)> = Vec::new();

    // setup UI
    stdout().execute(EnterAlternateScreen)?;
    enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let fps = 5.0;
    let tick_rate = Duration::from_millis((1_000.0 / fps) as u64);

    // training loop
    let mut is_paused = false;
    let mut trained_epochs: usize = 0;
    let mut generated_text: Option<String> = None;
    let mut msg_to_user: Option<String> = None;
    let mut last_tick = Instant::now();
    loop {
        let is_training = trained_epochs < epochs;

        if last_tick.elapsed() >= tick_rate {
            terminal.draw(|f| {
                let area = f.size();
                let verticals =
                    Layout::vertical([Constraint::Max(1), Constraint::Min(2), Constraint::Max(1)]);
                let [help_area, chart_area, status_area] = verticals.areas(area);

                let p = Paragraph::new(
                    "Commands: (q)uit | (p)ause | (g)enrate text | (s)ave model files",
                )
                .fg(Color::Gray);
                f.render_widget(p, help_area);

                // render the line chart for the loss
                let chart_ds = vec![
                    Dataset::default()
                        .name("Training loss".italic())
                        .marker(symbols::Marker::Braille)
                        .style(Style::default().fg(Color::Yellow))
                        .graph_type(GraphType::Line)
                        .data(&ui_loss_data),
                    Dataset::default()
                        .name("Validation loss".italic())
                        .marker(symbols::Marker::Dot)
                        .style(Style::default().fg(Color::LightCyan))
                        .data(&ui_val_loss_data),
                ];

                let chart = Chart::new(chart_ds)
                    .block(
                        ratatui::widgets::Block::default()
                            .title("Training Loss".cyan().bold())
                            .borders(Borders::ALL),
                    )
                    .x_axis(
                        Axis::default()
                            .title("Step")
                            .style(Style::default().fg(Color::Gray))
                            .labels(vec!["0".bold(), format!("{}", epochs).bold()])
                            .bounds([0.0, epochs as f64]),
                    )
                    .y_axis(
                        Axis::default()
                            .title("Loss")
                            .style(Style::default().fg(Color::Gray))
                            .labels(vec![
                                "0".bold(),
                                "1".bold(),
                                "2".bold(),
                                "3".bold(),
                                "4".bold(),
                                "5".bold(),
                            ])
                            .bounds([0.0, 6.0]),
                    );
                f.render_widget(chart, chart_area);

                // add a status line for the current step data
                let last_few_results = results.iter().rev().take(10);
                let last_few_len = last_few_results.len();
                if last_few_len > 0 {
                    let last_few_mean =
                        last_few_results.map(|r| r.loss).sum::<f32>() / last_few_len as f32;

                    let p = Paragraph::new(
                        format!(
                            "Current step: {} -> loss: {}",
                            trained_epochs, last_few_mean
                        )
                        .fg(Color::Gray),
                    );
                    f.render_widget(p, status_area);
                }

                // if we had generated sample text, show it in a popup box
                if let Some(sample_text) = generated_text.as_ref() {
                    let block = ratatui::widgets::Block::default()
                        .title("Generated text (Esc to close)")
                        .borders(Borders::ALL);
                    let area = centered_rect(40, 40, area);
                    let p = Paragraph::new(sample_text.as_str())
                        .block(block)
                        .wrap(Wrap { trim: true });
                    f.render_widget(Clear, area);
                    f.render_widget(p, area);
                } else if let Some(msg) = msg_to_user.as_ref() {
                    let block = ratatui::widgets::Block::default()
                        .title("Information (Esc to close)")
                        .borders(Borders::ALL);
                    let area = centered_rect(40, 20, area);
                    let p = Paragraph::new(msg.as_str())
                        .block(block)
                        .wrap(Wrap { trim: true });
                    f.render_widget(Clear, area);
                    f.render_widget(p, area);
                }
            })?;

            last_tick = Instant::now();
        }

        // if we're not actively training, then limit to fps
        let timeout = if !is_training {
            tick_rate.saturating_sub(last_tick.elapsed())
        } else {
            Duration::from_secs(0)
        };
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if key.code == KeyCode::Char('q') {
                        // quit the trainer
                        break;
                    } else if key.code == KeyCode::Esc {
                        // if there's generated text being shown, throw it away
                        generated_text = None;
                        // if there was a message to the user, toss that too
                        msg_to_user = None;
                    } else if key.code == KeyCode::Char('p') {
                        // pause the training
                        is_paused = !is_paused;
                    } else if key.code == KeyCode::Char('g') {
                        // generate a new string and display it
                        is_paused = true; // pause the training if it was running
                        let new_text_tensor =
                            Tensor::from_vec(newline_token.clone(), (1, 1), &device)?;
                        let new_text_tensor = model.generate(
                            &mut rng,
                            new_text_tensor,
                            args.tokens_to_generate,
                            args.temperature,
                        )?;
                        let new_text_tokens = new_text_tensor
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<u32>()
                            .unwrap();
                        generated_text = Some(vocab_meta.decode_tokens(new_text_tokens)?);
                    } else if key.code == KeyCode::Char('s') {
                        // test saving the model out
                        let file_stem = format!("model_step_{}", trained_epochs);
                        model.save_to_file(file_stem.as_str())?;

                        // save the model config json
                        let training_json = serde_json::to_string_pretty(&results)?;
                        std::fs::write(format!("{}.training_log.json", file_stem), training_json)?;

                        // leave a message to the user
                        msg_to_user = Some(format!("Wrote the model file to {}.safetensors and saved config and training log too.", file_stem));
                    }
                }
            }
        }

        // if we're not actively training, then limit to fps
        if is_training && !is_paused {
            // finally, get around to doing a training step...
            let now = std::time::Instant::now();
            let training_loss =
                model.train(&train_ds_name, 1, num_of_batches, &mut training_batcher)?;
            let step_time_ms = now.elapsed().as_millis();
            ui_loss_data.push((trained_epochs as f64, training_loss as f64));

            // check to see if it's validation time
            let mut validation_loss = None;
            if trained_epochs % validation_interval == 0 {
                validation_loss = Some(model.eval_loss(
                    &valid_ds_name,
                    validation_batch,
                    &mut training_batcher,
                )?);
                ui_val_loss_data.push((trained_epochs as f64, validation_loss.unwrap() as f64));
            }

            // update the results
            results.push(TrainingStepData {
                step: trained_epochs,
                loss: training_loss,
                duration_ms: step_time_ms,
                validation_loss,
            });

            trained_epochs += 1;
        }
    }

    // reset UI after running
    stdout().execute(LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

/// helper function to create a centered rect using up certain percentage of the available rect `r`
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(r);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(popup_layout[1])[1]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainingStepData {
    step: usize,

    loss: f32,

    duration_ms: u128,

    #[serde(skip_serializing_if = "Option::is_none")]
    validation_loss: Option<f32>,
}

// This represents the vocabulary metadata used when preparing the dataset.
// When using the character-level tokenizer process, both `itos` and `stoi`
// will be embedded as hashmaps. When using the GPT2 tokenizer, while such
// a table could be constructed, it's just nicer to use the Tokenizer class
// instead. Instead of using the field directly to encode and decode, it
// should now be preferred to use the utility functions that deail with
// abstracting the two possibilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct VocabMeta {
    vocab_size: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    itos: Option<HashMap<u32, char>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    stoi: Option<HashMap<char, u32>>,

    #[serde(skip)]
    tokenizer: Option<Tokenizer>,
}
impl VocabMeta {
    fn encode_char(&self, c: char) -> Result<u32> {
        if let Some(stoi) = &self.stoi {
            Ok(stoi[&c])
        } else if let Some(tokenizer) = &self.tokenizer {
            let result = tokenizer
                .encode(c.to_string().as_str(), false)
                .map_err(anyhow::Error::msg)?;
            Ok(result.get_ids()[0]) // we're encoding characters, so just pull the first id
        } else {
            Err(anyhow!(
                "Vocabulary Metadata doesn't have a stoi hashmap or a loaded tokenizer."
            ))
        }
    }

    fn encode_string(&self, s: &str) -> Result<Vec<u32>> {
        if let Some(stoi) = &self.stoi {
            let result: Vec<u32> = s.chars().map(|c| stoi[&c]).collect();
            Ok(result)
        } else if let Some(tokenizer) = &self.tokenizer {
            let result = tokenizer.encode(s, false).map_err(anyhow::Error::msg)?;
            Ok(result.get_ids().to_vec()) // we're encoding characters, so just pull the first id
        } else {
            Err(anyhow!(
                "Vocabulary Metadata doesn't have a stoi hashmap or a loaded tokenizer."
            ))
        }
    }

    fn decode_tokens(&self, tokens: Vec<u32>) -> Result<String> {
        if let Some(itos) = &self.itos {
            Ok(tokens
                .iter()
                .filter_map(|t| itos.get(&t))
                .collect::<String>())
        } else if let Some(tokenizer) = &self.tokenizer {
            Ok(tokenizer
                .decode(&tokens, false)
                .map_err(anyhow::Error::msg)?)
        } else {
            Err(anyhow!(
                "Vocabulary Meatadata doesn't have a itos hashmap or a loaded tokenizer."
            ))
        }
    }
}

// returns a vector of tuples with each tuple being a percentage and a byte vector; one tuple for each file
// in the source directory, or just one tuple if the source is a file.
//
// the percentage f32 is the suggested percentage of source bytes to use in the dataset. this is
// 1.0 (100%) for single file datasets. if the source path is a directory, then this function will
// check each subdirectory which should be name "NNN_identifier" (E.g. "50_shakespeare") where the NNN
// portion indicates a percentage the source files inside that folder should occupy in the final dataset.
fn get_source_bytes(source: &String) -> Result<Vec<(f32, Vec<u8>)>> {
    let source_path = PathBuf::from(source);
    if !source_path.exists() {
        return Err(anyhow!(
            "The source file specified doesn't exist: {:?}",
            source
        ));
    }

    let mut list_of_buffers = Vec::new();

    if source_path.is_file() {
        let mut source_bytes = Vec::new();
        let mut source_file = File::open(&source_path)?;
        source_file.read_to_end(&mut source_bytes)?;
        list_of_buffers.push((1.0, source_bytes));
    } else {
        // first load all of the source bytes
        let mut source_bytes = Vec::new();

        for entry in read_dir(&source_path)? {
            let entry = entry?;
            let subdir_path = entry.path();
            if subdir_path.is_dir() {
                let mut name = subdir_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                if let Some(offset) = name.find("_") {
                    name.truncate(offset);
                    if let Ok(pct_number) = name.parse::<u32>() {
                        let pct = pct_number as f32 / 100.0;
                        info!(
                            "Using percentage of {} for the sub-directory {:?}",
                            pct, subdir_path
                        );
                        // at this point we got a sub directory with a percentage
                        let mut dir_bytes: Vec<u8> = Vec::new();
                        for file_entry in read_dir(&subdir_path)? {
                            let file_entry = file_entry?;
                            let file_path = file_entry.path();
                            if file_path.is_file() {
                                info!("Loading bytes for file: {:?}", file_path);
                                let mut source_bytes = Vec::new();
                                let mut source_file = File::open(&file_path)?;
                                source_file.read_to_end(&mut source_bytes)?;
                                dir_bytes.append(&mut source_bytes);
                            }
                        }
                        source_bytes.push((pct, dir_bytes, subdir_path));
                    }
                }
            }
        }

        // print out some totals for what we've found so far
        let mut total_source_bytes = 0;
        for (pct, sb, p) in &source_bytes {
            let sb_size = sb.len();
            total_source_bytes += sb_size;
            info!(
                "Processed {:?} and made the following data streams:",
                source_path
            );
            info!("  ({:.3}) {:?}: total bytes: {}", pct, p, sb_size);
        }
        info!("Total bytes: {}", total_source_bytes);

        // scale the percentages based off of byte representation in the totals.
        let mut total_pct = 0.0;
        for (pct, sb, _) in &mut source_bytes {
            let sb_size = sb.len();
            let multiplier = total_source_bytes as f64 / sb_size as f64;
            let scaled = *pct * multiplier as f32;
            *pct = scaled;
            total_pct += scaled;
        }
        // normalize them
        for (pct, _, _) in &mut source_bytes {
            let scaled = *pct / total_pct;
            *pct = scaled;
        }

        // with the total bytes we can now clamp down the data based on the requested percentages
        info!("Creating a final datastream for the dataset:");
        for (pct, sb, p) in &source_bytes {
            let mut bytes_to_take = (pct * total_source_bytes as f32) as usize;
            bytes_to_take = bytes_to_take.min(sb.len());
            info!("  ({:.3}) {:?}: taking bytes: {}", pct, p, bytes_to_take);
            let bs = sb[..bytes_to_take].to_vec();
            list_of_buffers.push((*pct, bs));
        }
    }

    Ok(list_of_buffers)
}

fn prepare_datase_charlevel(args: &Args) -> Result<()> {
    if args.prepare_dataset.as_ref().is_none() {
        return Err(anyhow!(
            "To prepare the dataset, you must pass the source-text parameter."
        ));
    }

    let start_time = std::time::Instant::now();

    // load up the text from the source file into one buffer to tokenize.
    let source_bytes = get_source_bytes(args.prepare_dataset.as_ref().unwrap())?;

    let training_split = 0.9;
    let mut training_string = String::new();
    let mut validation_string = String::new();
    for (_pct, bs) in source_bytes {
        // to avoid splitting on a bad UTF boundary, we're going to convert
        // to a string first, and then split that based on length.
        let mut train_chunk = String::from_utf8(bs)?;
        let validation_chunk =
            train_chunk.split_off((train_chunk.len() as f32 * training_split) as usize);
        training_string.push_str(&train_chunk);
        validation_string.push_str(&validation_chunk);
    }

    // get all the unique characters that occur in this text
    let mut chars: HashSet<char> = HashSet::new();
    for c in training_string.chars() {
        chars.insert(c);
    }
    for c in validation_string.chars() {
        chars.insert(c);
    }
    let vocab_size = chars.len();
    info!("Unique characters: {:?}", chars);
    info!("Vocabulary size: {}", vocab_size);

    // build the lookup tables; we're going to use u32 datatypes here to match the gpt2 version
    let mut stoi: HashMap<char, u32> = HashMap::new();
    let mut itos: HashMap<u32, char> = HashMap::new();
    for (i, c) in chars.iter().enumerate() {
        stoi.insert(*c, i as u32);
        itos.insert(i as u32, *c);
    }

    info!(
        "Training string length: {} ; Validation string length: {}",
        training_string.len(),
        validation_string.len()
    );

    // use the lookup tables to encode the strings into ids.
    let training_ids: Vec<u32> = training_string
        .chars()
        .map(|c| *stoi.get(&c).unwrap())
        .collect();
    let validation_ids: Vec<u32> = validation_string
        .chars()
        .map(|c| *stoi.get(&c).unwrap())
        .collect();
    info!(
        "Training token count: {} ; Validation token count: {}",
        training_ids.len(),
        validation_ids.len()
    );

    // a little extra massaging is necessary to reliably encode u32 as a big endian byte stream
    let mut training_bytes = Vec::new();
    for tb_u16 in training_ids {
        let b = tb_u16.to_be_bytes();
        training_bytes.extend_from_slice(&b);
    }
    let mut validation_bytes = Vec::new();
    for vb_u16 in validation_ids {
        let b = vb_u16.to_be_bytes();
        validation_bytes.extend_from_slice(&b);
    }
    info!(
        "Training token bytes: {} ; Validation token bytes: {}",
        training_bytes.len(),
        validation_bytes.len()
    );

    // finally write out the bytes to the respective files
    let mut source_path = PathBuf::from(args.prepare_dataset.as_ref().unwrap());
    if source_path.is_file() {
        source_path.pop();
    }
    let mut training_filepath = source_path.clone();
    training_filepath.push("train.bin");
    let mut training_file = File::create(&training_filepath)?;
    training_file.write_all(&training_bytes)?;

    let mut validation_filepath = source_path.clone();
    validation_filepath.push("val.bin");
    let mut validation_file = File::create(&validation_filepath)?;
    validation_file.write_all(&validation_bytes)?;

    // write out the vocab metadata
    let mut vocab_filepath = source_path.clone();
    vocab_filepath.push("vocab.json");
    let vocab_data = VocabMeta {
        vocab_size,
        stoi: Some(stoi),
        itos: Some(itos),
        tokenizer: None,
    };
    let vocab_data_json = serde_json::to_string_pretty(&vocab_data)?;
    std::fs::write(&vocab_filepath, vocab_data_json)?;

    info!(
        "Dataset prepare successfully in {:.2} seconds.",
        start_time.elapsed().as_secs_f32()
    );
    info!("Training bytes: {:?}", training_filepath);
    info!("Validation bytes: {:?}", validation_filepath);
    info!("Vocabulary metadata: {:?}", vocab_filepath);

    Ok(())
}

// This takes the filepath provided in the `prepare_dataset_gpt2` argument, loads it as a text
// file, splits the text into training and validation sets, encodes the incomding text
// with the `gpt2` tokenizer, and then writes the resulting bytes out to new files
// with the extensions "train.bin" and "val.bin".
fn prepare_datase_gpt2(args: &Args) -> Result<()> {
    if args.prepare_dataset_gpt2.as_ref().is_none() {
        return Err(anyhow!(
            "To prepare the dataset, you must pass the source-text parameter."
        ));
    }

    let start_time = std::time::Instant::now();

    // load up the text from the source file into one buffer to tokenize.
    let source_bytes = get_source_bytes(args.prepare_dataset_gpt2.as_ref().unwrap())?;

    let training_split = 0.9;
    let mut training_string = String::new();
    let mut validation_string = String::new();
    for (_pct, bs) in source_bytes {
        // to avoid splitting on a bad UTF boundary, we're going to convert
        // to a string first, and then split that based on length.
        let mut train_chunk = String::from_utf8(bs)?;
        let validation_chunk =
            train_chunk.split_off((train_chunk.len() as f32 * training_split) as usize);
        training_string.push_str(&train_chunk);
        validation_string.push_str(&validation_chunk);
    }

    info!(
        "Training string length: {} ; Validation string length: {}",
        training_string.len(),
        validation_string.len()
    );

    // with the `http` feature, tokenizers will pull the encoder down
    // from huggingface automatically.
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let training_encodings = tokenizer
        .encode(training_string, false)
        .map_err(anyhow::Error::msg)?;
    let validation_encodings = tokenizer
        .encode(validation_string, false)
        .map_err(anyhow::Error::msg)?;
    info!(
        "Training token count: {} ; Validation token count: {}",
        training_encodings.get_ids().len(),
        validation_encodings.get_ids().len()
    );

    let vocab_size = tokenizer.get_vocab_size(true);

    // a little extra massaging is necessary to reliably encode u32 as a big endian byte stream
    let mut training_bytes = Vec::new();
    for tb_u32 in training_encodings.get_ids() {
        let b = tb_u32.to_be_bytes();
        training_bytes.extend_from_slice(&b);
    }
    let mut validation_bytes = Vec::new();
    for vb_u32 in validation_encodings.get_ids() {
        let b = vb_u32.to_be_bytes();
        validation_bytes.extend_from_slice(&b);
    }
    info!(
        "Training token bytes: {} ; Validation token bytes: {}",
        training_bytes.len(),
        validation_bytes.len()
    );

    // finally write out the bytes to the respective files
    let mut source_path = PathBuf::from(args.prepare_dataset_gpt2.as_ref().unwrap());
    if source_path.is_file() {
        source_path.pop();
    }
    let mut training_filepath = source_path.clone();
    training_filepath.push("train.bin");
    let mut training_file = File::create(&training_filepath)?;
    training_file.write_all(&training_bytes)?;

    let mut validation_filepath = source_path.clone();
    validation_filepath.push("val.bin");
    let mut validation_file = File::create(&validation_filepath)?;
    validation_file.write_all(&validation_bytes)?;

    // write out the vocab metadata
    let mut vocab_filepath = source_path.clone();
    vocab_filepath.push("vocab.json");
    let vocab_data = VocabMeta {
        vocab_size,
        stoi: None,
        itos: None,
        tokenizer: Some(tokenizer),
    };
    let vocab_data_json = serde_json::to_string_pretty(&vocab_data)?;
    std::fs::write(&vocab_filepath, vocab_data_json)?;

    info!(
        "Dataset prepare successfully in {:.2} seconds.",
        start_time.elapsed().as_secs_f32()
    );
    info!("Training bytes: {:?}", training_filepath);
    info!("Validation bytes: {:?}", validation_filepath);
    Ok(())
}

fn load_vocab_meta(fp: &str) -> Result<VocabMeta> {
    let f = File::open(fp)?;
    let bf = BufReader::new(f);
    let mut meta: VocabMeta = serde_json::from_reader(bf)?;

    // if we don't have one of the hashmaps for the vocabulary,
    // then load the tokenizer
    if meta.stoi.is_none() || meta.itos.is_none() {
        meta.tokenizer = Some(Tokenizer::from_pretrained("gpt2", None).unwrap());
    }

    Ok(meta)
}

fn load_training_tokens(fp: &str) -> Result<Vec<u32>> {
    let f = File::open(fp)?;
    let mut bf = BufReader::new(f);
    let mut data: Vec<u32> = Vec::new();
    let mut buffer = [0; 4];

    loop {
        let bytes_read = bf.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        assert!(bytes_read == 4);

        let value = u32::from_be_bytes(buffer);
        data.push(value);
    }

    Ok(data)
}
