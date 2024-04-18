# Nanogpt-Rusted

A Rust implementation of [Karpathy's nanogpt](https://github.com/karpathy/nanoGPT) using 
[Candle](https://github.com/huggingface/candle). For inspiration, you can watch [his youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY) that goes over the implementation from scratch in Python.

Essentially, this is a 'from-scratch' implementation of a GPT-2 like model architexture,
with a decoder-only transformer, that can be pre-trained on consumer level hardware at 
'small' sizes. And everything is written in Rust.

Check out what you can manage with a 618k parameter model tokenized by characters:

![picture of a terminal showing the training user interface](https://github.com/tbogdala/nanogpt-rusted/blob/main/assets/Screenshot_618k_parameters.png)

## Features

* Can tokenize text files to prepare for training
* Fancy terminal UI for training loop that graphs the training and validation losses in real-time
* Can generate text at any point in the training ('g' key)
* Can save the model files for text inference via command line ('s' key)
* Supports generating text from trained models via command line


## Caveats with this implementation

* The code is biased towards being verbose and hopefully commented well enough as this was
  primarily a training exercise for the author. 
* Focus was spent on making this run on one GPU only and training everything from scratch.
* Flash attention isn't implemented.
* The `metal` feature currently will not work because `candle_nn::ops::dropout` doesn't support
  accelleration with metal in Candle; cpu training on MacOS works however. T_T
* Resuming training is not yet supported.


# How to run

The overall steps needed are:

1. Download, or otherwise create, a text file to be used as a training source
2. Prepare the source text file
3. Train the model


## Downloading the source data

To replicate the original nanogpt repository, you can download the shakespeare text file like this:

```bash
mkdir data
cd data
wget wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakes.txt
```

A copy of this file has been included in the repository for reference.


## Preparing the source data

In the root project folder, you can run the project with the command line options necessary
to take the incoming text data, encode it and then write out the binary files. To prepare
a dataset for character-level tokenization, use the following command:

```bash
cargo run -- --prepare-dataset "data/shakes.txt"
```

The character-level tokenization is faster for the model training process, which is helpful
when playing around with paramters and implementation features, but ultimately will yield
text that's non-sensical.

To tokenize the incoming data with the GPT2 tokenizer and run the training process with a vocabulary
of approximately ~50k tokens instead of 65, prepare the dataset with the following command:

```bash
cargo run -- --prepare-dataset-gpt2 "data/shakes.txt"
```


## Training the model

After the data has been prepared into the binary tokenized files and the vocabulary has been
created, you can run the training process. A variety of command-line arguments support this feature.
To do a basic training, at a minimum supply a command-line like this:

```bash
cargo run --release -- \
--train \ 
--training-dataset "data/shakes.train.bin" \
--validation-dataset "data/shakes.val.bin" \
--vocab-metadata "data/shakes.vocab.json" \
```

Additionally the following parameters will be useful:

* `--seed`: The seed to use for the training run; if 0, a random seed will be used.
* `--steps`: The number of steps to do for training in total.
* `--batch-size`: How many batches of the training data to run with each step.
* `--block-size`: How big the 'max context' should be for each block layer.
* `--embedding-size`: How big the embedding size should be for each token.
* `--head-count`: The number of attention heads per block; MUST divide evenly into `--embedding-size`.
* `--layer-count`: The total number of layer blocks for the model.
* `--learning-rate`: The initial learning rate to set the optimizer (AdamW) at.
* `--validation-interval`: How many steps to train before executing a validation loss test.
* `--validation-batch`: The batch size of data to run as a validation loss test.

A lot of the defaults are aimed at the lower settings Karpathy starts out with in nanogpt,
but if you want to run a slightly larger training for a 618k parameter model, try this:

```bash
cargo run --release --features cuda,cudnn -- --train --training-dataset "data/shakes.train.bin" --validation-dataset "data/shakes.val.bin" --vocab-metadata "data/shakes.vocab.json" --steps 15000 --batch-size 32 --block-size 64 --embedding-size 128 --head-count 4 --layer-count 3 --validation-batch 32
```

## Text generation

If you've saved the model that you trained (currently bound to the 's' key), you should see
a new `model_step_*` set of files: one `.safetensors` file for the model, one `.config.json` file
that holds the configuration of the model, and then a `.training_log.json` file with the stepwise
training and validation loss values.

To generate text, you can use a different set of command line parameters. The vocabulary metadata
still needs to be passed in as well as the 'file_stem' for the model file. For the above training
session resulting in a 15000 step model file, this command-line will generate more text.

```bash
cargo run --release --features cuda,cudnn -- --vocab-metadata "data/shakes.vocab.json" --generate "model_step_15000" --seed 0 --tokens-to-generate 1000 --temperature 0.7
```

For my training results, this produces the following text using the character-level tokenization model:

```
The is have of the graves of his and me.

QUEEN ELIZABETH:
What I can thy ving Keep.

AUFIDIUS:
Why, have tempats of they well,
Nor and the fillowe bed comment thee
And but the roath, and and your pass again!

First Servant:
What, my lord heaven advent!

LADY AURENCE:
Now thou faven you would of the barks;
Thou heart was be we earth. Comment.

AUFIDIUS:
If shall land?

LUCIO:
No, if Lord.

SICINIUS:
What honour the constress of this revery;
Be the king the day ble them and the petter to me,
That are not his for and be they at me I come,
to such trease the be king to be no not,
And thou this a doss man, there to hunto madam,
And the true sounds thee the flatted that good of the breather batter,
And in report
Than blad and here to much world an the povery
The king of my lord! I am to and like be should,
And what, I will all not back of the ract to herears
Dost in this had kill the sone,--
That have dread my brother prove first.

BRAKENTIO:
Why, rear I would disconded of thee.

ESCALUS:
```

With the same parameters, but using a GPT2 tokenizer which yields a 13.5M parameter model, generating 300 new tokens
yields the following non-cherry-picked result:

```
I Senator:
He shall be made you, sir,
Which you have your voices, you have given us all
my son: I have been no more
Than to the vantage of our general:
Withdraw yourselves, sir, he's sentenced; and they say,
say, but the poor gentleman, he hath carried all all
biting, and a gentleman born to't.

AUTOLYCUS:
So that is that:
He has a friar, sir, he made a man of his
man's great, and he can warrant for his
virtues his own kinsman: but he shall be his countrymen
out of the table freedom of a quarter.

First Gentleman:
I tell you, sir, sir, sir, sir, I warrant him,
and my friend.

First Gentleman:
O, how is it like a deal, indeed stolen from the world,
As he doth a waking, a loathed Mercury,
That Angelo hath kept a judge of a man,
That case to crush down the ground.

Third Gentleman:
But, wherefore, as he is the child?

Third Gentleman:
Heaven became of an hour a man, a wagoner-on
Which heareth that way to be so.

First Gentleman:
Why, if my brother, such delight
to steal themselves as made the deed.
```

## Notes

* VRAM seems to spike to around 18gb used when using the baby GPT settings from nanogpt:
  (block_size = 256, batch_size = 64, n_layer = 6, n_head = 6, n_embed = 384, lr = 1e-3, max_iters = 5000).
* My 4090 currently appears to get about 2 steps/s while training the 'baby GPT' settings,
  with `--features cuda, cudnn` enabled.
* At more modest settings, for a 2.4M parameter model, it only uses ~4 GB on my GPU.
  (block_size = 128, batch_size = 32, n_layer = 3, n_head = 8, n_embed = 256, lr = 1e-3, max_iters = 20000).
  The final loss was 1.448624 (last validation was 1.5603111 at step 19800). Each step took around ~80ms
  on my 4090.
* Using the 618k parameter settings (block_size = 64, batch_size =32, n_layer = 3, n_head = 4, n_embed = 128, lr = 1e-3, max_iters = 15000)
  but with the GPT2 tokenizer made a 13.5M parameter model and yielded a final loss around ~3.1 (last valdiation loss was 5.21). 
  Each step took about 210ms.
* The attention heads still do all their work in separate tensors and are implemented
  naively while making sure the core of the implementation is working and does not implement
  the trick where they can make up a 4d vector for efficiency.

-steps 15000 --batch-size 32 --block-size 64 --embedding-size 128 --head-count 4 --layer-count 3 --validation-batch 32
## License

MIT licensed to match [Karpathy's nanogpt](https://github.com/karpathy/nanoGPT).
