from datetime import datetime
import transformers
from datasets import load_from_disk
from transformers import GPT2TokenizerFast, GPTNeoForCausalLM, GPTNeoConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import wandb
import argparse

parser = argparse.ArgumentParser(
                    prog='gpt-neo-baseline',
                    description='Trains baseline gpt neo',
                    epilog='the 4 arguments are: Tokenized data folder | Output folder | Tokenizer folder | wandb key txt')

parser.add_argument('data_dir')
parser.add_argument('out_dir')
parser.add_argument('tok_dir')
parser.add_argument('wandb_key')
args = parser.parse_args()

transformers.set_seed(123)
with open( args.wandb_key) as f: # args.wandb_key
    wandb.login(key = f.read())

#dataset = load_from_disk("../data/TinyStories")
tokenized_dataset = load_from_disk(args.data_dir) # "../data/TokenizedTinyStories"

config = GPTNeoConfig(

    # number of tokens in the vocabulary
    vocab_size = 10_000,
    # embedding size (vector length) of each token
    hidden_size=384,
    # we thus have an embedding block of 512 x 10'000 parameters

    # maximum sequence length, though inputs longer than `hidden_size` will be iteratively processed
    max_position_embeddings = 512,

    # number of transformer blocks. div by 2 for attention_types
    num_layers=2,
    # for global and local attention (GPT-Neo-specific)
    attention_types=[[["global", "local"], 1]],

    num_heads=4,     # attention heads
    window_size=384, # for local attention (GPT-Neo-specific)

    intermediate_size=384*8, # size of 'up-projection' layer in FFN
)

tokenizer = GPT2TokenizerFast.from_pretrained(args.tok_dir, model_max_length=config.max_position_embeddings) # ../10k-tok

assert tokenizer.model_max_length == config.max_position_embeddings
assert tokenizer.vocab_size == config.vocab_size

config.pad_token_id = tokenizer.pad_token_id

model = GPTNeoForCausalLM(config=config)

assert len(tokenized_dataset['train'][0]['input_ids']) == config.max_position_embeddings
tokenized_dataset['train'][0]['input_ids'][-10:]

train_dataset, eval_dataset = tokenized_dataset['train'], tokenized_dataset['validation']

batch_size = 16 # TinyStories claims 80, but I am training locally on my poor M1 Air
num_train_epochs = 1  # TinyStories doesn't mention
gradient_accumulation_steps = 16 # TinyStories claims 16

lr = 5e-4 # TinyStories claims 5e-4, higher values are preferable for smaller models

_train_steps = len(train_dataset) // (batch_size * gradient_accumulation_steps)
eval_steps = _train_steps // 10 # evaluate every 10% of training steps

model_name = f'gpt-neo-baseline-{str(datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))}'

training_args = TrainingArguments(

    seed       = 42,
    use_cpu    = False, # use GPU if available (not necessarily faster on laptops, but Apple's MPS have good support)
    output_dir = f'{args.out_dir}/{model_name}',  #./results/models/

    # NOTE: training params
    learning_rate    = lr,
    num_train_epochs = num_train_epochs,
    # Use a smaller batch size to fit into GPU RAM.
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    # You should aim to have the same amount of samples per acc step, in all of your experiments!
    # so, if you increase batch_size, decrease gradient_accumulation_steps by the same factor.
    gradient_accumulation_steps = gradient_accumulation_steps,

    # NOTE: Evaluation params
    # wandb is great for tracking experiments, it will even (try to) save your code nowadays
    evaluation_strategy = 'steps',
    eval_steps = eval_steps,
    save_steps = eval_steps,

    logging_first_step=True,
    logging_steps=eval_steps,
    report_to  = 'wandb',
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

wandb.init(project='gpt-neo', name=model_name, config=training_args)
trainer.train()
trainer.save_model(f'{args.out_dir}/{model_name}')