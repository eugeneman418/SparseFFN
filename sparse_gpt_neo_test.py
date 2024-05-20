import transformers
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from sparse_gpt_neo import SparseGPTNeoForCausalLM
from configuration_sparse_gpt_neo import SparseGPTNeoConfig, SparsityType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

dataset = load_from_disk("./data/TinyStories")
tokenized_dataset = load_from_disk("./data/TokenizedTinyStories")

config = SparseGPTNeoConfig(

    # number of tokens in the vocabulary
    vocab_size = 10_000,
    # embedding size (vector length) of each token
    hidden_size=256,
    # we thus have an embedding block of 512 x 10'000 parameters

    # maximum sequence length, though inputs longer than `hidden_size` will be iteratively processed
    max_position_embeddings = 512,

    # number of transformer blocks. div by 2 for attention_types
    num_layers=2,
    # for global and local attention (GPT-Neo-specific)
    attention_types=[[["global", "local"], 1]],

    num_heads=4,     # attention heads
    window_size=384, # for local attention (GPT-Neo-specific)

    sparsity_type=SparsityType.PKM,
    intermediate_size=256*2, # size of 'up-projection' layer in FFN
    num_subkeys=64,
    topk=16,
    num_query_heads=4,
)

tokenize_function = GPT2TokenizerFast.from_pretrained('./10k-tok', model_max_length=config.max_position_embeddings)

assert tokenize_function.model_max_length == config.max_position_embeddings
assert tokenize_function.vocab_size == config.vocab_size

# printing this because of a bug in tokenizers (should be fixed now) https://github.com/huggingface/transformers/issues/26500
print(f'padding token is {tokenize_function.pad_token}')
# HF wasn't saving this nor the tokenizer's pad_token
config.pad_token_id = tokenize_function.pad_token_id

model = SparseGPTNeoForCausalLM(config=config)

print(f'The model has {model.num_parameters():,} parameters.')

train_dataset, eval_dataset = tokenized_dataset['train'], tokenized_dataset['validation']

batch_size = 16 # TinyStories claims 80, but I am training locally on my poor M1 Air
num_train_epochs = 1  # TinyStories doesn't mention
gradient_accumulation_steps = 16 # TinyStories claims 16

lr = 5e-4 # TinyStories claims 5e-4, higher values are preferable for smaller models

_train_steps = len(train_dataset) // (batch_size * gradient_accumulation_steps)
eval_steps = _train_steps // 10 # evaluate every 10% of training steps

model_name = f'{model.num_parameters()//1e6:.1f}M-{config.num_layers}L-{config.num_heads}H-{config.hidden_size}C-{config.intermediate_size}I'

training_args = TrainingArguments(

    seed       = 42,
    use_cpu    = False, # use GPU if available (not necessarily faster on laptops, but Apple's MPS have good support)
    output_dir = f'./results/models/{model_name}',

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
    report_to="none",
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = DataCollatorForLanguageModeling(tokenize_function, mlm=False),
)

# print amount of training steps, and how often the model is evaluated
print(f'''
    training for {num_train_epochs} epochs, {len(train_dataset)} samples
    {batch_size} batch size, {gradient_accumulation_steps} accumulation steps
    gives {_train_steps} training steps.

    evaluating every {eval_steps} steps, {len(eval_dataset)} samples 
    ''')

trainer.train()