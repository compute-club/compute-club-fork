# Import standard libraries
import time
import numpy as np
import pdb
import torch
import os

# Import third party libraries
import evaluate
import transformers
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers.data import DataCollatorForLanguageModeling
import logging
import wandb

wandb.login()

start = time.time()

GPTJ_FINE_TUNED_FILE = "./fine_tuned_models/gpt-j-6B"

os.environ['WANDB_PROJECT'] = 'chai_research'
os.environ['WANDB_WATCH'] = 'all'

print("Loading model")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
model.config.pad_token_id = model.config.eos_token_id

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset")
current_dataset = load_dataset("wikitext", 'wikitext-103-v1')


def tokenize_function(examples):
    current_tokenizer_result = tokenizer(examples["text"], padding="max_length", truncation=True)
    return current_tokenizer_result


print("Splitting and tokenizing dataset")
tokenized_datasets = current_dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count())
small_eval_dataset = tokenized_datasets["validation"].select(range(100))

print("Preparing training arguments")

 # The default of training_args.log_level is passive, so we set log level at info here to have that default.
transformers.utils.logging.set_verbosity_info() 

training_args = TrainingArguments(output_dir=GPTJ_FINE_TUNED_FILE,
                                  gradient_accumulation_steps=32,
                                  per_device_train_batch_size=1,
                                  label_names=['input_ids', 'attention_mask'],  # 'logits', 'past_key_values'
                                  num_train_epochs=1,
                                  report_to=["wandb"],
                                  logging_steps = 1,
                                  eval_steps = 5000,
                                  run_name = 'custom_training',
                                  no_cuda=False)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

wandb.finish()