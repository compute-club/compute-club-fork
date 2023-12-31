"""
This file is a modified version of the example file from the Ray documentation

Reference here: https://docs.ray.io/en/latest/ray-air/examples/gptj_deepspeed_fine_tuning.html 

Just aiming to get a better feel of the ergonomics of working with Ray. 
"""


# Standard libraries
import os

# Libraries for data processing
import numpy as np
import pandas as pd
import torch

# Ray related libraries
import ray
import ray.data
from ray.air import session
from ray.air.config import ScalingConfig
from ray.data.preprocessors import BatchMapper, Chain
from ray.train.huggingface import TransformersTrainer, TransformersPredictor

# Huggingface related libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPTJForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    PreTrainedTokenizerBase
)
from transformers.utils.logging import disable_progress_bar, enable_progress_bar

# Other libraries
import evaluate

model_name = "EleutherAI/gpt-j-6B"
use_gpu = True
num_workers = 1
cpus_per_worker = 8

def split_text(batch: pd.DataFrame) -> pd.DataFrame:
    text = list(batch["text"])
    flat_text = "".join(text)
    split_text = [
        x.strip()
        for x in flat_text.split("\n")
        if x.strip() and not x.strip()[-1] == ":"
    ]
    return pd.DataFrame(split_text, columns=["text"])


def tokenize(batch: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["text"]),
        truncation=True,
        max_length=block_size,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)


def trainer_init_per_worker(train_dataset, eval_dataset=None, **config):
    # Use the actual number of CPUs assigned by Ray
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )
    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 2)
    warmup_steps = config.get("warmup_steps", 0)
    learning_rate = config.get("learning_rate", 0.00002)
    weight_decay = config.get("weight_decay", 0.01)

    print("Preparing training arguments")
    training_args = TrainingArguments(
        "output",
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        save_strategy="no",
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        label_names=["input_ids", "attention_mask"],
        num_train_epochs=epochs,
        push_to_hub=False,
        disable_tqdm=True,  # declutter the output a little
        fp16=True,
        gradient_checkpointing=True,
    )
    disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model")

    model = GPTJForCausalLM.from_pretrained(model_name, use_cache=False)
    model.resize_token_embeddings(len(tokenizer))

    print("Model loaded")

    enable_progress_bar()

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
      print("computing metrics")
      logits, labels = eval_pred
      pndredictions = np.argmax(logits, axis=-1)
      return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    return trainer


def train_model():
  model_name = "EleutherAI/gpt-j-6B"
  use_gpu = True
  num_workers = 1
  cpus_per_worker = 8

  splitter = BatchMapper(split_text, batch_format="pandas")
  tokenizer = BatchMapper(tokenize, batch_format="pandas")

  trainer = TransformersTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    trainer_init_config={
        "batch_size": 64,  # per device
        "epochs": 1,
    },
    scaling_config=ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"GPU": 1, "CPU": cpus_per_worker},
    ),
    datasets={"train": ray_datasets["train"], "evaluation": ray_datasets["validation"]},
    preprocessor=Chain(splitter, tokenizer),
  )

  results = trainer.fit()
  checkpoint = results.checkpoint
  return checkpoint 

def predict_with_model(checkpoint):
  checkpoint.set_preprocessor(None)

  prompts = pd.DataFrame(["Romeo and Juliet", "Romeo", "Juliet"], columns=["text"])

  # Predict on the head node.
  predictor = TransformersPredictor.from_checkpoint(
      checkpoint=checkpoint,
      task="text-generation",
      torch_dtype=torch.float16 if use_gpu else None,
      device_map="auto",
      use_gpu=use_gpu,
  )
  prediction = predictor.predict(
      prompts,
      do_sample=True,
      temperature=0.9,
      min_length=32,
      max_length=128,
  )
  print(prediction)


if __name__ == '__main__':
  ray.init(ignore_reinit_error=True, num_cpus=26)
  print("Loading tiny_shakespeare dataset")
  current_dataset = load_dataset("tiny_shakespeare")
  print(current_dataset)

  ray_datasets = ray.data.from_huggingface(current_dataset)

  block_size = 512

  checkpoint = train_model()
  predict_with_model(checkpoint)