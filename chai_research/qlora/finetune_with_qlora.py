import click
from datasets import load_dataset
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset



def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@click.command()
@click.option('--config_path')
def main(config_path):
    config = load_yaml(config_path)
    model_path = config["model_path"]
    input_dataset_path = config["input_dataset_path"]
    output_dataset_path = config["output_dataset_path"]
    verbose = config.get("verbose", False)
    lora_args = config["lora_args"]
    training_args = config["training_args"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map={"":0})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
    r=lora_args["r"], 
    lora_alpha=lora_args["lora_alpha"], 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=lora_args["lora_dropout"], 
    bias=lora_args["bias"], 
    task_type=lora_args["task_type"],
)

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    data = load_dataset(input_dataset_path)
    # data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    # needed for gpt-neo-x tokenizer
    # tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=training_args["per_device_train_batch_size"],
            gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
            warmup_steps=training_args["warmup_steps"],
            max_steps=training_args["max_steps"],
            learning_rate=training_args["learning_rate"],
            fp16=True,
            logging_steps=training_args["logging_steps"],
            optim=training_args["optimizer"],
            output_dir=output_dataset_path,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = verbose  # silence the warnings.
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub()


if __name__ == "__main__":
    main()