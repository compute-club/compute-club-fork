# Finetune with QLoRA

## Getting started

1. Create a virtual environment:

```
cd qlora
virtualenv -p python3 env
source ./env/bin/activate
```

Or if you're like me and use `fish`:

```
source ./env/bin/activate.fish
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Login to HuggingFace:

Make sure you get a token first: <https://huggingface.co/docs/hub/security-tokens>

```
huggingface-cli login
```

## Running Training

To finetune a model, define a `.yaml` file in `experiments/` with the following syntax:

```
model_path: "EleutherAI/gpt-j-6b"
input_dataset_path: "AlekseyKorshuk/hh-lmgym-demo"
output_dataset_path: "rrustom/gptj-qlora-peft"
verbose: True

lora_args:
  r: 8
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  warmup_steps: 2
  max_steps: 10
  learning_rate: 2e-4
  fp16: True
  logging_steps: 1
  optimizer: "paged_adamw_8bit"
```

Then run:

```
$ python finetune_with_qlora.py --config_path path_to_config.yaml
```

To find the dataset, navigate to `https://huggingface.co/datasets/output_dataset_path`


## Contributing

Just remember to add dependencies to `requirements.txt` if you install anything:

```
pip freeze > requirements.txt
```
