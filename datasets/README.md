# Synthetic Dataset Generator

## Getting started

1. Create a virtual environment:

```
$ cd datasets
$ virtualenv -p python3 env
$ source ./env/bin/activate
```

Or if you're like me and use `fish`:
```
$ source ./env/bin/activate.fish
```

2. Install dependencies:
```
$ pip install -r requirements.txt
```

3. Login to HuggingFace:

Make sure you get a token first: <https://huggingface.co/docs/hub/security-tokens>
```
$ huggingface-cli login
```

4. Define `.env`
```
$ cp .env.template .env
```
Then replace the values with your own. For now, you just need your OpenAI API key.

## Generating a dataset
To run an experiment, define a `.yaml` file in `experiments/` with the following syntax:
```
input_dataset_path: "ChaiML/100_example_conversations"
output_dataset_path: "rrustom/synthetic_conversations_test"
verbose: True

generation_config:
  min_convo_length: 1
  max_convo_length: 3
  num_samples: 3
```

- `input_dataset_path`: path to HuggingFace dataset to use as input.
- `output_dataset_path`: path to HuggingFace dataset to save output to (doesn't need to exist yet).
- `min_convo_length`: minimum number of turns for generated conversations
- `max_convo_length`: maximum number of turns for generated conversations
- `num_samples`: number of conversations to generate

The output dataset will have the same features and structure as the input dataset.

Then run:
```
python generate_synthetic_dataset.py --config_path path_to_config.yaml
```

To find the dataset, navigate to `https://huggingface.co/datasets/output_dataset_path`

## Implementation details
- The prompt is seeded with randomly chosen words in an effort to increase dataset diversity
- Sometimes parsing the JSON output from OpenAI fails, so we use an exponential backoff strategy to keep retrying


## Contributing
Just remember to add dependencies to `requirements.txt` if you install anything:
```
$ pip freeze > requirements.txt
```
