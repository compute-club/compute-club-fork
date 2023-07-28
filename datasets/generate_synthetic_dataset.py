import click
from datasets import load_dataset

import utils
from generator import DatasetGenerator

@click.command()
@click.option('--config_path')
def main(config_path):
    config = utils.load_yaml(config_path)
    input_dataset_path = config["input_dataset_path"]
    output_dataset_path = config["output_dataset_path"]
    verbose = config.get("verbose", False)
    generation_config = config["generation_config"]

    dataset = load_dataset(input_dataset_path)

    dataset = dataset['train']

    preprocessor = DatasetGenerator(
        dataset=dataset,
        config=generation_config,
        verbose=verbose,
    )
    output_dataset = preprocessor.run()

    output_dataset = output_dataset.shuffle(seed=42)
    output_dataset.push_to_hub(output_dataset_path)


if __name__ == "__main__":
    main()