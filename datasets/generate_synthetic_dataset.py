import os

import click
from datasets import load_dataset, concatenate_datasets

# from chat_data_pipeline.pipeline import logger
import utils
from generator import DatasetGenerator

PAD = 32


@click.command()
@click.option('--config_path')
def main(config_path):
    config = utils.load_yaml(config_path)
    # dataset_paths = [dataset["dataset_path"] for dataset in config["datasets"]]
    input_dataset_path = config["input_dataset_path"]
    output_dataset_path = config["output_dataset_path"]
    verbose = config.get("verbose", False)

    generation_config = config["generation_config"]

    print('generation_config', generation_config)

    dataset = load_dataset(input_dataset_path)

    dataset = dataset['train']


    # dataset = dataset.map(
    #     convert_to_input_output,
    #     batched=True,
    #     num_proc=os.cpu_count(),
    #     remove_columns=list(dataset.features),
    #     desc="Converting to I/O..."
    # )

    # dataset = dataset.map(
    #     add_content_columns,
    #     batched=False,
    #     num_proc=os.cpu_count(),
    #     desc="Adding content column..."
    # )

    preprocessor = DatasetGenerator(
        dataset=dataset,
        config=generation_config,
        verbose=verbose,
    )
    output_dataset = preprocessor.run()

    print('output_dataset', output_dataset)

    # cleaners = utils.get_cleaners_from_config(instruction_config)
    # if len(cleaners) > 0:
    #     logger.warning("Cleaner does not work on instructions. Cleaners set to empty list.")
    # preprocessor = DataPreprocessor(
    #     dataset=dataset,
    #     column_name="instruction",
    #     cleaners=[],
    #     filters=utils.get_filters_from_config(instruction_config),
    #     deduplication_config=instruction_config.get("deduplication", {}),
    #     verbose=verbose,
    # )
    # dataset = preprocessor.run()

    # prepared_dataset_chatml = dataset.map(
    #     convert_to_chatml,
    #     batched=False,
    #     num_proc=os.cpu_count(),
    #     remove_columns=list(dataset.features)
    # )




    # output_dataset = output_dataset.shuffle(seed=42)
    # output_dataset.push_to_hub(output_dataset_path)
    # logger.info(output_dataset)


if __name__ == "__main__":
    main()