import shutil
import asyncio
import openai
import random
from tqdm import tqdm
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset, load_from_disk

openai.api_key = os.environ["OPENAI_API_KEY"]

# from chat_data_pipeline.pipeline import logger
# from chat_data_pipeline import utils
# from chat_data_pipeline.minhash_deduplication import deduplicate

random_verb = ['eating', 'playing', 'sleeping', 'walking', 'running', 'drinking', 'singing', 'dancing', 'talking']
random_modifier = ['together', 'outside', 'at night', 'in the morning', 'in the afternoon', 'in the evening', 'at the park', 'at the zoo']


class DatasetGenerator:
    dataset: Dataset

    def __init__(
            self,
            dataset,
            config,
            verbose=False
    ):
        self.dataset = dataset
        self.config = config
        self.verbose = verbose

    # def run(self):
    #     # self._clean_dataset()
    #     # self._filter_dataset()
    #     # if self.deduplication_config.get("do_deduplication", False):
    #     #     self._deduplicate_dataset()
    #     # return self.dataset
    #     asyncio.run(self.generate_function(self.config., convo_length))

    def run(self):
        min_convo_length = self.config['min_convo_length']
        max_convo_length = self.config['max_convo_length']
        num_samples = self.config['num_samples']
        return asyncio.run(self.async_run(min_convo_length, max_convo_length, num_samples))

    async def async_run(self, min_convo_length, max_convo_length, num_samples):
        convo_lengths = np.arange(min_convo_length, max_convo_length + 1)
        num_buckets = len(convo_lengths)

        base_samples_per_bucket = num_samples // num_buckets
        extra_samples = num_samples % num_buckets

        samples_per_bucket = [base_samples_per_bucket + (1 if i < extra_samples else 0) for i in range(num_buckets)]

        print('samples per bucket: ', samples_per_bucket)

        # Create a progress bar
        pbar = tqdm(total=num_samples, desc="Generating synthetic conversations")

        for convo_length, n in zip(convo_lengths, samples_per_bucket):
            await self.generate_convos_for_length(n, convo_length)

            # Update the progress bar
            pbar.update(n)

        # Close the progress bar
        pbar.close()

        return ['hello']
    
    async def generate_convos_for_length(self, n, convo_length):
        tasks = [self.generate_convo(convo_length) for _ in range(n)]
        await asyncio.gather(*tasks)

    async def generate_convo(self, convo_length):
        # print('dataset: ', list(self.dataset))
        samples = list(self.dataset)[:3]
        print('samples: ', samples)
        keys = list(samples[0].keys())
        print('keys: ', keys)
        keys_str = "[" + ", ".join(keys) + "]"

        samples_str = ""
        for i, obj in enumerate(samples, start=1):
            samples_str += f"Example {i}:\n{obj}\n\n"

        word1 = random.choice(random_verb)
        word2 = random.choice(random_modifier)

        prompt = f"""
        Generate a sample conversation between two people. The conversation should be {convo_length * 2} turns long. Each conversation is also associated with some metadata. The entire conversation object is a JSON with the following keys: {keys_str}. 

        Examples:
        {samples_str}

        Now generate a JSON object with {convo_length * 2} turns of conversation and the associated metadata. The conversation should include the following words: {word1} and {word2}.
        """
        print('prompt: ', prompt)
        completion = await openai.ChatCompletion.acreate(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])

        print('completion', completion)
        # print(completion.choices[0].message.content)
        return completion


    # def _clean_dataset(self):
    #     if len(self.cleaners) > 0:
    #         self.dataset = utils.run_cleaner(self.dataset, self.column_name, self.cleaners)
    #     return self.dataset

    # def _filter_dataset(self):
    #     for filter_func in self.filters:
    #         dataset_length = len(self.dataset)
    #         ids = range(dataset_length)
    #         self.dataset = self.dataset.add_column("ids", ids)
    #         filtered_dataset = utils.run_filter(
    #             dataset=self.dataset,
    #             column_name=self.column_name,
    #             filter_func=filter_func,
    #             dry_run=self.dry_run
    #         )
    #         self._print_filter_logs(filtered_dataset, filter_func.__name__)
    #         self.dataset = filtered_dataset.remove_columns("ids")

    #     return self.dataset

    # def _deduplicate_dataset(self):
        dataset_length = len(self.dataset)
        ids = range(dataset_length)
        self.dataset = self.dataset.add_column("ids", ids)
        # need to save to disk and load again, otherwise it is very slow
        target_directory = "./.temp-dataset"
        shutil.rmtree(target_directory, ignore_errors=True)
        try:
            self.dataset.save_to_disk(target_directory)
        except PermissionError:
            logger.info("Can not save dataset, nothing changed. Skipping...")
        gc.collect()
        self.dataset = load_from_disk(target_directory)
        deduplicated_ds = deduplicate(
            self.dataset,
            column=self.column_name,
            **self.deduplication_config.get("args", {})
        )
        self.dataset = deduplicated_ds.remove_columns("ids")
        return self.dataset

    # def _print_filter_logs(self, filtered_dataset, filter_name):
    #     original_length = len(self.dataset)
    #     filtered_length = len(filtered_dataset)
    #     reduced_percent = round(100 * (original_length - filtered_length) / original_length, 2)
    #     logger.info(
    #         f'Filtered by {filter_name} on {self.column_name}:\n'
    #         f'{reduced_percent}% = {original_length - filtered_length:,} samples reduced\n'
    #         f'New dataset size: {filtered_length:,} rows'
    #     )
    #     if self.verbose:
    #         utils.print_sample_dropped_examples(self.dataset, filtered_dataset, num_samples=10)


