import asyncio
import openai
import random
from tqdm import tqdm
import numpy as np
import json
import os
import time
from dotenv import load_dotenv
from datasets import Dataset, load_from_disk
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import utils

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

class DatasetGenerator:
    dataset: Dataset

    def __init__(self, dataset, config, verbose=False):
        self.dataset = dataset
        self.config = config
        self.verbose = verbose
        self.sem = asyncio.Semaphore(5) # change 10 to control the number of concurrent requests

    def run(self):
        min_convo_length = self.config['min_convo_length']
        max_convo_length = self.config['max_convo_length']
        num_samples = self.config['num_samples']
        return asyncio.run(self._async_run(min_convo_length, max_convo_length, num_samples))

    async def _async_run(self, min_convo_length, max_convo_length, num_samples):
        convo_lengths = np.arange(min_convo_length, max_convo_length + 1)
        num_buckets = len(convo_lengths)

        base_samples_per_bucket = num_samples // num_buckets
        extra_samples = num_samples % num_buckets

        samples_per_bucket = [base_samples_per_bucket + (1 if i < extra_samples else 0) for i in range(num_buckets)]

        results = []
        tasks = []
        for convo_length, n in zip(convo_lengths, samples_per_bucket):
            tasks.extend([self._generate_convo(convo_length) for _ in range(n)])

        pbar = tqdm(total=len(tasks), desc="Generating Completions")
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            pbar.update(1)

        pbar.close()

        return Dataset.from_dict(utils.group_by_key(results))

    async def _generate_convo(self, convo_length):
        async with self.sem:
            samples = list(self.dataset)[:3]
        
            keys = list(samples[0].keys())
        
            keys_str = "[" + ", ".join(keys) + "]"

            samples_str = ""
            for i, obj in enumerate(samples, start=1):
                samples_str += f"Example {i}:\n{obj}\n\n"

            word1 = random.choice(utils.random_verb)
            word2 = random.choice(utils.random_modifier)

            prompt = f"""
            Generate a sample conversation between two people. The conversation should be {convo_length * 2} turns long. Each conversation is also associated with some metadata. The entire conversation object is a JSON with the following keys: {keys_str}. 

            Examples:
            {samples_str}

            Now generate a JSON object with {convo_length * 2} turns of conversation and the associated metadata. The conversation should include the following words: {word1} and {word2}.
            """

            return await self._gen_completion([{"role": "user", "content": prompt}])

    def log_retry_attempt(retry_state):
        if retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            print(f"Retrying _gen_completion due to {exception}. Retry attempt {retry_state.attempt_number}.")


    @retry(
        stop=stop_after_attempt(10), 
        wait=wait_exponential(multiplier=2, min=4, max=120), 
        retry=(retry_if_exception_type(json.decoder.JSONDecodeError) | retry_if_exception_type(openai.error.RateLimitError)) | retry_if_exception_type(openai.error.Timeout) | retry_if_exception_type(openai.error.APIError) | retry_if_exception_type(openai.error.APIConnectionError),
    )
    async def _gen_completion(self, messages):
        completion = await openai.ChatCompletion.acreate(model="gpt-3.5-turbo", messages=messages)
        print("Completion:", completion)
        try:
            return json.loads(completion.choices[0].message.content)
        except json.decoder.JSONDecodeError:
            print("JSONDecodeError")
            return "JSONDecodeError"