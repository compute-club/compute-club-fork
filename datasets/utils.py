import random
import re
from functools import partial
from collections import Counter

from datasets import load_dataset, Dataset, concatenate_datasets
import tqdm
import yaml


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)