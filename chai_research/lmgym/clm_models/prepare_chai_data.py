from chai_guanaco import PygmalionFormatter
from datasets import load_dataset
import re
import pdb
from functools import partial
from clm_models.custom import training_utils as utils
from clm_models.custom import training_utils as utils
from clm_models.custom import tokenization

dataset = load_dataset("ChaiML/100_example_conversations")

def format_row(formatter, row, tokenizer):
  memory = formatter.memory_template.format(bot_name=row['name'], memory=row['memory'])
  prompt = formatter.prompt_template.format(prompt=row['prompt'])
  response = formatter.response_template.format(bot_name=row['name'])

  tokenized_conversation_length = len(
    tokenizer(row['conversation'], padding="max_length", max_length=block_size, truncation=False, return_token_type_ids=False)['input_ids']
  )

  return memory + prompt + row['conversation'] + "\n" + response


formatter = PygmalionFormatter()

formatted_data = []
model_args, data_args, train_args = utils.get_parsed_arguments()
tokenizer = utils.get_tokenizer(model_args)
block_size = utils._get_block_size(data_args, tokenizer)
tokenize_function = partial(
  utils.tokenization.tokenize_function,
  tokenizer=tokenizer,
  block_size=block_size,
  train_to_probs=data_args.train_to_probs
)

# Iterate through each row of the 'train' split of the dataset
# How do we handle the case that the conversation is too large 
# To fit within the context window. 

for row in dataset['train']:
  formatted_row = format_row(formatter, row)
  inputs = tokenizer(formatted_row, padding="max_length", max_length=1024, truncation=True, return_token_type_ids=False)
  pdb.set_trace()

# Now `formatted_data` is a list of formatted strings for each row in the 'train' split of the dataset
