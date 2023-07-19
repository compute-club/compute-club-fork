from datasets import load_dataset
import re
import pandas as pd
import os

import pdb 

soda = load_dataset('allenai/soda')
anthropic_hh_rlhf = load_dataset('Anthropic/hh-rlhf')
#share_gpt = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered')
dolly = load_dataset('databricks/databricks-dolly-15k')
shp = load_dataset('stanfordnlp/SHP')
# shp seems irrelevant
gptteacher_instruct = load_dataset('teknium/GPTeacher-General-Instruct')
gptforall_j = load_dataset('nomic-ai/gpt4all-j-prompt-generations')
# gptforall_j seems irrelevant
#chain_of_thoughts = load_dataset("QingyiSi/Alpaca-CoT")
# chain of thoughts errors out? 
gpt_roleplay = load_dataset('IlyaGusev/gpt_roleplay_realm')

def generate_example_from_soda(soda_line): 
    dialog = soda_line['dialogue']
    current_input = ""
    for idx, line in enumerate(dialog[:-1]):
      actor = "Human: "
      if (idx % 2) == 0: 
        actor = "Assistant: "
      current_input += actor + line + "\n"
    current_input += "Assistant: "
    output_text = dialog[-1] + "\n" + '\n'
    return {
       'input_text': current_input,
       'output_text': output_text
    }

def generate_example_from_anthropic_hh_rlhf(anthropic_line):
  # anthropic hh_rlhf is tuples of dialogs, with "chosen" being the one the human chose and
  # rejected being the one they rejected
  # Example: 
  # {
  #   'chosen': '\n\nHuman: What kind of noises did dinosaurs make?\n\nAssistant: Humans and dinosaurs didn’t live at the same time, so it’s really hard to say. The best place to find out what noises dinosaurs made would be\n\nHuman: yes they did\n\nAssistant: to guess, and that would probably require lots of reading and a certain amount of imagination, so we’re not really prepared to do that.\n\nHuman: you cant read\n\nAssistant: You can read?', 
  #   'rejected': '\n\nHuman: What kind of noises did dinosaurs make?\n\nAssistant: Humans and dinosaurs didn’t live at the same time, so it’s really hard to say. The best place to find out what noises dinosaurs made would be\n\nHuman: yes they did\n\nAssistant: to guess, and that would probably require lots of reading and a certain amount of imagination, so we’re not really prepared to do that.\n\nHuman: you cant read\n\nAssistant: there’s a lot of stuff humans don’t know'
  # }
  dialog = anthropic_line['chosen']

  # Splitting the string by "Human:" or "Assistant:"
  result = re.split(r'(Human:|Assistant:)', dialog)

  # Removing empty strings from the result
  result = [s.strip() for s in result]
  actor = "Human: "
  current_input = ""
  for line in result[:-1]:
    if line == '': 
      continue
    elif line == "Human:":
      actor = "Human: "
    elif line == "Assistant:":
      actor = "Assistant: "
    else:
      current_input += actor + line + "\n"
  current_input += "Assistant: "
  output_text = result[-1] + "\n" + '\n'
  return { 
    'input_text': current_input,
    'output_text': output_text
  }


def generate_example_from_dolly(dolly_line):
  return { 
    'input_text': "Human: " + dolly_line['instruction'] + "\n" + "Assistant: ",
    'output_text': dolly_line['response'] + "\n" + '\n'
  }


def generate_example_from_gpt_instructor(gpt_instructor_line):
  # A series of instructions from humans, optionally with input, and the response from the assistant
  # Example:
  # {
  #   'input': 'There was a boy named Romeo and a girl named Juliet. They were from two different families who were enemies. Despite this, they fell in love with each other at their very first meeting.', 
  #   'instruction': 'Translate the excerpt from English to French.', 
  #    'response': "Il y avait un garçon nommé Roméo et une fille nommée Juliette. Ils étaient issus de deux familles différentes qui étaient ennemies. Malgré cela, ils sont tombés amoureux l'un de l'autre dès leur première rencontre."
  # }
  return { 
    'input_text': "Human: " + (gpt_instructor_line['input'] + "\n" + gpt_instructor_line['instruction']).strip() + '\n' + "Assistant: ",
    'output_text': gpt_instructor_line['response'] + "\n" + '\n'
  }

def generate_example_from_gpt_roleplay(gpt_roleplay_line): 
  dialog = gpt_roleplay_line['example_dialogue']
  current_input = ""
  for line in dialog[:-1]:
    actor = 'Human: ' if line['role'] == 'user' else 'Assistant: '
    current_input += actor + line['content'] + "\n"
  current_input += "Assistant: "
  output_text = dialog[-1]['content'] + "\n" + '\n'
  return {
    'input_text': current_input,
    'output_text': output_text
  }


#pdb.set_trace()

datasets = { 
  'soda': {
    'dataset': soda, 
    'dataset_keys_to_use': ['train', 'validation', 'test'], 
    'generate_example': generate_example_from_soda,
  },
  'anthropic_hh_rlhf': {
    'dataset': anthropic_hh_rlhf,
    'dataset_keys_to_use': ['train', 'test'],
    'generate_example': generate_example_from_anthropic_hh_rlhf,
  },
  'dolly': {
    'dataset': dolly,
    'dataset_keys_to_use': ['train'],
    'generate_example': generate_example_from_dolly,
  },
  'gpt_instructor': { 
    'dataset': gptteacher_instruct,
    'dataset_keys_to_use': ['train'],
    'generate_example': generate_example_from_gpt_instructor,
  },
  'gpt_roleplay': {
    'dataset': gpt_roleplay,
    'dataset_keys_to_use': ['en'],
    'generate_example': generate_example_from_gpt_roleplay,
  },
}

save_directory = '/home/ubuntu/chai-research/data'
intermediate_directory = '/home/ubuntu/chai-research/data/intermediate'
if not os.path.exists(intermediate_directory):
  os.makedirs(intermediate_directory)

all_examples = pd.DataFrame({'input_text': [], 'output_text': [], 'dataset': []})
for dataset_name, dataset_info in datasets.items():
  overall_dataset = dataset_info['dataset']
  dataset_keys_to_use = dataset_info['dataset_keys_to_use']
  generate_example = dataset_info['generate_example']
  for dataset_key in dataset_keys_to_use: 
    print(f"Processing {dataset_name} {dataset_key}")
    processed_dataset_save_path = os.path.join(intermediate_directory, dataset_name + '_' + dataset_key + '.csv')
    if os.path.exists(processed_dataset_save_path):
      current_dataset = pd.read_csv(processed_dataset_save_path)
      all_examples = pd.concat([all_examples, current_dataset])
      print(f"Already processed {dataset_name} {dataset_key}")
      continue
    dataset = overall_dataset[dataset_key]
    dataset = dataset.map(generate_example)
    # create a dataframe with the same columns as all_examples
    current_dataset = pd.DataFrame({
      'input_text': dataset['input_text'], 
      'output_text': dataset['output_text'], 
      'dataset': [dataset_name + '_' + dataset_key] * len(dataset)
    })
    current_path = os.path.join(intermediate_directory, dataset_name + '_' + dataset_key + '.csv')
    current_dataset.to_csv(current_path)
    all_examples = pd.concat([all_examples, current_dataset])
  
all_examples.to_csv(os.path.join(save_directory, 'all_examples.csv'))