import yaml
import pandas as pd


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def group_by_key(list_of_dicts):
    df = pd.DataFrame(list_of_dicts)
    result_dict = df.to_dict('list')
    return result_dict

random_verb = ['eating', 'playing', 'sleeping', 'walking', 'running', 'drinking', 'singing', 'dancing', 'talking']
random_modifier = ['together', 'outside', 'at night', 'in the morning', 'in the afternoon', 'in the evening', 'at the park', 'at the zoo']