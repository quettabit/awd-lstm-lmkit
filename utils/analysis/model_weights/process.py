import os
import torch
import statistics

from utils.analysis.shared import load_config

def get_model_weight_stats(model_path):
    model_state = torch.load(
        os.path.join(model_path, 'model.pt'), map_location='cpu'
    )
    model_stats = []
    for layer_name in model_state:
        


def print_stats(stats):
    pass

def compare_model_weights(configs):
    result_set = []
    for config in configs:
        result = {}
        result['stats'] = get_model_weight_stats(config['path'])
        break
        result['name'] = config['name']
    print_stats(result_set)

def start(args):
    configs = load_config(args.config_csv)
    compare_model_weights(configs)