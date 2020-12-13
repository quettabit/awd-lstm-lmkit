import os

import numpy as np
import torch

from sklearn.decomposition import PCA

from utils.analysis.shared import load_config

def get_emb(model_state):
    # config can have appropriate model_state key in future.
    return model_state['top_model.decoder.weight'].numpy()

def n_pcs_for_r_variance(r, explained_variance_ratio):
    pcs = 0
    tot = 0
    for evr in explained_variance_ratio:
        tot += evr
        pcs += 1
        if tot >= r:
            return pcs


def calc_emb_variance_from_emb(emb, ratios):
    pca = PCA(n_components=emb.shape[1])
    pca.fit_transform(emb)
    result = {}
    for r in ratios:
        result[r] = n_pcs_for_r_variance(r, pca.explained_variance_ratio_)
    return result

def calc_emb_variance_from_sv(model_state, ratios):
    sv = model_state['top_model.decoder.S'].sort(descending=True)[0].numpy()
    # https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/decomposition/_pca.py#L461
    sv[sv < 0.0] = 0.0 # tail components shall have non-negative values.
    explained_variance = (sv ** 2) / (sv.shape[0] - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    result = {}
    for r in ratios:
        result[r] = n_pcs_for_r_variance(r, explained_variance_ratio)
    return result
    


def start(args):
    configs = load_config(args.config_csv)
    for config in configs:
        model_state = torch.load(
            os.path.join(config['path'], 'model.pt'), 
            map_location='cpu'
        )
        ratios = [0.75, 0.95]
        if 'spectrum' in config and config['spectrum'] == 'yes':
            emb_variance = calc_emb_variance_from_sv(
                model_state, ratios
            )
        else:
            emb_variance = calc_emb_variance_from_emb(
                get_emb(model_state), ratios
            )
        print(config['name'], emb_variance)
