import argparse
import os

import numpy as np
import torch

from utils.analysis.word_similarity.ranking import *
from utils.analysis.shared import load_config
from utils.data import CorpusLoader

MARKER_LEN = 85

def get_dictionary(dataset_path):
    corpus = CorpusLoader.load(dataset_path)
    return corpus.dictionary


def get_word_vectors(embedding, dictionary, normalize=True):
    """
    embedding: a tensor (dictionary_len, nemb)
    returns a dict with key being word 
    and value being word vector
    """
    word_vectors = {}
    for idx in range(0, dictionary.__len__()):
        word = dictionary.idx2word[idx]
        vector = embedding[idx].numpy()
        if normalize:
            word_vectors[word] = vector/np.linalg.norm(vector)
    return word_vectors

def get_embedding(model_path, embedding_key='base_model.encoder.weight'):
    model_state = torch.load(
        os.path.join(model_path, 'model.pt'), map_location='cpu'
    )
    return model_state[embedding_key]


def evaluate(word_vecs, args, meta_stats=False):
    if meta_stats:
        print("="*MARKER_LEN)
        print("%6s%20s%15s%15s" % (
            'Serial', 'Sim Data', 'Num Pairs', 'Not found'
            )
        )
        print("="*MARKER_LEN)
    word_sim_dir = args.similarity_data_dir
    rhos = []
    for i, filename in enumerate(os.listdir(word_sim_dir)):
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open(os.path.join(word_sim_dir, filename),'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in word_vecs and word2 in word_vecs:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(
                    word_vecs[word1], word_vecs[word2]
                )
            else:
                not_found += 1
            total_size += 1
        if meta_stats:    
            print("%6s%20s%15s%15s" % (
                str(i+1), filename, str(total_size), str(not_found)
                )
            )
        rhos.append(spearmans_rho(
                assign_ranks(manual_dict), assign_ranks(auto_dict)
            )
        )
    return rhos

def start(args):
    dictionary = get_dictionary(args.model_dataset)
    print("="*MARKER_LEN)
    print("Dataset is %s" % (args.model_dataset))
    print("="*MARKER_LEN)
    configs = load_config(args.config_csv)
    meta_stats = True
    rhos = []
    for config in configs:
        word_vectors = get_word_vectors(
            get_embedding(config['path']), dictionary
        )
        rhos.append(evaluate(word_vectors, args, meta_stats))
        meta_stats = False
    print("="*MARKER_LEN)
    print("Rho values for different models are below ..")
    print("="*MARKER_LEN)
    model_names = [cfg['name'] for cfg in configs]
    formatter = "{:>6}" + "{:>15}" * len(model_names)
    col_names = ['Serial'] + model_names
    print(formatter.format(*col_names))
    print("="*MARKER_LEN)
    s_no = 1
    for rho in zip(*rhos):
        formatter = "{:6d}" + "{:15.4f}" * len(rho)
        rho = [s_no] + list(rho)
        print(formatter.format(*rho))
        s_no += 1
    print("="*MARKER_LEN)
        




