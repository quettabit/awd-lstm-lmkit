import argparse
import glob
import math
import os
import shutil
import sys
import torch

from argparse import Namespace
from collections import Counter
from tabulate import tabulate

from utils.analysis.shared import load_config
from utils.analysis.qualitative.model import load_model, get_args
from utils.data import CorpusLoader
from utils.main import batchify, get_batch, get_model_args, parse_model_result



def mock_args(exp_id):
    return Namespace(
        path="<your_path>%s" % (exp_id),
        cuda=False,
        bptt=70
    )

def get_path(paths, tensor_type):
    '''
    res:
        mostly it can be [x]
        but occasionally it can be [x, y, z]
    '''
    res = []
    for path in paths:
        if tensor_type in path:
            res.append(path)
    return res


def merge_tensor_paths(exp_ids, tensors):
    '''
        exp_ids[0] is the base from which 
        data, target, and pred is accumulated.
        
        pred from all other exp_ids[i] are accumulated.
        
        merged: {
            'seq_end': [data, target, pred_1, pred_2 ..] 
            .
            .
        }
        
        the list in 'seq_end' can be in any order
    '''
    merged = {}
    merged = tensors[exp_ids[0]]
    for i in range(1, len(exp_ids)):
        for seq_end, paths in tensors[exp_ids[i]].items():
            pred_path = get_path(paths, 'pred')[0]
            merged[seq_end].append(pred_path)
    return merged


def load_tensor_paths(exp_ids, root_path):
    '''
    {
        'exp_id':{
            'seq_len': [inp, out, pred]
        }
        .
        .
    }
    '''
    tensors = {}
    for exp_id in exp_ids:
        tensors[exp_id] = {}
        exp_path = root_path % (exp_id) 
        #print(exp_path)
        tensor_paths = glob.glob("%s/*.pt" % exp_path)
        for path in tensor_paths:
            #print(path)
            seq_end = int(path.split('.')[0].split('_')[-1])
            #print(seq_end)
            if seq_end not in tensors[exp_id]:
                tensors[exp_id][seq_end] = [path]
            else:
                tensors[exp_id][seq_end].append(path)
    return tensors

# save data, target and preds in temp directory 
def save_tensors(seq_num, args, data_source, model, exp_id,
                    root_path, model_ids, batch_size=1):
    root_path = root_path % (exp_id)
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path)
    
    model.eval()
    total_loss = 0
    hidden = model.base_model.init_hidden(batch_size)
    with torch.no_grad():
        #for i in range(0, 210, args.bptt):
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            if seq_num == 0:
                # as data and targets are same for all exp_ids. Save only once.
                torch.save(targets, "%s/targets_%s.pt" % (root_path, i))
                torch.save(data, "%s/data_%s.pt" % (root_path, i))
            targets = targets.view(-1)
            log_prob, hidden = parse_model_result(
                model(
                    **get_model_args(
                        get_args(model_ids[exp_id]), 
                        data, hidden, None, False
                    )
                ),
                None, False
            )
            vals, idxs = torch.topk(log_prob, 5, dim=1)
            torch.save((vals, idxs), "%s/preds_%s.pt" % (root_path, i))

def get_exp_id_from_pred_path(exp_ids, pred_path):
    for exp_id in exp_ids:
        if exp_id in pred_path:
            return exp_id
    return ''

def table_print(merged_paths, corpus, human_name, exp_ids, file_path):
    #print(merged_paths)
    keys = sorted(merged_paths)
    #print(keys)
    with open(file_path, 'w+') as fp:
        for key in keys:
            data_path = get_path(merged_paths[key], 'data_')[0]
            #print(data_path)
            #print('meauhhh')
            target_path = get_path(merged_paths[key], 'targets_')[0]
            data = torch.load(data_path)
            target = torch.load(target_path)
            pred_paths = get_path(merged_paths[key], 'preds_')
            headers = ['word', 'true output']
            preds = []
            for pred_path in pred_paths:
                #print(pred_path)
                exp_id = get_exp_id_from_pred_path(exp_ids, pred_path)
                #print('exp id is ..')
                #print(exp_id)
                headers.append("%s output" % (human_name[exp_id]))
                preds.append(torch.load(pred_path))
            print_seq(data, target, preds, corpus.dictionary, headers, fp)

def get_word_and_prob(pred, dictionary, r, topk=5):
    result = []
    for i in range(topk):
        result.append(dictionary.idx2word[pred[1][r][i]])
        result.append(math.exp(pred[0][r][i].item()))
        # math.exp used to convert log probs back into probs
    return result

def print_seq(data, target, preds, dictionary, headers, fp):
    """
    preds is a tuple
    preds[0] is the log prob val
    preds[1] is the vocabulary index
    """
    nr = data.size(0)
    nc = data.size(1) 
    # nr = bsz and nc = 1 for test data
    for c in range(nc):
        inp = []
        exp_op = []
        act_op = []
        for _ in range(len(preds)):
            act_op.append([])
        for r in range(nr):
            inp.append(dictionary.idx2word[data[r][c]])
            exp_op.append(dictionary.idx2word[target[r][c]])
            for idx, pred in enumerate(preds):
                act_op[idx].append(
                    "%s(%.2f),\n%s(%.2f),\n%s(%.2f),\n%s(%.2f),\n%s(%.2f)" % (
                        tuple(get_word_and_prob(pred, dictionary, r))
                    )
                )
        print(tabulate(
                list(zip(inp, exp_op, *act_op)),
                headers=headers, showindex='always', tablefmt="grid"
            ),
            file=fp
        )


def table_comparison(args, configs):

    root_path = "<your_path>%s"
    corpus = CorpusLoader.load(args.data)

    human_name = {}
    model_ids = {}
    exp_ids = []

    for config in configs:
        exp_id = config['exp_id']
        exp_ids.append(exp_id)
        human_name[exp_id] = config['human_name']
        model_ids[exp_id] = config['model_id']
        
    for idx, exp_id in enumerate(exp_ids):
        args = mock_args(exp_id)   
        val_data = batchify(corpus.test, 1, args)
        model_state = torch.load(
            os.path.join(args.path, 'model.pt'), map_location='cpu'
        )
        model = load_model(model_ids[exp_id], model_state)
        save_tensors(idx, args, val_data, model, exp_id, root_path, model_ids)
        print("tensors saved for %s .." % (exp_id))
    tensor_paths = load_tensor_paths(exp_ids, root_path)
    print("tensor paths loaded ..")
    #print(tensor_paths)
    merged_paths = merge_tensor_paths(exp_ids, tensor_paths)
    print("tensor paths merged ..")
    #print(merged_paths)
    #print(root_path)
    table_print(merged_paths, corpus,
        human_name, exp_ids, file_path=root_path % ('outputs.txt'))
    print("table comparison printed to file ..")


def start(args):
    configs = load_config(args.config_csv)
    table_comparison(args, configs)
