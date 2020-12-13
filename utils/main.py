import math
import random
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.shared import pkg_map


def deterministic_cudnn():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_random(seed, cuda):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if cuda:
        torch.cuda.random.manual_seed(seed)
        # set torch.cuda.random.manual_seed_all(seed) if you are using multi-GPU.
        # similarly add a corresponding line within continue_random()
        # do save them ofcourse within state 
        deterministic_cudnn()

def continue_random(state, cuda):
    random.setstate(state['random']['python'])
    np.random.set_state(state['random']['numpy'])
    torch.random.set_rng_state(state['random']['torch_cpu'])
    if cuda:
        torch.cuda.random.set_rng_state(state['random']['torch_gpu'])
        deterministic_cudnn()

def get_random_state():
    return {
        'python': random.getstate(), #random.setstate(state)
        'numpy': np.random.get_state(),
        'torch_cpu': torch.random.get_rng_state(),
        'torch_gpu': torch.cuda.get_rng_state()
    }

def get_state(epoch):
    return {
        'epoch': epoch,
        'random': get_random_state()
    }


def parallelize_module(module, args):
    parallel_module = module
    if torch.cuda.is_available() and args.cuda:
        if args.single_gpu:
            parallel_module = module.cuda()
        else:
            parallel_module = nn.DataParallel(module, dim=1).cuda()
    return parallel_module

def get_matrices(model, data_source, batch_size, args):
    if args.cuda and torch.cuda.is_available():
        log_prob_all = torch.cuda.FloatTensor(0, args.ntoken) 
        f_logits_all = torch.cuda.FloatTensor(0, args.ntoken)
    else:
        log_prob_all = torch.FloatTensor(0, args.ntoken)
        f_logits_all = torch.FloatTensor(0, args.ntoken)
    # beware that torch.size() of above tensors results [0]
    # also .nelement() return 0
    
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.rnn_type == 'QRNN':
        model.base_model.reset()
    hidden = model.base_model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            targets = targets.view(-1)
            # output should be logprob tm_result['output'] is logprob
            log_prob, hidden, f_logits = parse_model_result(
                model(
                    **get_model_args(args, data, hidden, None, False, True)
                ),
                None, False, True
            )
            if 'penn' in args.data and f_logits is not None:
                # f_logits is not availble for mos, sigsoftmax models. 
                # hence this check.
                # also ,it is only available for PTB
                f_logits_all = torch.cat((f_logits_all, f_logits.data))

            
            # sticking to logprob for better comparison with literature 
            log_prob_all = torch.cat((log_prob_all, log_prob.data))


            if 'wikitext-2' in args.data:
                # cannot store complete matrix on GPU. 
                # so instead of all contexts, considers only ~ ntoken number 
                # of contexts
                if log_prob_all.size(0) > log_prob_all.size(1):
                    break

            hidden = repackage_hidden(hidden)
    return log_prob_all, f_logits_all
   
def press_rank(sv, dim_sum):
    """
    sv is a 1-d numpy array
    0.5*sqrt(m+n+1.)*w[0]*eps according to Press et. al Numerical recipes book
    dim_sum = m+n
    max(sv) = sv[0] as sv is sorted in decreasing order
    """
    tol = 0.5 * np.sqrt(dim_sum + 1) * sv[0] * np.finfo(sv.dtype).eps
    return len([x for x in sv if x > tol])

def save_sv(sv, path, sv_prefix, comet):
    sv_path = os.path.join(path, 'misc', "%s_sv" % (sv_prefix))
    np.save(sv_path, sv)
    comet.log_asset("%s.npy" % sv_path)

def calc_rank(model, data_source, batch_size, args, sv_prefix, comet):
    log_prob_all, f_logits_all = get_matrices(
        model, data_source, batch_size, args
    )
    ans = {}
    print(log_prob_all.size())
    if log_prob_all.size(0) != 0:
        sv = np.linalg.svd(log_prob_all.cpu().numpy(), compute_uv=False)
        save_sv(sv, args.save, "%s_log_prob" % (sv_prefix), comet)
        ans['log_prob_press'] = press_rank(sv, sum(log_prob_all.size()))
    else:
        ans['log_prob_press'] = 0

    print(f_logits_all.size())
    if f_logits_all.size(0) != 0:
        sv = np.linalg.svd(f_logits_all.cpu().numpy(), compute_uv=False)
        save_sv(sv, args.save, "%s_f_logits" % (sv_prefix), comet)
        ans['f_logits_press'] = press_rank(sv, sum(f_logits_all.size()))
    else:
        ans['f_logits_press'] = 0
    return ans


def exp(x):
    try:
        return math.exp(x)
    except:
        return float('inf')



def parameter_counts(model, criterion):
    params = {
        'base': 0,
        'top': 0,
        'criterion': 0,
        'total': 0
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            for key in params.keys():
                if key in name:
                    x = param.data.nelement()
                    params[key] += x
                    params['total'] += x
                    break
    for param in criterion.parameters():
        if param.requires_grad:
            x = param.data.nelement()
            params['criterion'] += x
            params['total'] += x
        
    return params


def param_group_len(top_model_params):
    total_len = 0
    for param_group in top_model_params:
        total_len += len(param_group['params'])
    return total_len

def regroup_top_model_params(named_params, args):
    if args.top_model == 'plif_smax':
        plif_params = []
        other_params = []
        for name, param in named_params:
            if param.requires_grad:
                if 'top_model' in name and 'plif_w' in name:
                    plif_params.append(param)
                elif 'top_model' in name:
                    other_params.append(param)
        return [
            {
                'params': other_params,
                'weight_decay': args.wdecay
            },
            {
                'params': plif_params,
                'weight_decay': args.wdecay,
                'lr': args.plif_smax_lr
            }
        ]
    elif args.top_model == 'sc_smax':
        # decoder_bias is included in tied_params in case of sc_smax
        return []
    else:
        top_model_params = []
        for name, param in named_params:
            if param.requires_grad:
                if 'top_model' in name:
                    top_model_params.append(param)
        return [
            {
                'params': top_model_params,
                'weight_decay': args.wdecay
            }
        ]
            
def inspect_grad(named_params):
    name_list = ['base_model.rnns.2.module.weight_hh_l0_raw', 'base_model.encoder_U']
    for name, param in named_params:
        if name in name_list:
            print(torch.norm(param.grad))

def regroup_params(named_params, args):
    named_params = list(named_params)
    tied_params = []
    base_model_params = []
    for name, param in named_params:
        if param.requires_grad:
            if ('base_model' in name and 'encoder' in name) or\
                ('top_model.decoder_bias' in name):
                # for sc_smax, decoder_bias should also be included 
                tied_params.append(param)
            elif 'base_model' in name:
                base_model_params.append(param)
    top_model_params = regroup_top_model_params(named_params, args)
    print("total param groups before regrouping is %s" % (
            len([x[1] for x in named_params if x[1].requires_grad is True])
            )
        )
    print("total param groups after regrouping is %s + %s + %s = %s" % (
        len(tied_params), len(base_model_params), 
        param_group_len(top_model_params), 
        len(tied_params) + len(base_model_params)\
            + param_group_len(top_model_params)
    ))

    return [
        {
            'params': tied_params,
            'weight_decay': args.tied_wdecay if args.decay_tied else args.wdecay,
            'lr': args.tied_lr if args.tied_lr > 0.0 else args.lr
        },
        {
            'params': base_model_params,
            'weight_decay': args.wdecay
        },
        *top_model_params
    ]


def get_per_param_options(model, criterion, args):
    grouped_params = regroup_params(model.named_parameters(), args) 
    if len(list(criterion.parameters())) > 0:
        grouped_params.append(
            {
                'params': criterion.parameters(),
                'weight_decay': args.wdecay
            }
        )
    return grouped_params




def get_optimizer(model, criterion, args):

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            get_per_param_options(model, criterion, args), lr=args.lr
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            get_per_param_options(model, criterion, args), lr=args.lr
        )

    if args.continue_train:
        optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
        if args.optimizer == 'sgd' and\
            't0' in optimizer_state['param_groups'][0]:

            optimizer = torch.optim.ASGD(
                get_per_param_options(model, criterion, args), 
                lr=args.lr, t0=0, lambd=0.
            )
            
        optimizer.load_state_dict(optimizer_state)
    
    return optimizer


def get_model_args(args, data, hidden, targets, is_training, calc_rank=False):
    model_args =  {
        'input': data,
        'extras': {
            'hidden': hidden,
            'top_model': args.top_model,
            'return_top_metrics': True,
            'return_f_logits': False,
            'criterion': args.criterion
        }
    }

    if is_training:
        model_args['extras']['return_all'] = True
        model_args['extras']['noisy_emb'] = args.noisy_emb
        model_args['extras']['emb_noise_scale'] = args.emb_noise_scale
        if args.top_model == 'adv_smax':
            model_args['extras']['targets'] = targets
    else:
        model_args['extras']['return_all'] = False

    if calc_rank:
        model_args['extras']['return_f_logits'] = True
        model_args['extras']['return_top_metrics'] = False

    if pkg_map()[args.top_model] != 'smax':
        model_args['extras']['return_top_metrics'] = False

    return model_args

def get_criterion_args(criterion, output, targets, top_model):
    criterion_args = {
        'output': output,
        'targets': targets,
        'decoder': None
    }
    if criterion in ['split_cross', 'spectrum_cross']:
        criterion_args['decoder'] = top_model.decoder
    return criterion_args



def parse_model_result(result, top_metrics, is_training, calc_rank=False):
    if top_metrics is not None and 'top_metrics' in result:
        top_metrics.accumulate(result['top_metrics'])
    if is_training:
        return result['model_output'], result['hidden'], \
                result['raw_outputs'], result['outputs']
    else:
        if calc_rank:
            return result['model_output'], result['hidden'], \
                    result['f_logits']
        else:
            return result['model_output'], result['hidden']

def emb_patch_and_freeze(model, args):
    if args.emb_patch != '':
        if torch.cuda.is_available() and args.cuda:
            src_model = torch.load(args.emb_patch)
        else:
            src_model = torch.load(args.emb_patch, map_location='cpu')
        inp_emb_sum = model.base_model.encoder.weight.data.sum().item()
        out_emb_sum = model.top_model.decoder.weight.data.sum().item()
        print("before patching inp_emb sum is {:.12f}, out_emb sum is".format(
                inp_emb_sum, out_emb_sum
            )
        )
        model.base_model.encoder.weight.data =\
            src_model['base_model.encoder.weight']
        inp_emb_sum = model.base_model.encoder.weight.data.sum().item()
        out_emb_sum = model.top_model.decoder.weight.data.sum().item()
        print("after patching inp_emb sum is {:.12f}, out_emb sum is".format(
                inp_emb_sum, out_emb_sum
            )
        )
    if args.emb_freeze:
        model.base_model.encoder.weight.requires_grad = False      
        print("freezing emb with its sum {:.12f}".format(
            model.base_model.encoder.weight.data.sum().item()
        ))
    return model

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # target = Variable(source[i+1:i+1+seq_len].view(-1))
    target = source[i+1:i+1+seq_len]
    return data, target


def create_exp_dir(args):
    path = args.save
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, 'misc'))
        os.makedirs(os.path.join(path, 'scripts'))
    else:
        print('exp directory already exists. exiting ..')
        sys.exit(1)

    print('Experiment dir : {}'.format(path))
    
def copy_assets(args, comet):
    scripts_to_save = [
        'main.py', "base_models/%s.py" % (args.base_model),
        "top_models/%s/%s.py" % (pkg_map()[args.top_model], args.top_model)
    ]
    for script in scripts_to_save:
        comet.log_asset(script)
        dst_file = os.path.join(args.save, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)

def save_checkpoint(model, optimizer, path, state):
    torch.save(state, os.path.join(path, 'state.pt'))
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))

def save_recent(model, optimizer, path, state):
    torch.save(state, os.path.join(path, 'state_recent.pt'))
    torch.save(model.state_dict(), os.path.join(path, 'model_recent.pt'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_recent.pt'))
    
    