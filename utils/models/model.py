import importlib

import torch

from utils.shared import pkg_map
from utils.top_models.plif_smax import patch_and_freeze as plif_patch_and_freeze


def snake_to_camel(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))

def patch_extras(extras, base_model):
    if extras['top_model'] == 'adv_smax':
        extras['encoder'] = base_model.encoder
    return extras

def patch_result(bm_result, tm_result, extras):
    result = {}
    if tm_result is not None:
        result['model_output'] = tm_result['output']
        if extras['return_top_metrics']:
            result['top_metrics'] = tm_result['top_metrics']
        if extras['return_f_logits']:
            result['f_logits'] = tm_result['f_logits']
    else:
        result['model_output'] = bm_result['output']

    result['hidden'] = bm_result['hidden']
    if extras['return_all']:
        result['raw_outputs'] = bm_result['raw_outputs']
        result['outputs'] = bm_result['outputs']

    return result

    
def cls_from_module(module_name, is_base=True):
    if is_base:
        module_path = "base_models.%s" % (module_name)
    else:
        module_path = "top_models.%s.%s" % (pkg_map()[module_name], module_name)
    module =  importlib.import_module(module_path)
    return getattr(module, snake_to_camel(module_name))

def init_base_model(args):
    cls_ = cls_from_module(args.base_model)
    if args.base_model == 'dai':
        return cls_(
            args.rnn_type, args.ntoken, args.nemb, args.nhid, args.nhidlast,
            args.nlayer, args.dropout, args.dropouth, args.dropouti,
            args.dropoute, args.wdrop, args.spectrum_control
        )
    else:
        print('base model module not found')

def init_top_model(args):
    cls_ = cls_from_module(args.top_model, is_base=False)
    if args.top_model == 'gs_smax':
        return cls_(args.ntoken, args.nhidlast, args.gs_smax_c, args.gs_smax_k)
    elif args.top_model == 'shift_smax':
        return cls_(args.ntoken, args.nhidlast, args.shift_smax_s)
    elif args.top_model == 'plif_smax':
        top_model = cls_(
            args.ntoken, args.nhidlast, args.plif_smax_k, args.plif_smax_t,
            args.plif_smax_w_variance
        )
        return plif_patch_and_freeze(top_model, args)
    elif args.top_model == 'adv_smax':
        return cls_(
            args.ntoken, args.nhidlast, args.adv_smax_alpha
        )
    elif pkg_map()[args.top_model] == 'mos':
        return cls_(
            args.ntoken, args.nhidlast, args.nemb, 
            args.mos_nexpert, args.mos_dropoutl
        )
    elif args.top_model in pkg_map().keys():
        return cls_(args.ntoken, args.nhidlast)
    else:
        print('top model module not found')