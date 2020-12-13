from argparse import Namespace

from models.model import Model
from utils.analysis.qualitative.model_args import *


def get_args(model_id):
    if model_id == 'ptb_softmax':
        return ptb_softmax()
    elif model_id == 'ptb_ss':
        return ptb_ss()
    elif model_id == 'ptb_gss':
        return ptb_gss()
    elif model_id == 'ptb_lms_plif':
        return ptb_lms_plif()
    elif model_id == 'ptb_moc_ntasgd':
        return ptb_moc_ntasgd()
    elif model_id == 'ptb_moc_etasgd':
        return ptb_moc_etasgd()
    elif model_id == 'ptb_mos_ntasgd':
        return ptb_mos_ntasgd()
    elif model_id == 'ptb_mos_etasgd':
        return ptb_mos_etasgd()

def load_model(model_id, model_state):
    model = Model(get_args(model_id))
    model.load_state_dict(model_state)
    return model