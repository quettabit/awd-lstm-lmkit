import importlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models.model import init_base_model, init_top_model
from utils.models.model import patch_extras, patch_result


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.base_model = init_base_model(args)
        self.top_model = init_top_model(args)
        if args.tied:
            if args.spectrum_control and args.top_model == 'sc_smax':
                self.base_model.encoder = self.top_model.decoder
            else:
                self.base_model.encoder.weight = self.top_model.decoder.weight

    def forward(self, input, extras):

        bm_result = self.base_model(input, extras)
        if extras['criterion'] == 'split_cross':
            tm_result = None
        else:
            tm_result = self.top_model(
                bm_result['output'], patch_extras(extras, self.base_model)
            )
        return patch_result(bm_result, tm_result, extras)



