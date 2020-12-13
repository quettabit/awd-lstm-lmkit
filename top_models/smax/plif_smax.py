import math

import torch
import torch.nn as nn

from top_models.smax.smax import Smax

"""
Code courtesy: Ganea et al. 2019.

https://github.com/pytorch/pytorch/blob/35cdb785228b8abea1d3bdb844aa5980e6642f8d/tools/autograd/derivatives.yaml
Because of the above, plif smax is not reproducible for same seed when run twice.

Other helpful links:
https://github.com/pytorch/pytorch/blob/35cdb785228b8abea1d3bdb844aa5980e6642f8d/tools/autograd/derivatives.yaml
"""

class PlifSmax(Smax):

    # base_interval: defines the logits range on which the monotonic layer
    # will be applied
    def __init__(self, ntoken, nhidlast, K, T, w_variance):
        super(PlifSmax, self).__init__(ntoken, nhidlast)
        self.T = T
        self.K = K
        self.plif_w = nn.Parameter(
            torch.randn(self.K) * w_variance + math.log(math.exp(1) - 1)
        )

    # logits : size = num_ctxts * bs * num_vocab_words ,
    # i.e. <h,w> dot products
    def func(self, logits):
        size = logits.size()
        logits = logits.view(-1)
        delta = 2. * self.T / self.K
        indices = torch.clamp(
            ((logits + self.T) / delta).detach().long(),
            max=self.K - 1, min=0
        )
        all_pos_w = nn.Softplus()(self.plif_w)
        all_pos_cumsum = torch.cumsum(all_pos_w, dim=-1) - all_pos_w
        pos_w = torch.gather(all_pos_w, -1, indices)
        # use gather, not take
        pos_w_cumsum = torch.gather(all_pos_cumsum, -1, indices)
        knots = (-self.T + delta * indices.float())
        knots = torch.tensor(
            knots, dtype=knots.dtype, device=logits.device
        )
        result = (logits - knots) * pos_w + delta * pos_w_cumsum
        return result.view(size)

    def forward(self, input, extras):
        return super(PlifSmax, self).forward(input, extras)
