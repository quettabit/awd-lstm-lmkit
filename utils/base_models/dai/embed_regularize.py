import numpy as np

import torch
import torch.nn.functional as F

def svd_embedded_dropout_util(weight, words, dropout):
    # no support for scaling and padding idx yet.
    if dropout:
        mask = weight.data.new().resize_((weight.size(0), 1))\
                .bernoulli_(1 - dropout).expand_as(weight) / (1 - dropout)
        masked_weight = mask * weight
    else:
        masked_weight = weight
    
    return F.embedding(words, masked_weight)


def embedded_dropout_util(embed, weight, words, dropout, scale):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1))\
                .bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_weight = mask * weight

    else:
        masked_weight = weight

    if scale:
        masked_weight = scale.expand_as(masked_weight) * masked_weight
    
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    
    return F.embedding(
        words, masked_weight, padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )

def embedded_dropout(embed, words, extras, dropout=0.1, scale=None):
    add_noise = False
    if extras['is_training'] and extras['noisy_emb']:
        add_noise = True

    if extras['criterion'] == 'spectrum_cross':
        emb = svd_embedded_dropout_util(embed, words, dropout)
        if add_noise:
            sigma = svd_embedded_dropout_util(
                torch.ones_like(embed), words, dropout
            )
    else:
        emb = embedded_dropout_util(embed, embed.weight, words, dropout, scale)
        if add_noise:
            sigma = embedded_dropout_util(
                embed, torch.ones_like(embed.weight), words, dropout, scale
            )
    
    if add_noise:
        m = torch.distributions.Normal(
            torch.zeros_like(sigma), torch.ones_like(sigma) * 1
        )
        sigma = m.sample() * extras['emb_noise_scale']
        emb = emb + sigma
    return emb