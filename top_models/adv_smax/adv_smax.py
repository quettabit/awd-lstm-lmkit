import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvSmax(nn.Module):

    def __init__(self, ntoken, nhidlast, alpha):
        super(AdvSmax, self).__init__() 
        self.alpha = alpha
        self.decoder = nn.Linear(nhidlast, ntoken)
        self.ntoken = ntoken
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def func(self, logits):
        pass

    def forward(self, input, extras):
        """
        input: (num_batches, batch_size, nhidlast)
        extras['targets']: (num_batches * batch_size)
        extras['encoder']: from base model encoder
        """
        result = {}

        input_size = input.size()
        input = input.view(-1, input_size[-1])
        logits = self.decoder(input)
        if self.training:
            targets = extras['targets']
            encoder = extras['encoder']
            weight_noise = torch.zeros(
                self.decoder.weight.size(), dtype=self.decoder.weight.dtype,
                device=self.decoder.weight.device
            )
            neg_h = - input / torch.sqrt(
                torch.sum(input**2, 1, keepdim=True) + 1e-8
            )
            n_output = torch.sqrt(torch.sum(input**2, 1, keepdim=True) + 1e-8)
            n_w = torch.sqrt(
                torch.sum(encoder(targets)**2, 1, keepdim=True) + 1e-8
            )
            cos_theta = (torch.sum(input * encoder(targets), 1, keepdim=True))\
                        / n_output / n_w
            indicator = torch.tensor(
                torch.gt(cos_theta, 0e-1).view(-1, 1), dtype=cos_theta.dtype,
                device=cos_theta.device
            )
            epsilon = self.alpha * n_w * indicator
            weight_noise[targets.view(-1)] = epsilon.detach() * neg_h.detach()
            noise_outputs = (input * weight_noise[targets]).sum(1)
            l_idx = torch.tensor(
                torch.arange(targets.size(0)), dtype=torch.long,
                device=targets.device
            )
            logits[l_idx, targets] += noise_outputs

        logits = logits.view(input_size[0], input_size[1], -1)
        func_logits = self.func(logits)
        if func_logits is None:
            func_logits = logits

        log_prob = F.log_softmax(func_logits, -1)
        result['output'] = log_prob.view(-1, self.ntoken)
        if extras['return_f_logits']:
            result['f_logits'] = func_logits.view(-1, self.ntoken)
        return result