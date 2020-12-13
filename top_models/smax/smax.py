import torch
import torch.nn as nn
import torch.nn.functional as F


class Smax(nn.Module):

    def __init__(self, ntoken, nhidlast):
        super(Smax, self).__init__() 
        self.decoder = nn.Linear(nhidlast, ntoken)
        self.ntoken = ntoken
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def func(self, logits):
        pass

    def get_metrics(self, logits, calc_metrics=False):
        metrics = {}
        if calc_metrics:     
            metrics['min'] = logits.min().item()
            metrics['max'] = logits.max().item()
            metrics['mean'] = logits.mean().item()
        return metrics

    def forward(self, input, extras):
        """
        input: (num_batches, batch_size, nhidlast) 
        """
        result = {}
        calc_metrics = extras['return_top_metrics']
        result['top_metrics'] = {}

        logits = self.decoder(input)
        result['top_metrics']['logits'] = self.get_metrics(logits, calc_metrics)
        func_logits = self.func(logits)
        if func_logits is None:
            func_logits = logits
        else:
            result['top_metrics']['f_logits'] = self.get_metrics(
                func_logits, calc_metrics
            )

        log_prob = F.log_softmax(func_logits, -1)
        result['output'] = log_prob.view(-1, self.ntoken)
        if extras['return_f_logits']:
            result['f_logits'] = func_logits.view(-1, self.ntoken)   
        return result