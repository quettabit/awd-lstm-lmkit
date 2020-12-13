import torch
import torch.nn as nn


class SSmax(nn.Module):

    def __init__(self, ntoken, nhidlast):
        super(SSmax, self).__init__() 
        self.decoder = nn.Linear(nhidlast, ntoken)
        self.ntoken = ntoken
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def sigsoftmax(self, logits):
        num = torch.exp(logits) * torch.sigmoid(logits)
        den = num.sum(dim=-1, keepdim=True)
        # dim = -1 is the last dimension
        return num/den

    def forward(self, input, extras):
        """
        input: (num_batches, batch_size, nhidlast)
        """
        result = {}
        logits = self.decoder(input)
        log_prob = torch.log(self.sigsoftmax(logits))
        result['output'] = log_prob.view(-1, self.ntoken)
        if extras['return_f_logits']:
            result['f_logits'] = None
        return result
        