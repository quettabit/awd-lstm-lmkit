import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models.locked_dropout import LockedDropout

class Mos(nn.Module):

    def __init__(self, ntoken, nhidlast, nemb, nexpert, dropoutl):
        super(Mos, self).__init__() 
        self.decoder = nn.Linear(nemb, ntoken)
        self.prior = nn.Linear(nhidlast, nexpert, bias=False)
        self.latent = nn.Sequential(
            nn.Linear(nhidlast, nexpert * nemb), nn.Tanh()
        )
        self.lockdrop = LockedDropout()
        self.nexpert = nexpert
        self.nemb = nemb
        self.ntoken = ntoken
        self.dropoutl = dropoutl
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, extras):
        """
        input: (num_batches, batch_size, nhidlast)
        """
        result = {}
        latent = self.latent(input)
        latent = self.lockdrop(latent, self.dropoutl)
        logits = self.decoder(latent.view(-1, self.nemb))
        prob = F.softmax(logits.view(-1, self.ntoken), -1).view(
            -1, self.nexpert, self.ntoken
        )
        
        prior_prob = self.prior(input).contiguous().view(-1, self.nexpert)
        prior = F.softmax(prior_prob, -1)

        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)
        log_prob = torch.log(prob)
        result['output'] = log_prob.view(-1, self.ntoken)
        if extras['return_f_logits']:
            result['f_logits'] = None
        return result
        