import torch

from top_models.smax.smax import Smax

class GsSmax(Smax):

    def __init__(self, ntoken, nhidlast, c, k):
        super(GsSmax, self).__init__(ntoken, nhidlast)
        self.c = c 
        self.k = k
    
    def func(self, logits):
        return (self.k*(logits - self.c))\
                + self.c\
                -((self.k - 1) * torch.log(torch.exp(logits - self.c) + 1))
    
    def forward(self, input, extras):
        return super(GsSmax, self).forward(input, extras)




        