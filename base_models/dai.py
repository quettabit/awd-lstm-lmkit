import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from utils.base_models.dai.embed_regularize import embedded_dropout
from utils.base_models.dai.weight_drop import WeightDrop
from utils.models.locked_dropout import LockedDropout
from utils.models.svd_embed import SvdEmbed


class Dai(nn.Module):
    """
    Container module from https://github.com/zihangdai/mos 's model file 
    but excluding the components needed after the stacked LSTM's output.

    It is named as Dai based on the author Zihang Dai. 
    This base model is in turn trimmed from 
    https://github.com/salesforce/awd-lstm-lm/blob/master/model.py
    """

    def __init__(self, rnn_type, ntoken, nemb, nhid, nhidlast, nlayer,
                    dropout=0.5, dropouth=0.5, dropouti=0.5,
                    dropoute=0.1, wdrop=0, spectrum_control=False):
        super(Dai, self).__init__()
        self.use_dropout = True
        self.lockdrop = LockedDropout()
        if spectrum_control:
            self.encoder = SvdEmbed(nemb, ntoken)
        else:
            self.encoder = nn.Embedding(ntoken, nemb)
        self.wdropped = True
        if rnn_type == 'LSTM':
            self.rnns = [
                torch.nn.LSTM(
                    nemb if l == 0 else nhid, 
                    nhid if l != nlayer - 1 else nhidlast, 1, dropout=0
                ) for l in range(nlayer)
            ]
            if wdrop:
                self.rnns = [
                    WeightDrop(
                        rnn, ['weight_hh_l0'], 
                        dropout=wdrop if self.use_dropout else 0
                    ) for rnn in self.rnns
                ]
            else:
                self.wdropped = False
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [
                QRNNLayer(
                    input_size=nemb if l == 0 else nhid,
                    hidden_size=nhid if l != nlayer - 1 else nhidlast,
                    save_prev_x=True, zoneout=0, window=2 if l == 0 else 1,
                    output_gate=True
                ) for l in range(nlayer)
            ]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        
        self.rnns = torch.nn.ModuleList(self.rnns)
        
        self.rnn_type = rnn_type
        self.nemb = nemb
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayer = nlayer
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.spectrum_control = spectrum_control

        self.init_weights()

    def reset(self):
        if self.rnn_type == 'QRNN':
            [rnn.reset() for rnn in self.rnns]

    def init_weights(self):
        initrange = 0.1
        if not self.spectrum_control:
            self.encoder.weight.data.uniform_(-initrange, initrange)

    def get_embedding(self, input, extras):
        extras['is_training'] = self.training
        if self.spectrum_control:
            emb = embedded_dropout(
                self.encoder.get_W(), input, extras,
                dropout=self.dropoute if (self.training and self.use_dropout) else 0
            )
            return emb
            
        else:
            return embedded_dropout(
                self.encoder, input, extras,
                dropout=self.dropoute if (self.training and self.use_dropout) else 0
            )

    def forward(self, input, extras):
        """
        input: (num_batches, batch_size, -1)
        """
        batch_size = input.size(1)
        hidden = extras['hidden']
        emb = self.get_embedding(input, extras)
        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayer - 1:
                raw_output = self.lockdrop(
                    raw_output, self.dropouth if self.use_dropout else 0
                )
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(
            raw_output, self.dropout if self.use_dropout else 0
        )
        outputs.append(output)
        
        result = {}
        result['output'] = output
        result['hidden'] = hidden
        if extras['return_all']:
            result['raw_outputs'] = raw_outputs
            result['outputs'] = outputs

        return result

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden = []
        for l in range(self.nlayer):
            nhid = self.nhid if l != self.nlayer - 1 else self.nhidlast
            if self.rnn_type == 'LSTM':
                hidden.append(
                    (
                        weight.new(1, bsz, nhid).zero_(), 
                        weight.new(1, bsz, nhid).zero_()
                    )
                )
            if self.rnn_type == 'QRNN':
                hidden.append(
                    weight.new(1, bsz, nhid).zero_()
                )
        return hidden
            
       
if __name__ == '__main__':
    model = Dai('LSTM', 10, 12, 12, 12, 2)
    input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    hidden = model.init_hidden(9)
    model(input, hidden)
