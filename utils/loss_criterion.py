import sys

import torch
import torch.nn as nn

from utils.split_cross import SplitCrossEntropyLoss

class LossCriterion(nn.Module):

    def __init__(self, args):
        super(LossCriterion, self).__init__()

        if args.criterion == 'cross':
            self.criterion = nn.NLLLoss()
        elif args.criterion == 'flooded_cross':
            self.criterion = nn.NLLLoss()
            self.flood_b  = args.flood_b
        elif args.criterion == 'spectrum_cross':
            self.criterion = nn.NLLLoss()
            self.U_fn_rc = args.sc_smax_u_fn_rc
            self.V_fn_rc = args.sc_smax_v_fn_rc
            self.U_sn_rc = args.sc_smax_u_sn_rc
            self.V_sn_rc = args.sc_smax_v_sn_rc
            self.U_sm_rc = args.sc_smax_u_sm_rc
            self.V_sm_rc = args.sc_smax_v_sm_rc
            self.sv_rc = args.sc_smax_sv_rc
            self.GS = torch.load(args.sc_smax_gsv_dist_path) # gold sing. values
            # GS is a 1D sorted tensor
            if args.cuda and torch.cuda.is_available():
                self.I = torch.eye(
                    n=args.nemb, m=args.nemb, device=torch.device('cuda')
                )
                self.GS = self.GS.cuda()
            else:
                self.I = torch.eye(n=args.nemb, m=args.nemb)
        elif args.criterion == 'split_cross':
            self.criterion = SplitCrossEntropyLoss(
                args.nemb, splits=[2800, 20000, 76000]
            )
        self.criterion_type = args.criterion

    def forward(self, output, targets, decoder=None):
        # returns loss, raw_loss
        # loss contains additive regularization terms
        # output is log prob for all 'cross' models and basemodel output for 
        # only 'split_cross'
        if self.criterion_type == 'cross':
            raw_loss = self.criterion(output, targets)
            return raw_loss, raw_loss

        elif self.criterion_type == 'flooded_cross':
            "https://arxiv.org/pdf/2002.08709.pdf"
            raw_loss = self.criterion(output, targets)
            flodded_loss = (raw_loss - self.flood_b).abs() + self.flood_b
            return flodded_loss, raw_loss

        elif self.criterion_type == 'spectrum_cross':
            U_fn, V_fn, U_sn, V_sn, U_sm, V_sm = decoder.orthogonal_reg()
            raw_loss = self.criterion(output, targets)
            loss = raw_loss +\
                    (self.U_fn_rc * U_fn) + (self.V_fn_rc * V_fn) +\
                    (self.U_sn_rc * U_sn) + (self.V_sn_rc * V_sn) +\
                    (self.U_sm_rc * U_sm) + (self.V_sm_rc * V_sm) +\
                    (self.sv_rc * decoder.svs_reg(self.GS))
            return loss, raw_loss
                    
        elif self.criterion_type == 'split_cross':
            raw_loss = self.criterion(
                decoder.weight, decoder.bias, output, targets
            )
            return raw_loss, raw_loss


