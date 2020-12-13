import argparse
import sys

from base_models.args.dai import add_dai_args
from utils.shared import pkg_map
from top_models.adv_smax.args.adv_smax import add_adv_smax_args
from top_models.sc_smax.args.sc_smax import add_sc_smax_args
from top_models.smax.args.gs_smax import add_gs_smax_args
from top_models.smax.args.plif_smax import add_plif_smax_args
from top_models.smax.args.shift_smax import add_shift_smax_args
from top_models.mos.args.mos import add_mos_args
from top_models.xmoc.args.xmoc import add_xmoc_args
from top_models.xmos.args.xmos import add_xmos_args
from top_models.mt.args.mt import add_mt_args


class CLIParser():

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='PyTorch LM kit'
        )

    def add_experiment_args(self):
        self.parser.add_argument('--data', type=str,
                                 help='location of the data corpus')
        self.parser.add_argument('--seed', type=int, default=1111,
                                 help='random seed')
        self.parser.add_argument('--base_model', type=str, default='dai')
        self.parser.add_argument('--top_model', type=str, default='smax')

    def add_data_args(self):
        self.parser.add_argument('--batch_size', type=int, default=20,
                                 help='batch size')
        self.parser.add_argument('--eval_batch_size', type=int, default=10,
                                 help='validation batch size')
        self.parser.add_argument('--test_batch_size', type=int, default=1,
                                 help='batch size')
        self.parser.add_argument('--bptt', type=int, default=70,
                                 help='sequence length')
        self.parser.add_argument('--small_batch_size', type=int, default=-1,
                                 help="""the batch size for computation. 
                                        batch_size should be divisible by 
                                        small_batch_size. In our implementation, 
                                        we compute gradients with 
                                        small_batch_size multiple times, and 
                                        accumulate the gradients until 
                                        batch_size is reached. An update step
                                        is then performed.""")
        self.parser.add_argument('--max_seq_len_delta', type=int, default=40,
                                 help='max sequence length')
        self.parser.add_argument('--cuda', action='store_true',
                                 help='use CUDA')
        self.parser.add_argument('--single_gpu', default=False,
                                 action='store_true', help='use single GPU')

    def add_training_args(self):
        self.parser.add_argument('--criterion', type=str, default='cross',
                                 help='loss criterion to use')
        self.parser.add_argument('--flood_b', type=float,
                                 help='training loss val to random walk around')
        self.parser.add_argument('--optimizer', type=str, default='sgd',
                                 help='initial optimizer to use')
        self.parser.add_argument('--lr', type=float, default=30,
                                 help='initial learning rate')
        self.parser.add_argument('--clip', type=float, default=0.25,
                                 help='gradient clipping')
        self.parser.add_argument('--epochs', type=int, default=8000,
                                 help='upper epoch limit')
        self.parser.add_argument('--log-interval', type=int, default=200,
                                 metavar='N', help='report interval')
        self.parser.add_argument('--save', type=str,  default='EXP',
                                 help='path to save the final model')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training')
        self.parser.add_argument('--continue_from_epoch', type=int, default=1000,
                                 help='start epoch during continued training')
        self.parser.add_argument('--nonmono', type=int, default=5,
                                 help='NT ASGD parameter')
        self.parser.add_argument('--wdecay', type=float, default=1.2e-6,
                                 help='weight decay applied to all weights')
        self.parser.add_argument('--tied_wdecay', type=float, default=2.4e-6,
                                 help='weight decay applied to tied weights')
        self.parser.add_argument('--tied_lr', type=float, default=-1,
                                 help='lr for tied embeddings. used in sc_smax')
        self.parser.add_argument('--no_switch', action='store_true',
                                 help='do not switch to asgd')
        self.parser.add_argument('--switch_epoch', type=int, default=-1,
                                 help="""if sgd, switch to asgd at this epoch
                                        elif adam, decrease lr at this epoch""")
        self.parser.add_argument('--save_recent', action='store_true',
                                 help='if given saves the recent model')
        

    def add_base_model_args(self):
        add_dai_args(self.parser)

    def add_debug_args(self):
        self.parser.add_argument('--detect_anomaly', action='store_true',
                                 help='easily debug autograd issues')
        self.parser.add_argument('--local_debug', action='store_true',
                                 help='local debug mode')
        self.parser.add_argument('--no_comet', action='store_true',
                                 help='if given, disables comet')
        self.parser.add_argument('--no_analysis', action='store_true',
                                 help='if given, disables rank analysis')
        self.parser.add_argument('--recent_model_analysis', action='store_true',
                                 help="""if given, enables rank analysis for 
                                        recent model""")

    def add_top_model_args(self):
        add_gs_smax_args(self.parser)
        add_plif_smax_args(self.parser)
        add_adv_smax_args(self.parser)
        add_sc_smax_args(self.parser)
        add_shift_smax_args(self.parser)
        add_mos_args(self.parser)
        add_xmoc_args(self.parser)
        add_xmos_args(self.parser)
        add_mt_args(self.parser)
        self.parser.add_argument('--ctx_noise_scale', type=float, default=0.15,
                                    help='gaussian noise scaling factor')

    def add_model_args(self):
        self.parser.add_argument('--tied', action='store_true', default=True,
                                 help="""tie the word embedding and
                                        softmax weights""")
        self.parser.add_argument('--decay_tied', action='store_true',
                                 help='decay the tied weights')
        self.parser.add_argument('--emb_freeze', action='store_true',
                                    default=False, help='freezes embedding')
        self.parser.add_argument('--emb_patch', type=str, default='',
                                    help='model location to copy emb from')


    def add_args(self):
        self.add_experiment_args()
        self.add_data_args()
        self.add_base_model_args()
        self.add_top_model_args()
        self.add_model_args()
        self.add_training_args()
        self.add_debug_args()


    def enrich_args(self, args):
        if 'wikitext-103' in args.data and not args.no_analysis:
            print('rank analysis is not supported for WK103. changing it ..')
            args.no_analysis = True

        mix_models = ['mos', 'xmoc', 'xmos', 'mt']
        if pkg_map()[args.top_model] not in mix_models and args.nhidlast != -1:
            print('nhidlast is supported only for mix family. resetting to -1.')
            args.nhidlast = -1


        if args.nhidlast < 0:
            args.nhidlast = args.nemb
        if args.small_batch_size < 0:
            args.small_batch_size = args.batch_size

    def validate_args(self, args):
        if 'wikitext-103' in args.data:
            if args.top_model != 'smax':     
                print(("only smax top model is supported for wikitext-103. "
                        "exiting .."))
                sys.exit(1)
            if args.criterion != 'split_cross':
                print(("only split_cross_entropy is supported for wikitext-103."
                        " exiting .. "))
                sys.exit(1)

        if args.criterion == 'split_cross':
            if 'wikitext-103' not in args.data:
                print(("split_cross_entropy is currently supported only for "
                        "wikitext-103. exiting .."))
                sys.exit(1)
        
        if args.criterion == 'flooded_cross':
            if args.flood_b is None:
                print(("training loss to random walk around is not given. "
                        "exiting .. """))
                sys.exit(1)
        

    def parse_args(self):
        self.add_args()
        args = self.parser.parse_args()
        self.validate_args(args)
        self.enrich_args(args)
        return args