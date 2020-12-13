import argparse

def add_adv_smax_args(parser):
    parser.add_argument('--adv_smax_alpha', type=float, default=0.005,
                        help='word norm scaling factor')
    