import argparse

def add_shift_smax_args(parser):
    parser.add_argument('--shift_smax_s', type=float, default=0.0,
                        help='translate/shift along Y axis by add/subtract')