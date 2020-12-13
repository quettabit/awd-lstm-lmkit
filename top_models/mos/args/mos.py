import argparse

def add_mos_args(parser):
    parser.add_argument('--mos_dropoutl', type=float, default=0.0,
                        help='dropout for latent layer')
    parser.add_argument('--mos_nexpert', type=int, default=1,
                        help='number of experts')
    
    