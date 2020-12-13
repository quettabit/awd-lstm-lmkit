import argparse

def add_gs_smax_args(parser):
    parser.add_argument('--gs_smax_c', type=float, default=0.0,
                        help='soft bending point of the curve')
    parser.add_argument('--gs_smax_k', type=float, default=2.0,
                        help='slope of the curve after bending')
                        