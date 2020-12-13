import argparse

def add_plif_smax_args(parser):
    parser.add_argument('--plif_smax_k', type=int, default=100000,
                        help='number of knots')
    parser.add_argument('--plif_smax_t', type=float, default=20,
                        help='interval range [-t, t]')
    parser.add_argument('--plif_smax_w_variance', type=float, default=1.0,
                        help='variance for plif_w weights')
    parser.add_argument('--plif_smax_lr', type=float, default=0.02,
                        help='learning rate for plif parameters')
    parser.add_argument('--plif_smax_freeze', action='store_true', default=False,
                        help='freezes plif_w')
    parser.add_argument('--plif_smax_patch', type=str, default='',
                        help='model location to copy weights from')
    #Var[kX] = k^2 * Var[X]