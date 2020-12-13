import argparse


def add_dai_args(parser):
    parser.add_argument('--alpha', type=float, default=2,
                        help="""alpha L2 regularization on RNN activation 
                                (alpha = 0 means no regularization)""")
    parser.add_argument('--beta', type=float, default=1,
                        help="""beta slowness regularization applied on RNN 
                                activiation (0 means no regularization)""")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help="""dropout to remove words from
                                embedding layer""")
    parser.add_argument('--nemb', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nhidlast', type=int, default=-1,
                        help='number of hidden units for last layer')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--rnn_type', type=str, default='LSTM',
                        help='type of recurrent net LSTM or QRNN')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help="""amount of weight dropout to apply to the RNN 
                                hidden to hidden matrix""")
    parser.add_argument('--noisy_emb', action='store_true',
                        help='boolean flag to enable noise addition to emb')
    parser.add_argument('--emb_noise_scale', type=float, default=0.15,
                        help='gaussian noise scaling factor')
    parser.add_argument('--spectrum_control', action='store_true',
                        default=False, help='spectrum control for emb')

