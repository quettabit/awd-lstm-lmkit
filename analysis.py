
import argparse

from utils.analysis.embed_variance.process import start as eval_embed_var
from utils.analysis.qualitative.process import start as eval_qualitative
from utils.analysis.qualitative.args import add_qualitative_args
from utils.analysis.word_similarity.args import add_word_similarity_args
from utils.analysis.word_similarity.process import start as eval_word_sim



def add_common_args(parser):
    parser.add_argument('--config_csv', type=str,
                        help='path csv file that has model location and name')


def add_all_args(parser):
    parser.add_argument('--analysis_type', type=str, default='eval_word_sim')
    add_common_args(parser)
    add_word_similarity_args(parser)
    add_qualitative_args(parser)



def do_analysis(args):
    if args.analysis_type == 'eval_word_sim':
        eval_word_sim(args)
    elif args.analysis_type == 'eval_embed_var':
        eval_embed_var(args)
    elif args.analysis_type == 'qualitative':
        eval_qualitative(args)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='analysis after training')
        add_all_args(parser)
        args = parser.parse_args()
        do_analysis(args)


