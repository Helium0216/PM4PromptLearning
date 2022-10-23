import os
import argparse
from transformers import BertModel, BertTokenizer

from PerturbedMasking import get_matrix


if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    }
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument('--layers', default='12')

    # Data args
    parser.add_argument('--dataset', default='../data/LM-BEF/SST-2/', required=True)
    parser.add_argument('--output_dir', default='./results/LM-BEF/SST-2/')

    parser.add_argument('--metric', default='dist', help='metrics for impact calculation, support [dist, cos] so far')
    parser.add_argument('--cuda', action='store_true', help='invoke to use gpu')

    parser.add_argument('--matrix', default='../results/LM-BEF/SST-2/bert-dist-SciDTB-last.pkl')
    parser.add_argument('--subword', default='avg', choices=['first', 'avg', 'max'])

    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODEL_CLASSES[args.model_type]

    args.output_dir = args.output_dir + args.probe + '/'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_file = args.output_dir + '/{}-{}-{}.pkl'

    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    print(args)
    # dataset = ConllUDataset(args.dataset)
    get_matrix.get_token_matrix(args, model, tokenizer, args.dataset)
    get_matrix.get_word_matrix(args)
