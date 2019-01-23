import argparse
import sys

import numpy
import torch
from sklearn.metrics.classification import f1_score

from models.rnn_classifier import Classifier
from train import get_devtest_iterator


def predict(model, eval_iter):
    model.eval()
    y_pred = []
    for batch in eval_iter.iter_epoch():
        batch, (inputs, label) = batch
        logits = model(inputs)
        probs = model.get_probs(logits)
        y_pred.extend(probs.argmax(-1))
    return y_pred


def get_evaluation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fact', type=str, required=True)
    parser.add_argument('--label', type=str, help='optional')
    parser.add_argument('--checkpoint', type=str, help='path to saved model')
    return parser.parse_args()


def main(args):
    args, model_state, vocabulary, le = torch.load(args.checkpoint)
    # restore model params
    model = Classifier(args, vocabulary, args.num_class)
    model.load_state_dict(model_state)

    iterator = get_devtest_iterator(args.fact, vocabulary, le, args.label)

    model.eval()
    y_pred = predict(model, iterator)
    for y in le.inverse_transform(y_pred):
        sys.stdout.write(f'{y}\n')

    if args.label:
        y_pred = numpy.array(y_pred, dtype=numpy.int32)
        y_true = numpy.fromfile(args.label, dtype=numpy.int32, sep='\n')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        sys.stderr.write(f'\n'
                         f'|micro_f1:{micro_f1}, '
                         f'macro_f1:{macro_f1}, '
                         f'avg:{(micro_f1+macro_f1)/2}\n')


if __name__ == '__main__':
    args = get_evaluation_args()

    main(args)
