import argparse
import itertools
from collections import Counter

import numpy
import torch
import torch.nn as nn
from lunas.iterator import Iterator
from lunas.readers import Zip, Shuffle, TextLine
from sklearn import preprocessing
from sklearn.metrics.classification import f1_score

from data.plain import text_to_indices
from data.vocabulary import Vocabulary
from models.rnn_classifier import Classifier
from optim.adam import Adam
from optim.lr_scheduler.fixed_schedule import FixedSchedule
from utils.tensor import to_cuda, pack_tensors


def get_training_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--train', type=str, help='训练数据路径，格式为每行一段文本，表示事实，需要先进行分词。')
    parser.add_argument('--train-label', type=str, help='训练数据的标签文件路径，格式为每行一个标签。')
    parser.add_argument('--dev', type=str, help='开发集/验证集路径，格式同训练数据。')
    parser.add_argument('--dev-label', type=str, help='开发集/验证集的标签文件路径。')
    # vocab
    parser.add_argument('--vocab-size', type=int, help='限制词表大小，默认不限制')
    parser.add_argument('--vocab-threshold', type=int, help='限制词表中词的出现次数必须不低于阈值，默认为10', default=10)
    # training
    parser.add_argument('--batch-size', type=int, help='batch size.', defualt=100)
    parser.add_argument('--max-epoch', type=int, help='训练的最大轮数。')
    parser.add_argument('--eval-freq', type=int, help='evaluation frequency (default is 1500)', default=1500)
    parser.add_argument('--clip-norm', type=float, help='gradient clipping (default is 10.0)', default=10)

    # model
    parser.add_argument('--input-size', type=int, help='词向量的维度，默认为100', defualt=100)
    parser.add_argument('--hidden-size', type=int, help='词向量的维度，默认为150', defualt=150)
    parser.add_argument('--num-layers', type=int, help='网络层数，默认为2', defualt=2)
    parser.add_argument('--bidirectional', action='store_true', help='是否双向')
    parser.add_argument('--dropout', type=float, help='default: 0.3', default=0.3)

    FixedSchedule.add_args(parser)

    return parser.parse_args()


def get_shortlist(train, vocab_size=None, vocab_threshold=None):
    counter = Counter()
    with open(train) as r:
        for l in r:
            words = l.split()
            counter.update(words)
    word_counts = counter.most_common(vocab_size)
    if vocab_threshold is not None and vocab_threshold > 0:
        i = len(word_counts)
        for j in range(i - 1, -1, -1):
            count = word_counts[j][1]
            if count > vocab_threshold:
                i = j + 1
                break
        word_counts = word_counts[:i]
    return [word for word, _ in word_counts]


def save_model(name, *args):
    torch.save(args, name)


def get_train_iterator(args, vocab: Vocabulary, label_encoder: preprocessing.LabelEncoder):
    threads = 6
    buffer_size = 10000

    fact = TextLine(args.train, buffer_size=buffer_size, num_threads=threads)
    label = TextLine(args.train_label, buffer_size=buffer_size, num_threads=threads)
    ds = Zip([fact, label], buffer_size=buffer_size, num_threads=threads)

    ds = ds.select(
        lambda sample: (
            torch.as_tensor(text_to_indices(sample[0], vocab)),
            sample[1]
        )
    )
    ds = Shuffle(ds, buffer_size=-1, num_threads=threads)

    def collate_fn(samples):
        samples = list(itertools.zip_longest(*samples))
        xs, ys = samples
        return to_cuda(pack_tensors(xs, vocab.pad_id)), to_cuda(torch.as_tensor(label_encoder.transform(ys)).long())

    iterator = Iterator(
        ds, args.batch_size,
        cache_size=2048,
        collate_fn=collate_fn,
        sort_desc_by=lambda sample: sample[0].size(0)
    )

    return iterator


def get_devtest_iterator(fact_path, vocab: Vocabulary, label_encoder: preprocessing.LabelEncoder, label_path=None):
    threads = 6
    buffer_size = 10000
    if label_path is not None:
        fact = TextLine(fact_path, buffer_size=buffer_size, num_threads=threads)
        label = TextLine(label_path, buffer_size=buffer_size, num_threads=threads)
        ds = Zip([fact, label], buffer_size=buffer_size, num_threads=threads)

        ds = ds.select(
            lambda sample: (
                torch.as_tensor(text_to_indices(sample[0], vocab)),
                sample[1]
            )
        )

        def collate_fn(samples):
            samples = list(itertools.zip_longest(*samples))
            xs, ys = samples
            return to_cuda(pack_tensors(xs, vocab.pad_id)), to_cuda(torch.as_tensor(label_encoder.transform(ys)).long())

        iterator = Iterator(
            ds, 100,
            cache_size=100,
            collate_fn=collate_fn,
        )
    else:
        ds = TextLine(fact_path, buffer_size=buffer_size, num_threads=threads)
        ds = ds.select(
            lambda sample: torch.as_tensor(text_to_indices(sample[0], vocab))
        )

        def collate_fn(samples):
            return to_cuda(pack_tensors(samples, vocab.pad_id))

        iterator = Iterator(
            ds, 100,
            cache_size=100,
            collate_fn=collate_fn,
        )

    return iterator


def init_parameters(module):
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        return
    else:
        for name, para in module._parameters.items():
            if para is not None and para.requires_grad:
                if para.dim() >= 2:
                    nn.init.uniform_(para, -0.09, 0.09)
                else:
                    para.data.zero_()


def stat_parameters(model):
    num_params = 0
    num_params_requires_grad = 0
    for n, p in model.named_parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_requires_grad += p.numel()
    return num_params_requires_grad


def clip_grad(parameters, max_norm):
    max_norm = -1
    if max_norm is not None and max_norm > 0:
        max_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip_norm)
    return max_norm


def evaluate(model, eval_iter):
    model.eval()
    y_pred = []
    y_true = []
    for batch in eval_iter.iter_epoch():
        batch, (inputs, label) = batch
        logits = model(inputs)
        probs = model.get_probs(logits)
        y_pred.extend(probs.argmax(-1))
        y_true.extend(list(label))
    y_pred = numpy.array(y_pred, dtype=numpy.int32)
    y_true = numpy.array(y_true, dtype=numpy.int32)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return (micro_f1 + macro_f1) / 2


def main(args):
    shortlist = get_shortlist(args.train)
    vocabulary = Vocabulary(shortlist)

    le = preprocessing.LabelEncoder()
    with open(args.train_label) as r:
        le.fit(r.readlines())

    model = Classifier(args, vocabulary, args.num_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    lr_scheduler = FixedSchedule(args, optimizer)

    # initialize params
    model.apply(init_parameters)

    print('Model:', model)
    print('Num. parameters:', stat_parameters(model))

    train_iter = get_train_iterator(args, vocabulary, le)
    dev_iter = get_devtest_iterator(args.dev, vocabulary, le, args.dev_label)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    eval_scores = []

    for batch in train_iter.while_true(
            lambda: train_iter.epoch < args.max_epoch
    ):
        model.train()
        optimizer.zero_grad()
        # unpack
        batch, (facts, labels) = batch
        # forward
        logits = model(facts)
        # compute loss
        loss = criterion(logits, labels)
        # clip gradient norm
        grad_norm = clip_grad(model.parameters(), args.clip_norm)
        # update params
        optimizer.step()
        # update learning rate
        lr_scheduler.step_update(train_iter.step)

        print(f'|{train_iter.step}({train_iter.epoch}-{train_iter.step_in_epoch}) '
              f'|loss={loss:.4f} |'
              f'|grad={grad_norm:.2f}')

        if train_iter.step % args.eval_freq == 0:
            score = evaluate(model, dev_iter)
            if not eval_scores or score > max(eval_scores):
                save_model(f'ckp.best.{train_iter.step}.pt', args, model.state_dict(), vocabulary, le)
            eval_scores.append(score)
    score = evaluate(model, dev_iter)
    if not eval_scores or score > max(eval_scores):
        save_model(f'ckp.best.{train_iter.step}.pt', model, vocabulary, le)
    eval_scores.append(score)
    print('Training finished')


if __name__ == '__main__':
    args = get_training_args()
    main(args)
