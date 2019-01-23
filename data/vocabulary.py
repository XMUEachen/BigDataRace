import numpy
import torch


class Vocabulary(object):
    def __init__(self, file_or_object, limit: int = None, eos: str = '</s>', unk: str = '<unk>',
                 pad: str = "<blank>") -> None:
        super().__init__()

        self.limit = limit
        self.eos = eos
        self.unk = unk
        self.pad = pad

        word2index, index2word = self.load(file_or_object)

        self.eos_id: int = word2index[eos]
        self.unk_id: int = word2index[unk]
        self.pad_id: int = word2index[pad]

        self._word2index, self._index2word = word2index, index2word

    def load(self, file_or_object):
        limit = self.limit
        tokens = [self.pad, self.eos, self.unk]

        if isinstance(file_or_object, str):
            word_types = self._from_file(file_or_object)
        else:
            word_types = self._from_object(file_or_object)

        if self.limit is None:
            limit = len(word_types)

        word_types = word_types[:limit]

        word2index = dict()
        index2word = dict()
        for i, word in enumerate(tokens + word_types):
            word2index[word] = i
            index2word[i] = word

        return word2index, index2word

    def _from_object(self, obj):
        word_types = [word for word in obj]
        return word_types

    def _from_file(self, voc_file):
        word_types = []
        if voc_file is not None:
            with open(voc_file) as r:
                for l in r:
                    word_types.append(l.split()[0])
        return word_types

    def to_indices(self, seq, dtype='int64'):
        prepends = []
        appends = [self.eos_id]

        w2i = self._word2index

        ids = [w2i[word] if word in w2i else self.unk_id for word in seq]
        return numpy.array(prepends + ids + appends, dtype)

    def to_words(self, ids, discards=None):
        if isinstance(discards, int):
            discards = [discards]

        if not isinstance(discards[0], int):
            raise ValueError('discards should be a list of token ids, got {}'.format(type(discards[0])))

        if isinstance(ids, torch.Tensor):
            ids = list(ids.numpy())

        seq = [self._index2word[id] for id in ids if id not in discards]
        return seq

    def __len__(self):
        return len(self._word2index)
