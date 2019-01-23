import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as pad

from nn.embedding import Embedding
from nn.ff import PositionWiseFeedForward

class Classifier(nn.Module):
    def __init__(self, args, vocabulary, num_class):
        super().__init__()

        # 把词编码成向量
        self.embedding = Embedding(len(vocabulary), args.input_size, True, vocabulary.pad_id)
        # 编码文本
        self.rnn = nn.LSTM(input_size=args.input_size,
                           hidden_size=args.hidden_size,
                           bidirectional=args.bidirectional,
                           num_layers=args.num_layers)
        # FFN
        self.ffn=PositionWiseFeedForward(args.hidden_size,4*args.hidden_size,args.hidden_size,args.dropout)
        # 文本表示转换成类别
        self.logits = nn.Linear(args.hidden_size, num_class)

        self.vocabulary = vocabulary
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout

    def forward(self, input):
        B, T = input.size()

        lens = input.ne(self.vocabulary.pad_id).sum(1).long()

        x = self.embedding(input)  # B x T x D
        x = F.dropout(x, self.dropout, self.training)
        x = pack(x, lens, batch_first=True)

        x, _ = self.rnn(x)

        x = pad(x, True)
        B_, T_, H = x.size()
        assert B_ == B
        assert T_ == T

        x=F.dropout(x, self.dropout, self.training)
        x=self.ffn(x)
        x=F.dropout(x, self.dropout, self.training)

        x = x.sum(1) / lens.unsqueeze(1).float()  # B x H

        logits = self.logits(x)  # B x C, C: num_class
        return logits

    def get_probs(self, logits, log_prob=False):
        if log_prob:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=1)
