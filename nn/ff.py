import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # y=w2(f(w1x))
        return self.linear2(self.dropout(self.relu(self.linear1(input))))
