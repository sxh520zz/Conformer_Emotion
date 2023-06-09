import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Utterance_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(Utterance_net, self).__init__()
        self.hidden_dim = args.hidden_layer
        self.num_layers = args.dia_layers
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        # gru
        self.bigru = nn.GRU(input_size, self.hidden_dim,
                            batch_first=True, num_layers=self.num_layers, bidirectional=True)
        # linear
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size // 4), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_size // 4, output_size))

    def forward(self, x):
        x_1 = self.layer1(x)
        x_1 = self.dropout(x_1)
        out = self.layer2(x_1)
        return out


