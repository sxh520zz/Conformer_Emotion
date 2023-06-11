import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
import math


# ====================================================================================================
class FFN(nn.Module):
    def __init__(self, dim_in, dim_out, drop):
        super(FFN, self).__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size=1)
        self.active = nn.ReLU()
        self.drop = drop

    def forward(self, x):
        x = self.drop(self.active(self.conv1(x.transpose(1, 2))))
        out = self.drop(self.conv2(x)).transpose(1, 2)
        return out


# ====================================================================================================
class RNNLayer(nn.Module):
    def __init__(self, dim_in, hidden_size):
        super(RNNLayer, self).__init__()
        self.rnn = nn.LSTM(dim_in, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x, h=None):
        self.rnn.flatten_parameters()  # To reduce the mem store
        out = self.rnn(x)[0] if h is None else self.rnn(x, h)[0]
        return out


class RNN1DLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(RNN1DLayer, self).__init__()
        self.rnn = nn.GRUCell(dim_in, dim_out)

    def forward(self, x, h=None):
        out = self.rnn(x) if h is None else self.rnn(x, h)
        return out


# ====================================================================================================
class CNN1DLayer(nn.Module):
    """ Using Separable Conv to reduce params:
        (ch_in*ch_out*k) -> (ch_in*1*k + ch_in*ch_out*1). """
    def __init__(self, ch_in, ch_out, k):
        super(CNN1DLayer, self).__init__()
        self.conv1 = nn.Conv1d(ch_in, ch_in, k, groups=ch_in, padding=k//2, bias=False)
        self.conv2 = nn.Conv1d(ch_in, ch_out, kernel_size=1)
        self.active = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x.transpose(1, 2))
        out = self.active(self.conv2(x)).transpose(1, 2)
        return out

class DRSelfAttention(nn.Module):
    """ Dimension Reduction Self-Attention. """
    def __init__(self, dim_in, hidden_size, drop):
        super(DRSelfAttention, self).__init__()
        if dim_in != hidden_size:
            self.linear = nn.Linear(dim_in, hidden_size)
        self.dr_linear = nn.Linear(dim_in, 1)
        self.drop = nn.dropout(drop)
        
    def forward(self, x):
        x = self.drop(x)
        score = self.dr_linear(x).squeeze()
        score = torch.softmax(score, dim=-1)
        if hasattr(self, "linear"):
            x = self.linear(x)
        attned = torch.bmm(score.unsqueeze(1), x).squeeze()
        return attned


# ====================================================================================================
# *** A New Recurrent Unit. *** #
class DRU(nn.Module):
    """ Diversity Recurrent Unit.
        Including global, local and 2 directions info. """
    def __init__(self, dim_in, hidden_size, drop=0.3, k=5):
        super(DRU, self).__init__()
        self.dr_self_attn = DRSelfAttention(dim_in, hidden_size*2, drop)
        self.conv = CNN1DLayer(dim_in, dim_in, k=k)
        self.gate = nn.Linear(dim_in + hidden_size*2, dim_in)
        self.rnn = RNNLayer(dim_in, hidden_size)
        self.drop = nn.dropout(drop)

    def forward(self, x):
        h = self.dr_self_attn(x)
        x_ = self.conv(self.drop(x))
        gate = torch.sigmoid(self.gate(self.drop(
            torch.cat([x, h.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1))))
        x = gate * x + (1 - gate) * x_
        h = torch.stack(torch.chunk(h, chunks=2, dim=-1), dim=0)
        out = self.rnn(self.drop(x), (h, h))
        return out
