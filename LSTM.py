import gin
import torch
import torch.nn as nn

from torch import Tensor


def lstm(horizon_len: int):
    return LSTM(horizon_len)


def relstm(horizon_len: int):
    return ReLSTM(horizon_len)


class LSTM(nn.Module):
    def __init__(self, horizon_len: int):
        super().__init__()
        layer_size = 512
        self.lstm = nn.LSTM(input_size=1,
                          hidden_size=layer_size,
                          num_layers=2,
                          bidirectional=False,
                          batch_first=True)
        # self.linears = nn.ModuleList([nn.Linear(layer_size, 1) for _ in range(horizon_len)])
        self.linear = nn.Linear(layer_size, horizon_len)

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]  # last layer
        pred = self.linear(hn)
        # pred = torch.cat([x(hn) for x in self.linears], dim=1)
        return pred.unsqueeze(-1)


class ReLSTM(nn.Module):
    def __init__(self, horizon_len: int):
        super().__init__()
        layer_size = 512
        self.lstm = nn.LSTM(input_size=1,
                          hidden_size=layer_size,
                          num_layers=2,
                          bidirectional=False,
                          batch_first=True)
        self.linear = nn.Linear(layer_size, 1)

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        pred = []
        h0 = None
        c0 = None
        lstm_in = x
        for _ in range(x.shape[1]):
            if h0 is None:
                _, (hn, cn) = self.lstm(lstm_in)
            else:
                _, (hn, cn) = self.lstm(lstm_in, (h0, c0))
            h0 = hn
            c0 = cn
            hn = hn[-1]  # last layer
            out = self.linear(hn)
            pred.append(out)
            lstm_in = out.unsqueeze(-1)
        pred = torch.cat(pred, dim=1)
        return pred.unsqueeze(-1)
