import gin
import torch
import torch.nn as nn

from torch import Tensor


def gru(horizon_len: int):
    return GRU(horizon_len)


def regru(horizon_len: int):
    return ReGRU(horizon_len)


class GRU(nn.Module):
    def __init__(self, horizon_len: int):
        super().__init__()
        layer_size = 512
        self.gru = nn.GRU(input_size=1,
                          hidden_size=layer_size,
                          num_layers=2,
                          bidirectional=False,
                          batch_first=True)
        # self.linears = nn.ModuleList([nn.Linear(layer_size, 1) for _ in range(horizon_len)])
        self.linear = nn.Linear(layer_size, horizon_len)

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        _, hn = self.gru(x)
        hn = hn[-1]  # last layer
        pred = self.linear(hn)
        # pred = torch.cat([x(hn) for x in self.linears], dim=1)
        return pred.unsqueeze(-1)


class ReGRU(nn.Module):
    def __init__(self, horizon_len: int):
        super().__init__()
        layer_size = 512
        self.gru = nn.GRU(input_size=1,
                          hidden_size=layer_size,
                          num_layers=2,
                          bidirectional=False,
                          batch_first=True)
        self.linear = nn.Linear(layer_size, 1)

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        pred = []
        h0 = None
        gru_in = x
        for _ in range(x.shape[1]):
            _, hn = self.gru(gru_in, h0)
            h0 = hn
            hn = hn[-1]  # last layer
            out = self.linear(hn)
            pred.append(out)
            gru_in = out.unsqueeze(-1)
        pred = torch.cat(pred, dim=1)
        return pred.unsqueeze(-1)
