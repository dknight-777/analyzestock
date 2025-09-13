import math
import torch
from torch import nn
from typing import List, Optional


class LSTMModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_unit_size: int = 100, output_dim: int = 1, num_layers: int = 1
    ):
        super().__init__()
        self.hidden_unit_size = hidden_unit_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_unit_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_unit_size, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = input_seq.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_unit_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_unit_size).to(input_seq.device)

        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


class NNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        output_dim: int = 1,
        nn_layer_units: Optional[List[int]] = None,
    ):
        super().__init__()
        if nn_layer_units is None:
            nn_layer_units = [100, 50]

        self.flatten = nn.Flatten()

        layers = []
        in_features = input_dim * seq_length
        for units in nn_layer_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            in_features = units
        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.flatten(input_seq)
        return self.network(x)


class GRUModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_unit_size: int = 100, output_dim: int = 1, num_layers: int = 1
    ):
        super().__init__()
        self.hidden_unit_size = hidden_unit_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_unit_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_unit_size, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = input_seq.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_unit_size).to(input_seq.device)
        gru_out, _ = self.gru(input_seq, h0)
        predictions = self.linear(gru_out[:, -1, :])
        return predictions


def get_model(
    model_type: str,
    input_dim: int,
    seq_length: int,
    output_dim: int = 1,
    hidden_unit_size: int = 100,
    num_layers: int = 1,
    nn_layer_units: Optional[List[int]] = None,
) -> nn.Module:
    """モデル名に基づいてモデルのインスタンスを生成して返します。"""
    if model_type == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_unit_size=hidden_unit_size,
            num_layers=num_layers,
        )
    elif model_type == "gru":
        return GRUModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_unit_size=hidden_unit_size,
            num_layers=num_layers,
        )
    elif model_type == "nn":
        return NNModel(
            input_dim=input_dim,
            seq_length=seq_length,
            output_dim=output_dim,
            nn_layer_units=nn_layer_units,
        )
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")
