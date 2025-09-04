import math
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_layer_size: int = 100, output_dim: int = 2
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_dim, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = input_seq.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)

        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


class NNModel(nn.Module):
    def __init__(self, input_dim: int, seq_length: int, output_dim: int = 2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim * seq_length, 100)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(50, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.flatten(input_seq)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        predictions = self.linear3(x)
        return predictions


class GRUModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_layer_size: int = 100, output_dim: int = 2
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_dim, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = input_seq.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        gru_out, _ = self.gru(input_seq, h0)
        predictions = self.linear(gru_out[:, -1, :])
        return predictions


def get_model(model_type: str, input_dim: int, seq_length: int, output_dim: int = 2) -> nn.Module:
    """モデル名に基づいてモデルのインスタンスを生成して返します。"""
    if model_type == "lstm":
        return LSTMModel(input_dim=input_dim, output_dim=output_dim)
    elif model_type == "gru":
        return GRUModel(input_dim=input_dim, output_dim=output_dim)
    elif model_type == "nn":
        return NNModel(input_dim=input_dim, seq_length=seq_length, output_dim=output_dim)
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")
