import math
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_layer_size: int = 100, output_size: int = 1
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_dim, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = input_seq.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)

        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions.squeeze(-1)


class NNModel(nn.Module):
    def __init__(self, input_dim: int, seq_length: int, output_size: int = 1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim * seq_length, 50)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.flatten(input_seq)
        x = self.linear1(x)
        x = self.relu(x)
        predictions = self.linear2(x)
        return predictions.squeeze(-1)


class GRUModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_layer_size: int = 100, output_size: int = 1
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_dim, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        batch_size = input_seq.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        gru_out, _ = self.gru(input_seq, h0)
        predictions = self.linear(gru_out[:, -1, :])
        return predictions.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Transformerモデル用の位置エンコーディング"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Tensor, shape [batch_size, seq_len, embedding_dim]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 2,
        d_hid: int = 64,
        nlayers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output.squeeze(-1)


def get_model(model_type: str, input_dim: int, seq_length: int) -> nn.Module:
    """モデル名に基づいてモデルのインスタンスを生成して返します。"""
    if model_type == "lstm":
        return LSTMModel(input_dim=input_dim)
    elif model_type == "gru":
        return GRUModel(input_dim=input_dim)
    elif model_type == "nn":
        return NNModel(input_dim=input_dim, seq_length=seq_length)
    elif model_type == "transformer":
        return TransformerModel(
            input_dim=input_dim, d_model=32, nhead=2, d_hid=64, nlayers=2, dropout=0.2
        )
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")
