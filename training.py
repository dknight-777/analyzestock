import sys
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from pandas import DatetimeIndex, DataFrame, to_datetime

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> nn.Module:
    """モデルの学習を実行します。"""
    # PyTorch 2.0+ で利用可能なモデルのコンパイル
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            try:
                model = torch.compile(model)
                print("Model compiled successfully with torch.compile().")
            except Exception as e:
                print(f"Failed to compile model, proceeding without compilation: {e}")
        else:
            print(
                f"GPU compute capability ({major}.x) is less than 7.0. Skipping torch.compile()."
            )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    grad_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    print("モデルの学習を開始します...")
    model.train()
    for i in range(epochs):
        total_loss = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                y_pred = model(seq)
                loss = loss_function(y_pred, labels)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            total_loss += loss.item()

        if (i + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"\rEpoch {i+1}/{epochs} completed, Loss: {avg_loss:.4f}", end="")
            sys.stdout.flush()
    print("\nモデルの学習が完了しました。")
    return model


def predict_future_values(
    model: nn.Module,
    data_normalized: np.ndarray,
    future_dates: DatetimeIndex,
    scaler: MinMaxScaler,
    seq_length: int,
    device: torch.device,
    feature_columns: list,
) -> np.ndarray:
    """学習済みモデルを使用して将来の値を予測します。"""
    model.eval()

    test_inputs = data_normalized[-seq_length:].tolist()
    predictions_normalized = []

    for i in range(len(future_dates)):
        seq = torch.FloatTensor([test_inputs[-seq_length:]]).to(device)

        with torch.no_grad():
            predicted_close_normalized = model(seq).item()
            predictions_normalized.append(predicted_close_normalized)

            if i < len(future_dates) - 1:
                next_date = future_dates[i]
                future_df = DataFrame([[next_date]], columns=["record_date"])
                future_df["record_date"] = to_datetime(future_df["record_date"])

                future_df["day_of_week"] = future_df["record_date"].dt.dayofweek
                future_df["day_of_month"] = future_df["record_date"].dt.day
                future_df["month"] = future_df["record_date"].dt.month
                future_df["year"] = future_df["record_date"].dt.year
                future_df["day_of_year"] = future_df["record_date"].dt.dayofyear
                future_df["week_of_year"] = (
                    future_df["record_date"].dt.isocalendar().week.astype(int)
                )

                for col, max_val in [
                    ("day_of_week", 7),
                    ("day_of_month", 31),
                    ("month", 12),
                    ("day_of_year", 366),
                    ("week_of_year", 53),
                ]:
                    future_df[f"{col}_sin"] = np.sin(
                        2 * np.pi * future_df[col] / max_val
                    )
                    future_df[f"{col}_cos"] = np.cos(
                        2 * np.pi * future_df[col] / max_val
                    )

                # Use a temporary DataFrame to handle column alignment for scaling
                temp_df = DataFrame(0, index=[0], columns=feature_columns)
                for col in future_df.columns:
                    if col in temp_df.columns:
                        temp_df[col] = future_df[col].values

                next_input_scaled = scaler.transform(temp_df)
                next_input_scaled[0, 0] = predicted_close_normalized
                test_inputs.append(next_input_scaled[0].tolist())

    dummy_array = np.zeros((len(predictions_normalized), len(feature_columns)))
    dummy_array[:, 0] = predictions_normalized
    predictions_rescaled = scaler.inverse_transform(dummy_array)[:, 0]

    return predictions_rescaled
