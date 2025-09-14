import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from pandas import DatetimeIndex, DataFrame, to_datetime, date_range, DateOffset

from data_processing import add_date_features, add_technical_features, add_dow_theory_features

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    log_interval: int = 10,
) -> nn.Module:
    """モデルの学習を実行します。"""
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            try:
                model = torch.compile(model)
                print("Model compiled successfully with torch.compile().")
            except Exception as e:
                print(f"Failed to compile model, proceeding without compilation: {e}")
        else:
            print(f"GPU compute capability ({major}.x) is less than 7.0. Skipping torch.compile().")

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    grad_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    print("モデルの学習を開始します...")
    model.train()
    for i in range(epochs):
        total_loss = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                y_pred = model(seq)
                loss = loss_function(y_pred, labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            total_loss += loss.item()
        if (i + 1) % log_interval == 0 or (i + 1) == epochs:
            avg_loss = total_loss / len(train_loader)
            print(f"\rEpoch {i+1}/{epochs} completed, Loss: {avg_loss:.4f}", end="")
            sys.stdout.flush()
    print("\nモデルの学習が完了しました。")
    return model

def predict_future_values(
    model: nn.Module,
    historical_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    scaler: MinMaxScaler,
    seq_length: int,
    device: torch.device,
    feature_columns: list,
    log_return_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """学習済みモデルを使用して将来の値を予測します。対数リターンを予測し、価格に変換します。"""
    model.eval()

    last_sequence = historical_df[feature_columns].values[-seq_length:]
    
    future_predictions = []
    
    open_offset = (historical_df['open'] - historical_df['close']).mean()
    high_offset = (historical_df['high'] - historical_df['close']).mean()
    low_offset = (historical_df['low'] - historical_df['close']).mean()
    
    current_df = historical_df.copy()

    for date in future_dates:
        # DataFrameに変換して警告を抑制
        last_sequence_df = pd.DataFrame(last_sequence, columns=feature_columns)
        last_sequence_scaled = scaler.transform(last_sequence_df)
        seq_tensor = torch.from_numpy(last_sequence_scaled).float().unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_scaled_log_return = model(seq_tensor)[0].cpu().numpy()

        dummy_array = np.zeros((1, len(feature_columns)))
        dummy_array[0, log_return_idx] = predicted_scaled_log_return
        unscaled_log_return = scaler.inverse_transform(dummy_array)[0, log_return_idx]

        last_close_price = last_sequence[-1, feature_columns.index('close')]
        predicted_close = last_close_price * np.exp(unscaled_log_return)

        price_change_threshold = 0.005
        price_change_ratio = (predicted_close - last_close_price) / last_close_price
        if abs(price_change_ratio) < price_change_threshold:
            predicted_volume = current_df['volume'].iloc[-5] if len(current_df) >= 5 else current_df['volume'].iloc[0]
        else:
            price_range_margin = 0.01
            price_min, price_max = predicted_close * (1 - price_range_margin), predicted_close * (1 + price_range_margin)
            similar_days = historical_df[(historical_df['close'] >= price_min) & (historical_df['close'] <= price_max)]
            if not similar_days.empty:
                predicted_volume = similar_days['volume'].mean()
            else:
                predicted_volume = current_df['volume'].iloc[-5] if len(current_df) >= 5 else current_df['volume'].iloc[0]

        new_row = pd.DataFrame([{
            'record_date': date,
            'close': predicted_close,
            'volume': predicted_volume,
            'open': predicted_close + open_offset,
            'high': max(predicted_close, predicted_close + high_offset),
            'low': min(predicted_close, predicted_close + low_offset),
        }])
        
        temp_df = pd.concat([current_df, new_row], ignore_index=True)
        temp_df = add_date_features(temp_df)
        temp_df = add_technical_features(temp_df)
        temp_df = add_dow_theory_features(temp_df)
        
        new_sequence_row = temp_df[feature_columns].iloc[-1].values
        last_sequence = np.vstack([last_sequence[1:], new_sequence_row])
        
        future_predictions.append(new_row.iloc[0])
        current_df = temp_df

    predictions_df = pd.DataFrame(future_predictions)
    predicted_prices = predictions_df['close'].values
    predicted_volumes = predictions_df['volume'].values

    return np.maximum(0, predicted_prices), np.maximum(0, predicted_volumes)


def create_future_dates(
    last_date_str: str,
    periods: int,
    freq: str = "B"
) -> DatetimeIndex:
    """指定された最終日から将来の営業日の日付インデックスを生成します。"""
    last_date = to_datetime(last_date_str)
    future_dates = date_range(start=last_date + DateOffset(days=1), periods=periods, freq=freq)
    return future_dates
