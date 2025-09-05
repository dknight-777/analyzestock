import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from pandas import DatetimeIndex, DataFrame, to_datetime, date_range, DateOffset

from data_processing import add_date_features, add_technical_features

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
    historical_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    scaler: MinMaxScaler,
    seq_length: int,
    device: torch.device,
    feature_columns: list,
) -> tuple[np.ndarray, np.ndarray]:
    """学習済みモデルを使用して将来の値を予測します。特徴量を動的に再計算します。"""
    model.eval()

    close_col_idx = feature_columns.index("close")
    volume_col_index = feature_columns.index("volume")

    open_offset = (historical_df['open'] - historical_df['close']).mean()
    high_offset = (historical_df['high'] - historical_df['close']).mean()
    low_offset = (historical_df['low'] - historical_df['close']).mean()

    predictions_df = historical_df.copy()

    for date in future_dates:
        last_sequence = predictions_df[feature_columns].tail(seq_length)
        last_sequence_scaled = scaler.transform(last_sequence)
        
        seq = torch.from_numpy(last_sequence_scaled).float().unsqueeze(0).to(device)

        with torch.no_grad():
            prediction_normalized = model(seq)[0].cpu().numpy()
            prediction_normalized = np.clip(prediction_normalized, 0, 1)

            dummy_array = np.zeros((1, len(feature_columns)))
            dummy_array[0, close_col_idx] = prediction_normalized[0]
            dummy_array[0, volume_col_index] = prediction_normalized[1]
            inversed_pred = scaler.inverse_transform(dummy_array)
            predicted_close = inversed_pred[0, close_col_idx]

            # --- ユーザー提供のロジックで出来高を決定 ---
            price_change_threshold = 0.005 # 0.5%の変動を「横ばい」の閾値とする
            last_close = predictions_df['close'].iloc[-1]
            price_change_ratio = (predicted_close - last_close) / last_close

            if abs(price_change_ratio) < price_change_threshold:
                # 横ばいの場合: 5営業日前の出来高を使用
                if len(predictions_df) >= 5:
                    predicted_volume = predictions_df['volume'].iloc[-5]
                else:
                    # データが5日未満の場合は利用可能な最も古いデータを使用
                    predicted_volume = predictions_df['volume'].iloc[0]
            else:
                # トレンドがある場合: 過去の同じ価格帯の出来高の平均を使用
                price_range_margin = 0.01 # 1%の価格帯マージン
                price_min = predicted_close * (1 - price_range_margin)
                price_max = predicted_close * (1 + price_range_margin)
                
                similar_days = historical_df[
                    (historical_df['close'] >= price_min) & (historical_df['close'] <= price_max)
                ]
                
                if not similar_days.empty:
                    predicted_volume = similar_days['volume'].mean()
                else:
                    # 該当する過去データがない場合は5営業日前の出来高を使用
                    if len(predictions_df) >= 5:
                        predicted_volume = predictions_df['volume'].iloc[-5]
                    else:
                        predicted_volume = predictions_df['volume'].iloc[0]
            
            # --- 新しい行を作成 ---
            new_row_data = {
                'record_date': date,
                'close': predicted_close,
                'volume': predicted_volume,
                'open': predicted_close + open_offset,
                'high': max(predicted_close, predicted_close + high_offset),
                'low': min(predicted_close, predicted_close + low_offset),
            }
            new_row_df = pd.DataFrame([new_row_data])
            
            predictions_df = pd.concat([predictions_df, new_row_df], ignore_index=True)
            
            predictions_df = add_date_features(predictions_df)
            predictions_df = add_technical_features(predictions_df)
            
            predictions_df[feature_columns] = predictions_df[feature_columns].ffill()
            predictions_df[feature_columns] = predictions_df[feature_columns].bfill()

    future_predictions = predictions_df[predictions_df['record_date'].isin(future_dates)]
    predicted_prices = future_predictions['close'].values
    predicted_volumes = future_predictions['volume'].values

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
