import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device,
    num_features: int,
    close_idx: int,
    log_return_idx: int,
    test_df: pd.DataFrame
) -> tuple[np.ndarray, dict[str, float]]:
    """
    学習済みモデルをテストデータで評価します。
    予測された対数リターンを価格に変換し、評価指標を計算します。
    """
    model.eval()
    
    predicted_log_returns = []
    with torch.no_grad():
        for seq, _ in test_loader:
            seq = seq.to(device)
            y_pred_scaled = model(seq)
            predicted_log_returns.extend(y_pred_scaled.cpu().numpy())

    predicted_log_returns = np.array(predicted_log_returns).flatten()

    # スケールを元に戻す
    dummy_array = np.zeros((len(predicted_log_returns), num_features))
    dummy_array[:, log_return_idx] = predicted_log_returns
    unscaled_log_returns = scaler.inverse_transform(dummy_array)[:, log_return_idx]

    # 対数リターンを価格に変換
    # 評価期間の最初の価格を取得
    last_known_price = test_df['close'].iloc[0]
    
    predicted_prices = []
    for log_return in unscaled_log_returns:
        next_price = last_known_price * np.exp(log_return)
        predicted_prices.append(next_price)
        # 次の反復のために、価格を更新（実績値ではなく予測値を使う）
        last_known_price = next_price

    predicted_prices = np.array(predicted_prices)
    
    # 実績値（評価期間の2日目から）
    actual_prices = test_df['close'].iloc[1:len(predicted_prices) + 1].values

    # 予測と実績の長さを合わせる
    min_len = min(len(predicted_prices), len(actual_prices))
    predicted_prices = predicted_prices[:min_len]
    actual_prices = actual_prices[:min_len]

    # 評価指標を計算
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    metrics = {"RMSE": rmse, "MAE": mae, "R2 Score": r2}

    print("\n--- バックテスト評価 ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("-----------------------\n")

    # グラフ描画用に、バックテスト期間の予測価格を返す
    return predicted_prices, metrics