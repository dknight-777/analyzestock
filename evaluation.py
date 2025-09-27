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
    # 各予測は、その前日の実際の終値に基づいて計算されるべき
    # これにより、予測誤差の累積を防ぐ
    
    # 前日の終値のリストを取得 (テストセットの最初から最後から2番目まで)
    previous_actual_prices = test_df['close'].iloc[:-1].values

    # 予測と実績の長さを合わせる
    min_len = min(len(unscaled_log_returns), len(previous_actual_prices))
    unscaled_log_returns = unscaled_log_returns[:min_len]
    previous_actual_prices = previous_actual_prices[:min_len]

    # 予測価格を計算
    predicted_prices = previous_actual_prices * np.exp(unscaled_log_returns)
    
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

def evaluate_gbm_garch(backtest_predictions: np.ndarray, test_df: pd.DataFrame) -> dict[str, float]:
    """
    GBM+GARCHモデルのバックテスト結果を評価します。
    """
    actual_prices = test_df['close'].values
    
    # 予測と実績の長さを合わせる
    min_len = min(len(backtest_predictions), len(actual_prices))
    backtest_predictions = backtest_predictions[:min_len]
    actual_prices = actual_prices[:min_len]

    # 評価指標を計算
    rmse = np.sqrt(mean_squared_error(actual_prices, backtest_predictions))
    mae = mean_absolute_error(actual_prices, backtest_predictions)
    r2 = r2_score(actual_prices, backtest_predictions)

    metrics = {"RMSE": rmse, "MAE": mae, "R2 Score": r2}

    print("\n--- バックテスト評価 (GBM-GARCH) ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("------------------------------------\n")

    return metrics