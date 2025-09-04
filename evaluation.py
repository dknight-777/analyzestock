import numpy as np
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
) -> tuple[np.ndarray, dict[str, float]]:
    """
    学習済みモデルをテストデータで評価します。
    予測値と評価指標の辞書を返します。
    """
    model.eval()
    predictions_normalized = []
    actuals_normalized = []

    with torch.no_grad():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            predictions_normalized.extend(y_pred[:, 0].cpu().numpy())
            actuals_normalized.extend(labels[:, 0].cpu().numpy())

    # スケールを元に戻す
    # 予測値と実績値の両方を、元の多次元の形に擬似的に復元してからinverse_transformを呼び出す
    dummy_predictions = np.zeros((len(predictions_normalized), num_features))
    dummy_predictions[:, 0] = predictions_normalized
    predictions_rescaled = scaler.inverse_transform(dummy_predictions)[:, 0]

    dummy_actuals = np.zeros((len(actuals_normalized), num_features))
    dummy_actuals[:, 0] = actuals_normalized
    actuals_rescaled = scaler.inverse_transform(dummy_actuals)[:, 0]

    # 評価指標を計算
    rmse = np.sqrt(mean_squared_error(actuals_rescaled, predictions_rescaled))
    mae = mean_absolute_error(actuals_rescaled, predictions_rescaled)
    r2 = r2_score(actuals_rescaled, predictions_rescaled)

    metrics = {"RMSE": rmse, "MAE": mae, "R2 Score": r2}

    # 評価指標を表示
    print("\n--- バックテスト評価 ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("-----------------------\n")

    # グラフ描画用に、実績値ではなく予測値のみを返す
    return predictions_rescaled, metrics
