import numpy as np
from typing import Tuple


def create_sequences(
    data: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """時系列データを教師あり学習用のシーケンスデータに変換します。"""
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length), :]
        y = data[i + seq_length, 0]  # 予測対象は最初の列（終値）
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
