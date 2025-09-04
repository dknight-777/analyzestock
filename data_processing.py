import numpy as np
import pandas as pd
from typing import Tuple


def create_sequences(
    data: np.ndarray, seq_length: int, close_idx: int = 0, volume_idx: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """時系列データを教師あり学習用のシーケンスデータに変換します。"""
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length), :]
        y = data[i + seq_length, [close_idx, volume_idx]]  # 予測対象は終値と出来高
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) を計算します。"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームにテクニカル指標を追加します。"""
    df['price_change_ratio'] = (df['close'] - df['open']) / df['open'].replace(0, np.nan)
    df['price_range_ratio'] = (df['high'] - df['low']) / df['low'].replace(0, np.nan)
    df['rsi'] = calculate_rsi(df['close'])
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームに日付関連の特徴量を追加します。"""
    # record_dateがインデックスにあるか列にあるかを確認
    if isinstance(df.index, pd.DatetimeIndex):
        date_accessor = df.index
    elif 'record_date' in df.columns:
        # .dt accessor を使うために、必ずdatetime型に変換
        date_accessor = pd.to_datetime(df['record_date']).dt
    else:
        raise ValueError("DataFrameに 'record_date' 列が見つからないか、DatetimeIndexが設定されていません。")

    df["day_of_week"] = date_accessor.dayofweek
    df["day_of_month"] = date_accessor.day
    df["month"] = date_accessor.month
    df["year"] = date_accessor.year
    df["day_of_year"] = date_accessor.dayofyear
    df["week_of_year"] = date_accessor.isocalendar().week.astype(int)

    for col, max_val in [
        ("day_of_week", 7),
        ("day_of_month", 31),
        ("month", 12),
        ("day_of_year", 366),
        ("week_of_year", 53),
    ]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df