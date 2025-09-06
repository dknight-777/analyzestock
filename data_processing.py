import numpy as np
import pandas as pd
from typing import Tuple


def create_sequences(
    data: np.ndarray, seq_length: int, target_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """時系列データを教師あり学習用のシーケンスデータに変換します。"""
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length), :]
        y = data[i + seq_length, target_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) を計算します。"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) を計算します。"""
    high_low = df['high'] - df['low']
    high_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_prev_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_rci(series: pd.Series, period: int = 9) -> pd.Series:
    """RCI (Rank Correlation Index) を計算します。"""
    # 日付のランク (常に 1, 2, ..., period)
    date_rank = np.arange(1, period + 1)
    
    def get_rci(window: pd.Series) -> float:
        # 価格のランクを計算
        price_rank = window.rank(method='first')
        # ランクの差の二乗和 (d) を計算
        d = np.sum((date_rank - price_rank) ** 2)
        # RCIの計算式
        return (1 - (6 * d) / (period * (period**2 - 1))) * 100

    return series.rolling(window=period).apply(get_rci, raw=False)


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ボリンジャーバンドを計算します。"""
    middle_band = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, middle_band, lower_band


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームにテクニカル指標を追加します。"""
    df['price_change_ratio'] = (df['close'] - df['open']) / df['open'].replace(0, np.nan)
    df['price_range_ratio'] = (df['high'] - df['low']) / df['low'].replace(0, np.nan)
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    df['rci'] = calculate_rci(df['close'])

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
    df['bb_percent'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームに日付関連の特徴量を追加します。"""
    if isinstance(df.index, pd.DatetimeIndex):
        date_accessor = df.index
    elif 'record_date' in df.columns:
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
