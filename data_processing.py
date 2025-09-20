import numpy as np
import pandas as pd
from typing import Tuple
import os
import glob
from datetime import datetime

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


def calculate_macd(series: pd.Series, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence) を計算します。"""
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ボリンジャーバンドを計算します。"""
    middle_band = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band, middle_band, lower_band


def update_interest_rate_data(start_date: datetime, end_date: datetime) -> dict[str, pd.DataFrame]:
    """
    FREDから各種金利データをダウンロードし、更新・キャッシュ管理を行う。
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        print("pandas-datareaderがインストールされていません。`pip install pandas-datareader` を実行してください。")
        return {}

    interest_rate_symbols = {
        "jp_10y_yield": "IRLTLT01JPM156N",
        "us_10y_yield": "DGS10",
        "eu_10y_yield": "IRLTLT01EZM156N",
    }
    
    all_rates_df = {}
    now = datetime.now()

    for name, symbol in interest_rate_symbols.items():
        df_cache, cache_file_path = load_latest_data_from_csv(symbol)
        download_required = True

        if df_cache is not None and cache_file_path:
            try:
                cache_filename = os.path.basename(cache_file_path)
                date_str = cache_filename.split('_')[0]
                time_str = cache_filename.split('_')[1]
                cache_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

                today = now.date()
                cache_date = cache_datetime.date()

                if symbol == "DGS10": # 米国10年債は日次データとして扱う
                    if cache_date < today:
                        print(f"Cached data for {name} is old. Downloading new data.")
                        download_required = True
                    else:
                        print(f"Using cached data for {name} ({symbol}).")
                        df_cache['record_date'] = pd.to_datetime(df_cache['record_date'])
                        all_rates_df[name] = df_cache
                        download_required = False
                else: # 日本と欧州の10年債は月次データとして扱う
                    # Condition 1: If it's the 16th of the month
                    if today.day == 16:
                        # Download if cache is older than the 16th of the current month
                        if cache_date.month != today.month or cache_date.year != today.year or cache_date.day < 16:
                            print(f"It's the 16th and cached data for {name} is old. Downloading new data.")
                            download_required = True
                        else:
                            print(f"It's the 16th, but cached data for {name} is already up-to-date for this month. Using cached data.")
                            df_cache['record_date'] = pd.to_datetime(df_cache['record_date'])
                            all_rates_df[name] = df_cache
                            download_required = False
                    # Condition 2: If it's NOT the 16th of the month
                    else:
                        # Download if cache is not from today
                        if cache_date < today:
                            print(f"Cached data for {name} is old. Downloading new data.")
                            download_required = True
                        else:
                            print(f"Using cached data for {name} ({symbol}).")
                            df_cache['record_date'] = pd.to_datetime(df_cache['record_date'])
                            all_rates_df[name] = df_cache
                            download_required = False

            except (ValueError, IndexError) as e:
                print(f"Could not parse date from cache file name for {symbol}. Will download new data.")
        
        if download_required:
            print(f"Downloading new data for {name} ({symbol})...")
            try:
                df_downloaded = web.DataReader(symbol, 'fred', start_date, end_date)
                df_downloaded.reset_index(inplace=True)
                df_downloaded.rename(columns={'DATE': 'record_date', symbol: 'close'}, inplace=True)
                df_downloaded['record_date'] = pd.to_datetime(df_downloaded['record_date'])
                
                # 株価データに合わせてカラムを整形
                df_downloaded['open'] = df_downloaded['close']
                df_downloaded['high'] = df_downloaded['close']
                df_downloaded['low'] = df_downloaded['close']
                df_downloaded['volume'] = 0

                save_data_to_csv(df_downloaded, symbol)
                all_rates_df[name] = df_downloaded
            except Exception as e:
                print(f"Failed to download data for {symbol}: {e}")
                if df_cache is not None:
                    print(f"Falling back to cached data for {name} ({symbol}).")
                    df_cache['record_date'] = pd.to_datetime(df_cache['record_date'])
                    all_rates_df[name] = df_cache
                else:
                    continue # キャッシュもなければスキップ

    return all_rates_df


def add_dow_theory_features(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """ダウ理論に基づいたトレンド特徴量を追加します。"""
    
    highs = df['high']
    lows = df['low']
    
    peak_indices = []
    for i in range(order, len(highs) - order):
        if highs.iloc[i] == highs.iloc[i-order:i+order+1].max():
            peak_indices.append(df.index[i])
            
    trough_indices = []
    for i in range(order, len(lows) - order):
        if lows.iloc[i] == lows.iloc[i-order:i+order+1].min():
            trough_indices.append(df.index[i])

    df['dow_trend'] = 0
    
    extrema = []
    for idx in peak_indices:
        extrema.append((idx, df.loc[idx, 'high'], 'peak'))
    for idx in trough_indices:
        extrema.append((idx, df.loc[idx, 'low'], 'trough'))
        
    extrema.sort(key=lambda x: x[0])
    
    if len(extrema) < 4:
        return df

    for i in range(len(extrema) - 3):
        e1, e2, e3, e4 = extrema[i:i+4]
        
        # e1: trough, e2: peak, e3: trough, e4: peak
        if e1[2] == 'trough' and e2[2] == 'peak' and e3[2] == 'trough' and e4[2] == 'peak':
            if e3[1] > e1[1] and e4[1] > e2[1]: # higher low, higher high
                df.loc[e1[0]:e4[0], 'dow_trend'] = 1
                
        # e1: peak, e2: trough, e3: peak, e4: trough
        elif e1[2] == 'peak' and e2[2] == 'trough' and e3[2] == 'peak' and e4[2] == 'trough':
            if e3[1] < e1[1] and e4[1] < e2[1]: # lower high, lower low
                df.loc[e1[0]:e4[0], 'dow_trend'] = -1

    return df


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

    # MACD
    macd, macd_signal, macd_hist = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

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

def save_data_to_csv(df: pd.DataFrame, ticker_code: str, data_dir: str = "data") -> str:
    """DataFrameをタイムスタンプ付きのCSVファイルに保存します。"""
    os.makedirs(data_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ticker_code may contain characters that are not suitable for file names, so we sanitize it.
    sanitized_ticker = "".join(c for c in ticker_code if c.isalnum())
    file_path = os.path.join(data_dir, f"{date_str}_{sanitized_ticker}.csv")
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    return file_path

def load_latest_data_from_csv(ticker_code: str, data_dir: str = "data") -> Tuple[pd.DataFrame | None, str | None]:
    """指定されたticker_codeの最新のCSVファイルを読み込み、DataFrameとファイルパスを返します。"""
    sanitized_ticker = "".join(c for c in ticker_code if c.isalnum())
    search_pattern = os.path.join(data_dir, f"*_{sanitized_ticker}.csv")
    file_list = glob.glob(search_pattern)
    if not file_list:
        return None, None
    
    latest_file = max(file_list, key=os.path.getctime)
    print(f"Loading data from {latest_file}")
    df = pd.read_csv(latest_file)
    # record_dateをdatetime型に変換
    if 'record_date' in df.columns:
        df['record_date'] = pd.to_datetime(df['record_date'])
    return df, latest_file


if __name__ == "__main__":
    # --- テスト用のデータフレームを作成 ---
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="D"))
    data = {
        "record_date": dates,
        "open": np.random.uniform(95, 105, 100),
        "high": np.random.uniform(100, 110, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(98, 108, 100),
        "volume": np.random.randint(10000, 50000, 100),
    }
    test_df = pd.DataFrame(data)

    # --- 各関数のテスト ---
    print("--- Testing add_technical_features ---")
    tech_df = add_technical_features(test_df.copy().set_index("record_date"))
    print(tech_df.head())
    print(tech_df.tail())
    print("\n--- Testing add_date_features ---")
    date_df = add_date_features(test_df.copy())
    print(date_df.head())
    print(date_df.tail())

    # --- create_sequencesのテスト ---
    print("\n--- Testing create_sequences ---")
    test_data = np.arange(100).reshape(20, 5)  # 20 samples, 5 features
    seq_len = 3
    target_feature_idx = 0  # 最初の特徴量をターゲットとする
    X, y = create_sequences(test_data, seq_len, target_feature_idx)
    print("Shape of X:", X.shape)  # Expected: (17, 3, 5)
    print("Shape of y:", y.shape)  # Expected: (17, 1)
    print("First sequence (X[0]):\n", X[0])
    print("First target (y[0]):", y[0])  # Expected: test_data[3, 0] = 15
    print("Last sequence (X[-1]):\n", X[-1])
    print("Last target (y[-1]):", y[-1])  # Expected: test_data[19, 0] = 95

    # --- Testing save_data_to_csv and load_latest_data_from_csv ---
    print("\n--- Testing CSV functions ---")
    ticker = "9999.T"
    save_data_to_csv(test_df, ticker)
    loaded_df, _ = load_latest_data_from_csv(ticker)
    if loaded_df is not None:
        print("Loaded DataFrame successfully.")
        # Check if dtypes are correct
        print(loaded_df.dtypes)
        assert loaded_df['record_date'].dtype == '<M8[ns]>'
        print("record_date dtype is correct.")
        # Clean up created dummy files and directory
        files = glob.glob(f"data/*_{''.join(c for c in ticker if c.isalnum())}.csv")
        for f in files:
            os.remove(f)
        if os.path.exists("data"):
            os.rmdir("data")