import numpy as np
import pandas as pd
from arch import arch_model
from typing import Tuple
from tqdm import tqdm

def fit_garch_and_forecast_volatility(
    log_returns: pd.Series, p: int = 1, q: int = 1, forecast_horizon: int = 5
) -> np.ndarray:
    """
    Fits a GARCH(p, q) model to the log returns and forecasts future conditional volatility.

    Args:
        log_returns (pd.Series): Series of logarithmic returns of the stock price.
        p (int): The order (lag) of the GARCH term.
        q (int): The order (lag) of the ARCH term.
        forecast_horizon (int): The number of steps to forecast ahead.

    Returns:
        np.ndarray: An array of forecasted conditional volatilities (standard deviations).
    """
    # 対数収益率を100倍してスケーリング
    scaled_log_returns = log_returns.dropna() * 100

    # GARCHモデルを定義
    model = arch_model(
        scaled_log_returns, vol="Garch", p=p, q=q, mean="Zero", dist="Normal"
    )

    # モデルをフィット
    res = model.fit(disp="off")

    # 将来のボラティリティを予測
    forecast = res.forecast(horizon=forecast_horizon, reindex=False)

    # 予測された分散を取得し、標準偏差（ボラティリティ）に変換
    forecasted_variance = forecast.variance.values[-1, :]
    forecasted_volatility = np.sqrt(forecasted_variance)

    # スケールを元に戻す
    return forecasted_volatility / 100


def run_gbm_garch_simulation(
    last_price: float,
    drift: float,
    forecasted_volatilities: np.ndarray,
    num_simulations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a Monte Carlo simulation for future stock prices using a GBM framework
    with time-varying volatilities from a GARCH model.

    Args:
        last_price (float): The last known closing price of the stock.
        drift (float): The drift term (e.g., average log return).
        forecasted_volatilities (np.ndarray): Array of forecasted future volatilities.
        num_simulations (int): The number of Monte Carlo simulations to run.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - An array of all simulated price paths (num_simulations, forecast_horizon).
            - An array representing the median price path.
    """
    forecast_horizon = len(forecasted_volatilities)
    dt = 1  # Time step is 1 day (or 1 period)

    # シミュレーション結果を格納する配列
    price_paths = np.zeros((num_simulations, forecast_horizon))

    for i in range(num_simulations):
        price = last_price
        path = []
        for t in range(forecast_horizon):
            # GARCHから予測されたボラティリティを使用
            sigma_t = forecasted_volatilities[t]
            # 標準正規分布に従うランダム変数を生成
            Z = np.random.standard_normal()
            # GBMの式に基づいて次の価格を計算
            # S_t = S_{t-1} * exp((mu - 0.5 * sigma_t**2) * dt + sigma_t * sqrt(dt) * Z)
            price = price * np.exp((drift - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * Z)
            path.append(price)
        price_paths[i, :] = path

    # シミュレーション結果の中央値を計算
    median_path = np.median(price_paths, axis=0)

    return price_paths, median_path

def predict_with_gbm_garch(
    df: pd.DataFrame, fut_pred: int, test_size: int, num_simulations: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orchestrates the GARCH+GBM prediction process, including backtesting.

    Args:
        df (pd.DataFrame): DataFrame containing historical data, including 'log_return' and 'close'.
        fut_pred (int): The number of future days to predict.
        test_size (int): The size of the test set for backtesting.
        num_simulations (int): The number of Monte Carlo simulations to run.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - The median path of the future prediction.
            - The median path of the backtest prediction.
            - All simulated paths for the future prediction.
    """
    # --- 1. バックテスト用の予測 ---
    print("\n--- Running Backtesting for GBM-GARCH ---")
    backtest_predictions = []
    # バックテスト期間を一つずつ進めながら予測を行う
    for i in tqdm(range(test_size), desc="Backtesting"):
        # 学習データはテストデータの手前まで
        train_data = df.iloc[: -(test_size - i)]
        log_returns_train = train_data['log_return'].dropna()
        last_price_train = train_data['close'].iloc[-1]
        drift_train = log_returns_train.mean()

        # 1ステップ先のボラティリティを予測
        vol_forecast = fit_garch_and_forecast_volatility(log_returns_train, forecast_horizon=1)

        # 1ステップ先の価格をシミュレーション
        _, median_sim_price = run_gbm_garch_simulation(
            last_price=last_price_train,
            drift=drift_train,
            forecasted_volatilities=vol_forecast,
            num_simulations=num_simulations,
        )
        backtest_predictions.append(median_sim_price[0])

    # --- 2. 将来予測 ---
    print("\n--- Running Future Prediction for GBM-GARCH ---")
    log_returns_full = df['log_return'].dropna()
    last_price_full = df['close'].iloc[-1]
    drift_full = log_returns_full.mean()

    # 将来のボラティリティを予測
    future_volatilities = fit_garch_and_forecast_volatility(
        log_returns_full, forecast_horizon=fut_pred
    )

    # 将来の価格をシミュレーション
    all_future_paths, median_future_path = run_gbm_garch_simulation(
        last_price=last_price_full,
        drift=drift_full,
        forecasted_volatilities=future_volatilities,
        num_simulations=num_simulations,
    )

    return median_future_path, np.array(backtest_predictions), all_future_paths
