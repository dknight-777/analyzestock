import os
import random
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# Import refactored modules
from data_processing import create_sequences
from db_utils import get_db_engine, get_stock_data
from models import get_model
from plotting import plot_prediction_chart
from training import train_model, predict_future_values
from evaluation import evaluate_model

# 再現性のためのシード固定
# --- Seed for reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    # 引数を解析
    parser = argparse.ArgumentParser(description="株価予測チャートを作成します。")
    parser.add_argument("--stock_code", type=str, default="9432", help="証券コード")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lstm", "nn", "gru", "transformer"],
        default=None,
        help="モデルの種類 (lstm, nn, gru, transformer)。指定しない場合は全てのモデルが実行されます。",
    )
    parser.add_argument(
        "--time_frame",
        type=str,
        choices=["daily", "weekly"],
        default="daily",
        help="時間枠 (daily or weekly)",
    )
    parser.add_argument("--epochs", type=int, default=150, help="学習のエポック数")
    parser.add_argument("--seq_length", type=int, default=30, help="シーケンス長")
    parser.add_argument("--fut_pred", type=int, default=5, help="予測期間（営業日数）")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="使用するデバイス (cpu or cuda)",
    )
    args = parser.parse_args()

    # デバイスの確認
    device_str = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # 株価データを取得
    engine = get_db_engine()
    if not engine:
        sys.exit(1)

    df = get_stock_data(engine, args.stock_code)

    if df.empty:
        print(f"銘柄コード {args.stock_code} のデータが見つかりませんでした。")
        sys.exit(1)

    stock_name = (
        df["stock_name"].dropna().iloc[0]
        if not df["stock_name"].dropna().empty
        else f"銘柄コード {args.stock_code}"
    )

    # --- 1. 特徴量エンジニアリング ---
    df["record_date"] = pd.to_datetime(df["record_date"])
    df.set_index("record_date", inplace=True)
    if args.time_frame == "daily":
        df = df.resample("D").last()
    else:
        df = df.resample("W").last()
    df.reset_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)

    df["day_of_week"] = df["record_date"].dt.dayofweek
    df["day_of_month"] = df["record_date"].dt.day
    df["month"] = df["record_date"].dt.month
    df["year"] = df["record_date"].dt.year
    df["day_of_year"] = df["record_date"].dt.dayofyear
    df["week_of_year"] = df["record_date"].dt.isocalendar().week.astype(int)

    for col, max_val in [
        ("day_of_week", 7),
        ("day_of_month", 31),
        ("month", 12),
        ("day_of_year", 366),
        ("week_of_year", 53),
    ]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)

    # --- 2. 特徴量とスケーリング ---
    feature_columns = [
        "close",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_month_sin", "day_of_month_cos",
        "month_sin", "month_cos",
        "day_of_year_sin", "day_of_year_cos",
        "week_of_year_sin", "week_of_year_cos",
        "year"
    ]
    features = df[feature_columns].copy()

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # --- 3. シーケンス作成とデータ分割 ---
    X, y = create_sequences(features_scaled, args.seq_length)

    # バックテスト用にデータを分割 (最後のN日間をテストデータとする)
    test_period = args.fut_pred
    X_train, y_train = X[:-test_period], y[:-test_period]
    X_test, y_test = X[-test_period:], y[-test_period:]

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 32
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # --- 4. モデル学習・評価・予測ループ ---
    models_to_run = (
        [args.model_type]
        if args.model_type
        else ["lstm", "nn", "gru", "transformer"]
    )
    
    future_dates = pd.bdate_range(
        start=df["record_date"].max() + pd.offsets.BDay(), periods=args.fut_pred
    )

    evaluation_results = []

    for model_type in models_to_run:
        print(f"\n--- Running prediction for model: {model_type} ---\
")

        model = get_model(
            model_type=model_type,
            input_dim=len(feature_columns),
            seq_length=args.seq_length
        ).to(device)

        model = train_model(model, train_loader, args.epochs, device)

        backtest_predictions, metrics = evaluate_model(
            model, test_loader, scaler, device, len(feature_columns)
        )
        metrics["model"] = model_type
        evaluation_results.append(metrics)

        predictions = predict_future_values(
            model,
            features_scaled,
            future_dates,
            scaler,
            args.seq_length,
            device,
            feature_columns,
        )

        plot_prediction_chart(
            df,
            predictions,
            future_dates,
            args.stock_code,
            stock_name,
            model_type,
            args.time_frame,
            backtest_data={
                "dates": df["record_date"].tail(test_period),
                "predictions": backtest_predictions,
            },
        )

    # --- 5. 最終評価結果 ---
    if len(evaluation_results) > 1:
        results_df = pd.DataFrame(evaluation_results).set_index("model")
        print("\n--- 全モデルの最終評価結果 ---")
        print(results_df.round(4))
        print("\n--- 評価指標の見方 ---")
        print("RMSE (二乗平均平方根誤差): 値が小さいほど良いです。")
        print("MAE (平均絶対誤差): 値が小さいほど良いです。")
        print("R2スコア (決定係数): 値が1に近いほど良いです。")
        print("----------------------")
        best_model = results_df["R2 Score"].idxmax()
        print(f"\n最も優れたモデル (R2スコア基準): {best_model}")
        print("---------------------------------")


if __name__ == "__main__":
    main()
