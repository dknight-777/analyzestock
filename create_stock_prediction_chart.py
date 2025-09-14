import os
import random
import sys
import argparse
import uuid
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# Import refactored modules
from data_processing import (
    create_sequences,
    add_technical_features,
    add_date_features,
    add_dow_theory_features,
)
from db_utils import (
    get_db_engine,
    get_stock_data,
    save_prediction_run,
    save_prediction_chart,
    save_stock_predictions,
    save_run_evaluations,
)
from models import get_model
from plotting import plot_prediction_chart
from training import train_model, predict_future_values
from evaluation import evaluate_model

# --- Program Version ---
__version__ = "0.7"

# 再現性のためのシード固定
# --- Seed for reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    # --- バージョン情報を表示 ---
    print(f"Stock Price Prediction v{__version__}\n---")

    # 引数を解析
    parser = argparse.ArgumentParser(description="株価予測チャートを作成します。")
    parser.add_argument("--stock_code", type=str, default="9432", help="証券コード")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lstm", "nn", "gru"],
        default="nn",
        help="モデルの種類 (lstm, nn, gru)。",
    )
    parser.add_argument(
        "--time_frame",
        type=str,
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="時間枠 (daily, weekly, monthly)",
    )
    parser.add_argument("--epochs", type=int, default=150, help="学習のエポック数")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="学習の進捗を表示する間隔"
    )
    parser.add_argument("--seq_length", type=int, default=30, help="シーケンス長")
    parser.add_argument("--fut_pred", type=int, default=5, help="予測期間（営業日数）")
    parser.add_argument("--batch_size", type=int, default=32, help="バッチサイズ")
    parser.add_argument(
        "--hidden_unit_size",
        type=int,
        default=128,
        help="隠れユニット数 (LSTM, GRU用)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="RNN層の数 (LSTM, GRU用)"
    )
    parser.add_argument(
        "--nn_layer_units",
        type=int,
        nargs="+",
        default=[4096, 2048, 1024, 512, 128, 64],
        help="NNモデルの各隠れ層のユニット数をスペース区切りで指定 (例: 100 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="使用するデバイス (cpu or cuda)",
    )
    args = parser.parse_args()

    # --- デバッグ: 引数の値を確認 ---
    print("\n--- Parsed Arguments ---")
    print(args)
    print("------------------------\n")

    # --- 0. 準備 ---
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    engine = get_db_engine()
    if not engine:
        sys.exit(1)

    prediction_batch_id = uuid.uuid4()
    print(f"Prediction Batch ID: {prediction_batch_id}")

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

    time_frame_options = {
        "daily": {"freq": "B", "offset": pd.offsets.BDay()},
        "weekly": {"freq": "W", "offset": pd.offsets.Week()},
        "monthly": {"freq": "ME", "offset": pd.offsets.MonthEnd()},
    }
    resample_freq = time_frame_options[args.time_frame]["freq"]

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df = df.resample(resample_freq).agg(ohlc_dict)
    df.reset_index(inplace=True)
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    df = add_technical_features(df)
    df = add_date_features(df)
    df = add_dow_theory_features(df)
    df.dropna(inplace=True)
    df = df[df["volume"] > 0].copy()

    # --- 2. データ分割とスケーリング ---
    feature_columns = [
        "close",
        "open",
        "high",
        "low",
        "volume",
        "price_change_ratio",
        "price_range_ratio",
        "rsi",
        "atr",
        "rci",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_percent",
        "macd",
        "macd_signal",
        "macd_hist",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_month_sin",
        "day_of_month_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "week_of_year_sin",
        "week_of_year_cos",
        "year",
        "log_return",
        "dow_trend",
    ]

    test_size = 30
    if len(df) <= test_size + args.seq_length:
        print("データが少なく、学習と評価を分割できません。")
        sys.exit(1)

    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_columns])

    train_scaled = scaler.transform(train_df[feature_columns])
    test_scaled = scaler.transform(test_df[feature_columns])

    # --- モデル学習・評価・予測ループ ---
    models_to_run = [args.model_type] if args.model_type else ["lstm", "nn", "gru"]

    date_offset = time_frame_options[args.time_frame]["offset"]
    future_dates = pd.date_range(
        start=df["record_date"].max() + date_offset,
        periods=args.fut_pred,
        freq=date_offset,
    )

    evaluation_results = []
    updated_by_script = os.path.basename(__file__)

    try:
        with engine.connect() as connection:
            with connection.begin() as transaction:
                print(
                    "\n--- データベースへの保存処理を開始します（トランザクション開始） ---"
                )
                try:
                    save_prediction_run(
                        connection, prediction_batch_id, args, updated_by_script
                    )

                    for model_type in models_to_run:
                        print(f"\n--- Running prediction for model: {model_type} ---")

                        # --- 3. シーケンス作成 ---
                        log_return_idx = feature_columns.index("log_return")

                        X_train, y_train = create_sequences(
                            train_scaled, args.seq_length, log_return_idx
                        )

                        padding_for_test = train_scaled[-args.seq_length :]
                        combined_for_test = np.concatenate(
                            [padding_for_test, test_scaled]
                        )
                        X_test, y_test = create_sequences(
                            combined_for_test, args.seq_length, log_return_idx
                        )

                        X_train_tensor = torch.from_numpy(X_train).float()
                        y_train_tensor = torch.from_numpy(y_train).float()
                        X_test_tensor = torch.from_numpy(X_test).float()
                        y_test_tensor = torch.from_numpy(y_test).float()

                        train_data = TensorDataset(X_train_tensor, y_train_tensor)
                        test_data = TensorDataset(X_test_tensor, y_test_tensor)

                        train_loader = DataLoader(
                            train_data,
                            shuffle=False,
                            batch_size=args.batch_size,
                            drop_last=True,
                        )
                        test_loader = DataLoader(
                            test_data, shuffle=False, batch_size=args.batch_size
                        )

                        # --- 4. モデルの取得と学習 ---
                        model = get_model(
                            model_type=model_type,
                            input_dim=len(feature_columns),
                            seq_length=args.seq_length,
                            output_dim=1,
                            hidden_unit_size=args.hidden_unit_size,
                            num_layers=args.num_layers,
                            nn_layer_units=args.nn_layer_units,
                        ).to(device)

                        model = train_model(
                            model,
                            train_loader,
                            args.epochs,
                            device,
                            log_interval=args.log_interval,
                        )

                        # --- 5. 評価と予測 ---
                        close_idx = feature_columns.index("close")
                        backtest_predictions, metrics = evaluate_model(
                            model,
                            test_loader,
                            scaler,
                            device,
                            len(feature_columns),
                            close_idx,
                            log_return_idx,
                            test_df,
                        )
                        metrics["model"] = model_type
                        evaluation_results.append(metrics)

                        if args.fut_pred > 0:
                            predictions, pred_volumes = predict_future_values(
                                model,
                                df.copy(),
                                future_dates,
                                scaler,
                                args.seq_length,
                                device,
                                feature_columns,
                                log_return_idx,
                            )
                            predictions_list = predictions.flatten().tolist()
                            pred_volumes_list = pred_volumes.flatten().tolist()

                            print(f"\n--- 予測詳細 (モデル: {model_type}) ---")
                            details_df = pd.DataFrame(
                                {
                                    "Date": future_dates,
                                    "Prediction": predictions_list,
                                    "Assumed Volume": pred_volumes_list,
                                }
                            )
                            print(details_df.to_string(index=False))
                        else:
                            predictions = np.array([])
                            pred_volumes = np.array([])

                        chart_binary = plot_prediction_chart(
                            df,
                            predictions,
                            future_dates,
                            args.stock_code,
                            stock_name,
                            model_type,
                            args.time_frame,
                            backtest_data={
                                "dates": test_df["record_date"].iloc[
                                    1 : len(backtest_predictions) + 1
                                ],
                                "predictions": backtest_predictions,
                            },
                        )
                        print(f"グラフをメモリ上に生成しました。")

                        chart_id = save_prediction_chart(
                            connection,
                            prediction_batch_id,
                            model_type,
                            chart_binary,
                            updated_by_script,
                        )

                        if chart_id and args.fut_pred > 0:
                            predictions_df = pd.DataFrame(
                                {
                                    "date": future_dates,
                                    "prediction": predictions.flatten(),
                                    "volume": pred_volumes.flatten(),
                                }
                            )
                            save_stock_predictions(
                                connection,
                                prediction_batch_id,
                                chart_id,
                                model_type,
                                args.stock_code,
                                predictions_df,
                                updated_by_script,
                            )

                    if evaluation_results:
                        save_run_evaluations(
                            connection,
                            prediction_batch_id,
                            evaluation_results,
                            updated_by_script,
                        )

                    print(
                        "\n--- データベースへの保存処理が正常に完了しました（トランザクションコミット） ---"
                    )

                except Exception as e:
                    print(
                        "\n--- トランザクション内でエラーが発生しました。処理をロールバックします。 ---"
                    )
                    traceback.print_exc()
                    transaction.rollback()
                    raise

    except Exception as e:
        print(f"\n--- データベース接続またはトランザクション開始に失敗しました。 ---")
        traceback.print_exc()

    # --- 6. 最終評価結果の表示 ---
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results).set_index("model")
        print("\n--- 全モデルの最終評価結果 ---")
        print(results_df.round(4))
        print("\n--- 評価指標の見方 ---")
        print("RMSE (二乗平均平方根誤差): 値が小さいほど良いです。")
        print("MAE (平均絶対誤差): 値が小さいほど良いです。")
        print("R2スコア (決定係数): 値が1に近いほど良いです。")
        print("【注意】")
        print("これらの指標は予測と実績の誤差の平均を示すものです。")
        print(
            "そのため、スコアが良い場合でも、単に一日遅れで価格を追従しているだけで、"
        )
        print("価格の転換点を予測できていない可能性があります。")
        print(
            "最終的にはグラフの形状も合わせて、総合的にモデルの良し悪しを判断することが重要です。"
        )
        print("----------------------")
        if len(results_df) > 1:
            best_model = results_df["R2 Score"].idxmax()
            print(f"\n最も優れたモデル (R2スコア基準): {best_model}")
            print("---------------------------------")


if __name__ == "__main__":
    main()
