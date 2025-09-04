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
from data_processing import create_sequences, add_technical_features, add_date_features
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

# 再現性のためのシード固定
# --- Seed for reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    # 引数を解析
    parser = argparse.ArgumentParser(description="株価予測チャートを作成します。 সন")
    parser.add_argument("--stock_code", type=str, default="9432", help="証券コード")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lstm", "nn", "gru"],
        default=None,
        help="モデルの種類 (lstm, nn, gru)。指定しない場合は全てのモデルが実行されます。",
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
        default=None, # デフォルトはNoneにして後で設定
        help="使用するデバイス (cpu or cuda)",
    )
    args = parser.parse_args()

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

    # 時間枠に応じてリサンプリング
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    if args.time_frame == "daily":
        df = df.resample("B").agg(ohlc_dict)
    else:
        df = df.resample("W").agg(ohlc_dict)
    
    df.reset_index(inplace=True)
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # 特徴量生成関数を呼び出す
    df = add_technical_features(df)
    df = add_date_features(df)

    # 特徴量生成で発生したNaNを持つ行を削除
    df.dropna(inplace=True)

    # Volumeが0の異常データを学習から除外
    df = df[df['volume'] > 0].copy()

    # --- 2. 特徴量とスケーリング ---
    feature_columns = [
        "close", "open", "high", "low", "volume",
        "price_change_ratio", "price_range_ratio", "rsi",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_month_sin", "day_of_month_cos",
        "month_sin", "month_cos",
        "day_of_year_sin", "day_of_year_cos",
        "week_of_year_sin", "week_of_year_cos",
        "year"
    ]
    features = df[feature_columns].copy()
    
    # 全特徴量のスケーラー
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # --- モデル学習・評価・予測ループ ---
    models_to_run = (
        [args.model_type]
        if args.model_type
        else ["lstm", "nn", "gru"]
    )
    
    future_dates = pd.bdate_range(
        start=df["record_date"].max() + pd.offsets.BDay(), periods=args.fut_pred
    )

    evaluation_results = []
    updated_by_script = os.path.basename(__file__)

    try:
        with engine.connect() as connection:
            with connection.begin() as transaction:
                print("\n--- データベースへの保存処理を開始します（トランザクション開始） ---")
                try:
                    save_prediction_run(connection, prediction_batch_id, args, updated_by_script)

                    for model_type in models_to_run:
                        print(f"\n--- Running prediction for model: {model_type} ---")

                        # --- 3. シーケンス作成とデータ分割 ---
                        close_idx = feature_columns.index('close')
                        volume_idx = feature_columns.index('volume')
                        
                        # スケール済みデータからシーケンスを作成
                        df_scaled = pd.DataFrame(features_scaled, columns=feature_columns)
                        X, y = create_sequences(df_scaled.values, args.seq_length, close_idx=close_idx, volume_idx=volume_idx)

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
                        
                        # --- 4. モデルの取得と学習 ---
                        model = get_model(
                            model_type=model_type,
                            input_dim=len(feature_columns),
                            seq_length=args.seq_length,
                            output_dim=2
                        ).to(device)

                        model = train_model(model, train_loader, args.epochs, device)

                        # --- 5. 評価と予測 ---
                        backtest_predictions, metrics = evaluate_model(
                            model, test_loader, scaler, device, len(feature_columns)
                        )
                        metrics["model"] = model_type
                        evaluation_results.append(metrics)

                        # 予測関数にはスケール前のdfを渡すように変更
                        predictions, pred_volumes = predict_future_values(
                            model,
                            df.copy(), # スケール前の元データを渡す
                            future_dates,
                            scaler,
                            args.seq_length,
                            device,
                            feature_columns,
                        )

                        # Ensure predictions and pred_volumes are Python native lists of scalars
                        predictions_list = predictions.flatten().tolist()
                        pred_volumes_list = pred_volumes.flatten().tolist()

                        print(f"\n--- 予測詳細 (モデル: {model_type}) ---")
                        details_df = pd.DataFrame({
                            'Date': future_dates,
                            'Prediction': predictions_list,
                            'Assumed Volume': pred_volumes_list
                        })
                        print(details_df.to_string(index=False))

                        chart_binary = plot_prediction_chart(
                            df,
                            predictions,
                            future_dates,
                            args.stock_code,
                            stock_name,
                            model_type,
                            args.time_frame,
                            backtest_data={
                                "dates": df["record_date"].tail(len(backtest_predictions)),
                                "predictions": backtest_predictions,
                            },
                        )
                        print(f"グラフをメモリ上に生成しました。 সন")

                        chart_id = save_prediction_chart(
                            connection, prediction_batch_id, model_type, chart_binary, updated_by_script
                        )

                        if chart_id:
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
                        save_run_evaluations(connection, prediction_batch_id, evaluation_results, updated_by_script)
                    
                    print("\n--- データベースへの保存処理が正常に完了しました（トランザクションコミット） ---")

                except Exception as e:
                    print(f"\n--- トランザクション内でエラーが発生しました。処理をロールバックします。 ---")
                    traceback.print_exc()
                    transaction.rollback()
                    raise

    except Exception as e:
        print(f"\n--- データベース接続またはトランザクション開始に失敗しました。 ---")
        traceback.print_exc()

    # --- 6. 最終評価結果の表示 ---
    if evaluation_results and len(evaluation_results) > 1:
        results_df = pd.DataFrame(evaluation_results).set_index("model")
        print("\n--- 全モデルの最終評価結果 ---")
        print(results_df.round(4))
        print("\n--- 評価指標の見方 ---")
        print("RMSE (二乗平均平方根誤差): 値が小さいほど良いです。 সন")
        print("MAE (平均絶対誤差): 値が小さいほど良いです。 সন")
        print("R2スコア (決定係数): 値が1に近いほど良いです。 সন")
        print("----------------------")
        best_model = results_df["R2 Score"].idxmax()
        print(f"\n最も優れたモデル (R2スコア基準): {best_model}")
        print("---------------------------------")

if __name__ == "__main__":
    main()
