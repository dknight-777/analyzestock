import os
import random
import sys
import argparse
import uuid
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf

# Import refactored modules
from data_processing import (
    create_sequences,
    add_technical_features,
    add_date_features,
    add_dow_theory_features,
    load_latest_data_from_csv,
    save_data_to_csv,
    update_fred_data,
)
from models import get_model
from plotting import plot_prediction_chart
from training import train_model, predict_future_values
from evaluation import evaluate_model, evaluate_gbm_garch
from financial_models import predict_with_gbm_garch

# --- Program Version ---
__version__ = "0.9"

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
        choices=["lstm", "nn", "gru", "gbm_garch"],
        default="lstm",
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
    parser.add_argument("--batch_size", type=int, default=16, help="バッチサイズ")
    parser.add_argument(
        "--hidden_unit_size",
        type=int,
        default=512,
        help="隠れユニット数 (LSTM, GRU用)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="RNN層の数 (LSTM, GRU用)"
    )
    parser.add_argument(
        "--nn_layer_units",
        type=int,
        nargs="+",
        default=[4096, 2048, 1024],
        help="NNモデルの各隠れ層のユニット数をスペース区切りで指定 (例: 100 50)",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=10000,
        help="GBM+GARCHモデルのシミュレーション回数",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="使用するデバイス (cpu or cuda)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグ情報を表示します。",
    )
    parser.add_argument(
        "--test_size", type=int, default=30, help="テストデータセットのサイズ"
    )
    parser.add_argument(
        "--external_indices",
        type=str,
        nargs="+",
        choices=[
            "jp_10y_yield",
            "us_10y_yield",
            "eu_10y_yield",
            "dow_jones",
            "sp500",
            "nikkei_225",
        ],
        default=[],
        help="特徴量として使用する外部指標をスペース区切りで指定します。",
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

    # yfinanceを使用して株価データを取得
    df = None
    now = datetime.now()
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    # 日本の証券コードに対応するため、末尾に ".T" を追加
    if args.stock_code.isdigit() and len(args.stock_code) == 4:
        ticker_code = f"{args.stock_code}.T"
    else:
        ticker_code = args.stock_code  # 米国株などの場合はそのまま

    # --- キャッシュの確認 ---
    df_cache, cache_file_path = load_latest_data_from_csv(
        ticker_code, time_frame=args.time_frame
    )
    download_required = True

    if df_cache is not None and cache_file_path:
        try:
            # ファイル名から日付を取得
            cache_filename = os.path.basename(cache_file_path)
            # The datetime is in the format YYYYMMDD_HHMMSS
            date_str = cache_filename.split("_")[0]
            time_str = cache_filename.split("_")[1]
            cache_datetime = datetime.strptime(
                f"{date_str}_{time_str}", "%Y%m%d_%H%M%S"
            )

            # --- 更新ロジック ---
            # 1. キャッシュが今日より古い場合
            if cache_datetime.date() < now.date():
                print("Cached data is from a previous day. Downloading new data.")
                download_required = True
            # 2. キャッシュは今日だが、取引終了時刻後に実行され、かつキャッシュが取引終了時刻より古い場合
            elif now > market_close_time and cache_datetime < market_close_time:
                print(
                    "Market has closed and cached data is from before the close. Downloading new data."
                )
                download_required = True
            # 3. それ以外はキャッシュを使用
            else:
                print("Using cached data.")
                df = df_cache
                download_required = False

        except (ValueError, IndexError) as e:
            print(
                f"Could not parse date from cache file name '{cache_file_path}'. Error: {e}. Will download new data."
            )
            download_required = True

    if df is None:  # キャッシュがない場合はダウンロードが必要
        download_required = True

    if download_required:
        print("Downloading new data...")
        try:
            # 予測に必要な期間を自動的に決定
            # モデルのシーケンス長、テストサイズ、テクニカル指標の最長期間を考慮
            test_size = args.test_size  # テストデータサイズ
            min_data_points_for_features = (
                26  # MACDのlong_periodなど、テクニカル指標に必要な最小データポイント数
            )

            # 処理後に必要な最小取引日数 = シーケンス長 + テストデータサイズ
            min_trading_days_after_processing = args.seq_length + test_size

            # dropna()で失われる可能性のある行数を考慮した、必要な最小取引日数
            # (min_data_points_for_features - 1) は、テクニカル指標計算でNaNになる初期行数
            min_raw_trading_days_needed = min_trading_days_after_processing + (
                min_data_points_for_features - 1
            )

            # 基準となる期間数（例: 日次データ20年分に相当する期間数）
            # 10年 * 250営業日/年 = 2500期間
            base_num_periods = 10 * 250
            if args.time_frame == "daily":
                base_num_periods = 15 * 250
            elif args.time_frame == "weekly":
                base_num_periods = 10 * 250
            elif args.time_frame == "monthly":
                base_num_periods = 6 * 250

            # time_frameに応じて必要なカレンダー日数を計算
            if args.time_frame == "daily":
                # 日次データの場合、base_num_periodsに安全マージンを追加
                required_historical_days = (
                    int(base_num_periods * 1.4) + 60
                )  # 1.4は営業日をカレンダー日に変換する係数
            elif args.time_frame == "weekly":
                # 週次データの場合、base_num_periods週分のカレンダー日数を取得
                required_historical_days = (
                    int(base_num_periods * 7 * 1.2) + 60
                )  # 1.2は週次データ取得時の安全マージン
            elif args.time_frame == "monthly":
                # 月次データの場合、base_num_periodsヶ月分のカレンダー日数を取得
                required_historical_days = (
                    int(base_num_periods * 30 * 1.1) + 60
                )  # 1.1は月次データ取得時の安全マージン
            else:
                # 未知のtime_frameの場合、デフォルトで20年分の日次データに相当する期間
                required_historical_days = 20 * 365  # Fallback to 20 calendar years

            # ただし、min_raw_trading_days_neededを満たす最低限の期間は保証する
            # (これは主に、base_num_periodsが非常に小さい場合に備える)
            min_required_by_logic = int(min_raw_trading_days_needed * 1.4) + 30
            if required_historical_days < min_required_by_logic:
                required_historical_days = min_required_by_logic

            end_date = datetime.now()
            start_date = end_date - timedelta(days=required_historical_days)

            print(
                f"Downloading stock data for {args.stock_code} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
            )
            # yfinanceを使用して株価データを取得
            yf_interval_map = {
                "daily": "1d",
                "weekly": "1wk",
                "monthly": "1mo",
            }
            yf_interval = yf_interval_map.get(
                args.time_frame, "1d"  # Default to 1d if not found
            )
            if args.debug:
                print(
                    f"DEBUG: yfinance interval set to: {yf_interval} for time_frame: {args.time_frame}"
                )

            df_downloaded = yf.download(
                ticker_code, start=start_date, end=end_date, interval=yf_interval
            )

            if df_downloaded.empty:
                print(
                    f"銘柄コード {args.stock_code} のデータをyfinanceから取得できませんでした。"
                )
                if df_cache is not None:
                    print("Falling back to cached data.")
                    df = df_cache
                else:
                    sys.exit(1)
            else:
                # yfinanceの列名を既存のコードに合わせる
                df_downloaded.reset_index(inplace=True)  # 'Date'インデックスを列に変換

                new_columns = []
                for col in df_downloaded.columns:
                    if isinstance(col, tuple):
                        metric_name = col[0].lower()
                        if metric_name == "date":
                            new_columns.append("record_date")
                        else:
                            new_columns.append(metric_name)
                    else:
                        if str(col).lower() == "date":
                            new_columns.append("record_date")
                        else:
                            new_columns.append(str(col).lower())
                df_downloaded.columns = new_columns

                # ダウンロードしたデータを保存
                save_data_to_csv(df_downloaded, ticker_code, time_frame=args.time_frame)
                df = df_downloaded

        except ImportError:
            print(
                "yfinanceがインストールされていません。`pip install yfinance` を実行してください。"
            )
            sys.exit(1)
        except Exception as e:
            print(f"yfinanceでのデータ取得中にエラーが発生しました: {e}")
            if df_cache is not None:
                print("Falling back to cached data.")
                df = df_cache
            else:
                sys.exit(1)

    # 銘柄名を取得 (dfがNoneでないことを確認)
    if df is not None:
        try:
            ticker_info = yf.Ticker(ticker_code).info
            stock_name = ticker_info.get("longName", args.stock_code)
            print(f"Successfully loaded data for: {stock_name}")
        except Exception as e:
            print(f"Could not retrieve stock name: {e}")
            stock_name = args.stock_code
    else:
        print("最終的に利用可能なデータがありません。プログラムを終了します。")
        sys.exit(1)

    # --- 1. 特徴量エンジニアリング ---
    df["record_date"] = pd.to_datetime(df["record_date"])

    external_data_columns = []
    external_data_log_return_columns = []
    if args.external_indices:
        # --- 外部データを取得・マージ ---
        # dfの日付範囲を取得
        if not df.empty:
            start_date_ext = df["record_date"].min() - timedelta(
                days=7
            )  # マージの余裕を持たせる
            end_date_ext = df["record_date"].max() + timedelta(days=7)
        else:  # dfが空の場合（初回ダウンロードなど）
            end_date_ext = datetime.now()
            start_date_ext = end_date_ext - timedelta(
                days=15 * 365
            )  # フォールバックとして15年分

        external_data = update_fred_data(
            start_date_ext, end_date_ext, indices_to_fetch=args.external_indices
        )
        if external_data:
            external_data_columns = list(external_data.keys())
            for name, rate_df in external_data.items():
                rate_df = rate_df[["record_date", "close"]].rename(
                    columns={"close": name}
                )
                df = pd.merge(df, rate_df, on="record_date", how="left")

            # 外部データの欠損値を前方フィルで補完
            df[external_data_columns] = df[external_data_columns].ffill()

            # 外部データを騰落率（対数リターン）に変換
            for col in external_data_columns:
                log_return_col_name = f"{col}_log_return"
                # 0や負の値を避けるために微小な値を加算
                df[log_return_col_name] = np.log(
                    df[col].replace(0, np.nan).ffill() + 1e-9
                ) - np.log(df[col].replace(0, np.nan).ffill().shift(1) + 1e-9)
                external_data_log_return_columns.append(log_return_col_name)
        else:
            print("外部データを取得できなかったため、特徴量なしで処理を続行します。")

    df.set_index("record_date", inplace=True)

    time_frame_options = {
        "daily": {"freq": "B", "offset": pd.offsets.BDay()},
        "weekly": {"freq": "W", "offset": pd.offsets.Week()},
        "monthly": {"freq": "ME", "offset": pd.offsets.MonthEnd()},
    }

    # yfinanceから直接weekly/monthlyデータを取得した場合はリサンプリング不要
    # キャッシュ使用時(download_required=False)、またはdaily指定時(args.time_frame == 'daily')はリサンプリングを実行
    if not download_required or args.time_frame == "daily":
        resample_freq = time_frame_options[args.time_frame]["freq"]
        if args.debug:
            print(
                f"DEBUG: Resampling frequency set to: {resample_freq} based on time_frame: {args.time_frame}"
            )

        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # 外部データカラムもリサンプリング対象に追加
        for col in external_data_columns:
            ohlc_dict[col] = "last"  # 各期間の最後の値を採用
        for col in external_data_log_return_columns:
            ohlc_dict[col] = "last"

        df = df.resample(resample_freq).agg(ohlc_dict)

    df.reset_index(inplace=True)
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
    if args.debug:
        print(f"DEBUG: len(df) after resampling and initial dropna: {len(df)}")
        print("DEBUG: df.head() after resampling:")
        print(df.head())
        print("DEBUG: df.tail() after resampling:")
        print(df.tail())

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
    ] + external_data_log_return_columns

    test_size = args.test_size
    if len(df) <= test_size + args.seq_length:
        print("データが少なく、学習と評価を分割できません。")
        sys.exit(1)

    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()

    # DEBUG: Print statistics of features before scaling
    print("\n--- Feature Statistics Before Scaling ---")
    print(train_df[feature_columns].describe())
    print("----------------------------------------\n")

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_columns])

    train_scaled = scaler.transform(train_df[feature_columns])
    test_scaled = scaler.transform(test_df[feature_columns])

    # --- モデル学習・評価・予測ループ ---
    model_type = args.model_type

    date_offset = time_frame_options[args.time_frame]["offset"]
    future_dates = pd.date_range(
        start=df["record_date"].max() + date_offset,
        periods=args.fut_pred,
        freq=date_offset,
    )

    evaluation_results = []

    try:
        print(f"\n--- Running prediction for model: {model_type} ---")

        if model_type == "gbm_garch":
            # --- 3. GARCH+GBMモデルの実行 ---
            (
                median_future_path,
                backtest_predictions,
                all_future_paths,
            ) = predict_with_gbm_garch(
                df,
                fut_pred=args.fut_pred,
                test_size=args.test_size,
                num_simulations=args.num_simulations,
            )

            # --- 5. 評価と予測 ---
            metrics = evaluate_gbm_garch(backtest_predictions, test_df)
            metrics["model"] = "gbm_garch"
            evaluation_results.append(metrics)

            predictions = median_future_path

            print(f"\n--- 予測詳細 (モデル: {model_type}) ---")
            details_df = pd.DataFrame({"Date": future_dates, "Prediction": predictions})
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
                    "dates": test_df["record_date"].iloc[: len(backtest_predictions)],
                    "predictions": backtest_predictions,
                },
                simulation_paths=all_future_paths,
            )

        else:
            # --- 3. シーケンス作成 ---
            log_return_idx = feature_columns.index("log_return")

            X_train, y_train = create_sequences(
                train_scaled, args.seq_length, log_return_idx
            )

            padding_for_test = train_scaled[-args.seq_length :]
            combined_for_test = np.concatenate([padding_for_test, test_scaled])
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
                drop_last=False,
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
                    "dates": test_df["record_date"].iloc[: len(backtest_predictions)],
                    "predictions": backtest_predictions,
                },
            )
        print(f"グラフをメモリ上に生成しました。")

        # グラフをファイルに保存する
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"{report_dir}/{date_str}_{args.stock_code}_{model_type}.png"
        with open(chart_filename, "wb") as f:
            f.write(chart_binary)
        print(f"グラフを {chart_filename} に保存しました。")

    except Exception as e:
        print(f"\n--- 処理中にエラーが発生しました。 ---")
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


if __name__ == "__main__":
    main()
