import io
from datetime import datetime
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prediction_chart(
    df: pd.DataFrame,
    predictions: np.ndarray,
    future_dates: pd.DatetimeIndex,
    stock_code: str,
    stock_name: str,
    model_type: str,
    time_frame: str,
    backtest_data: dict | None = None,
) -> bytes:
    """結果をプロットし、チャート画像のバイナリデータを返します。"""
    try:
        font_path = "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf"
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    except FileNotFoundError:
        print(f"警告: 日本語フォント '{font_path}' が見つかりません。")

    plt.figure(figsize=(12, 7))

    # 表示する過去のデータ範囲を決定
    plot_range_days = 30
    if backtest_data:
        plot_range_days = max(plot_range_days, len(backtest_data["dates"]) + 5)

    last_date = df["record_date"].max()
    start_date = last_date - pd.DateOffset(days=plot_range_days)
    plot_df = df[df["record_date"] >= start_date].copy()

    # Combine all dates for x-axis labeling
    all_dates = pd.concat([plot_df["record_date"], pd.Series(future_dates)]).sort_values().reset_index(drop=True)

    # Plot actual stock price using numerical index
    plt.plot(range(len(plot_df)), plot_df["close"], label="実際の株価", color="blue")

    # Plot future predictions using numerical index, offset by length of actual data
    # Prepend the last actual close value to predictions to create a continuous line
    # The x-coordinate for this point will be len(plot_df) - 1
    extended_predictions = np.insert(predictions, 0, plot_df["close"].iloc[-1])

    # The x-range for the extended predictions starts from the last actual data point's index
    start_index_extended_predictions = len(plot_df) - 1
    
    plt.plot(range(start_index_extended_predictions, start_index_extended_predictions + len(extended_predictions)),
             extended_predictions,
             label=f"将来の予測 ({model_type.upper()})",
             linestyle="--", # Keep the same linestyle as future predictions
             color="orange")

    # 予測接続線は将来予測に含められたため削除

    # Set x-axis ticks and labels to show actual dates
    tick_indices = np.linspace(0, len(all_dates) - 1, 10, dtype=int)
    tick_labels = [all_dates.iloc[i].strftime("%Y-%m-%d") for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=45, ha="right")

    # 最高値・最安値の注釈
    # Create a list of DataFrames, each with 'numerical_index', 'record_date', 'close'
    annotation_dfs = []

    # Actual data
    actual_annotation_df = plot_df[["record_date", "close"]].copy()
    actual_annotation_df["numerical_index"] = range(len(plot_df))
    annotation_dfs.append((actual_annotation_df, "blue"))

    # Predicted data
    if predictions.size > 0:
        predicted_annotation_df = pd.DataFrame({
            "record_date": future_dates,
            "close": predictions.flatten()
        })
        predicted_annotation_df["numerical_index"] = range(len(plot_df), len(plot_df) + len(predictions))
        # Ensure 'close' column contains scalar Python floats
        predicted_annotation_df["close"] = predicted_annotation_df["close"].astype(float)
        annotation_dfs.append((predicted_annotation_df, "orange"))

    for data_frame, color in annotation_dfs:
        if not data_frame.empty:
            for index, row in data_frame.iterrows():
                plt.text(
                    row["numerical_index"], # Use numerical index for x-coordinate
                    row["close"],
                    f" {row['close']:.2f}",
                    va="center", # Center vertically
                    ha="center", # Center horizontally
                    color=color,
                    fontsize=7, # Smaller font size to reduce clutter
                )

    time_frame_japanese = "日足" if time_frame == "daily" else "週足"
    plt.title(
        f"{stock_name} ({stock_code}) の株価予測 ({model_type.upper()}・{time_frame_japanese})",
        fontsize=16,
    )
    plt.xlabel("日付", fontsize=12)
    plt.ylabel("終値", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    # 旧X軸フォーマット設定 (ユーザーの要望により削除)

    # ファイルに保存する代わりに、バイナリデータを返す
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()  # メモリリークを防ぐために図を閉じる
    buf.seek(0)
    return buf.getvalue()
