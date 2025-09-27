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
    simulation_paths: np.ndarray | None = None, # 引数は残すが、使用しない
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
    if time_frame == 'weekly':
        start_date_offset = pd.DateOffset(months=5)
    elif time_frame == 'monthly':
        start_date_offset = pd.DateOffset(years=2)
    else: # daily
        start_date_offset = pd.DateOffset(months=1)

    last_date = df["record_date"].max()
    start_date = last_date - start_date_offset
    plot_df = df[df["record_date"] >= start_date].copy()

    # Combine all dates for x-axis labeling
    all_dates = pd.concat([plot_df["record_date"], pd.Series(future_dates)]).sort_values().reset_index(drop=True)

    # Plot actual stock price using numerical index
    plt.plot(range(len(plot_df)), plot_df["close"], label="実際の株価", color="blue")

    # Plot historical Bollinger Bands
    if all(col in plot_df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        plt.plot(range(len(plot_df)), plot_df['bb_middle'], linestyle='--', color='gray', alpha=0.7, label='ボリンジャーバンド (中央)')
        plt.fill_between(range(len(plot_df)), plot_df['bb_lower'], plot_df['bb_upper'], color='gray', alpha=0.2, label='ボリンジャーバンド (±2σ)')

    # Plot future prediction line
    start_index_extended_predictions = len(plot_df) - 1
    extended_predictions = np.insert(predictions, 0, plot_df["close"].iloc[-1])
    x_range_future = range(start_index_extended_predictions, start_index_extended_predictions + len(extended_predictions))
    
    plt.plot(x_range_future,
             extended_predictions,
             label=f"将来の予測 ({model_type.upper()})",
             linestyle="--",
             color="orange")

    # Set x-axis ticks and labels to show actual dates
    tick_indices = np.arange(len(all_dates))
    tick_labels = [d.strftime("%Y-%m-%d") for d in all_dates]
    plt.xticks(tick_indices, tick_labels, rotation=90, ha="center", color="black", fontsize=8)
    plt.yticks(color="black")

    # 最高値・最安値の注釈
    annotation_dfs = []
    actual_annotation_df = plot_df[["record_date", "close"]].copy()
    actual_annotation_df["numerical_index"] = range(len(plot_df))
    annotation_dfs.append((actual_annotation_df, "black"))

    if predictions.size > 0:
        predicted_annotation_df = pd.DataFrame({
            "record_date": future_dates,
            "close": predictions.flatten()
        })
        predicted_annotation_df["numerical_index"] = range(len(plot_df), len(plot_df) + len(predictions))
        predicted_annotation_df["close"] = predicted_annotation_df["close"].astype(float)
        annotation_dfs.append((predicted_annotation_df, "black"))

    for data_frame, color in annotation_dfs:
        if not data_frame.empty:
            for index, row in data_frame.iterrows():
                plt.text(
                    row["numerical_index"],
                    row["close"],
                    f" {row['close']:.2f}",
                    va="center",
                    ha="center",
                    color=color,
                    fontsize=7,
                )

    time_frame_japanese = "日足" if time_frame == "daily" else "週足"
    plt.title(
        f"{stock_name} ({stock_code}) の株価予測 ({model_type.upper()}・{time_frame_japanese})",
        fontsize=16,
        color="black",
    )
    plt.xlabel("日付", fontsize=12, color="black")
    plt.ylabel("終値", fontsize=12, color="black")
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_color("black")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return buf.getvalue()
