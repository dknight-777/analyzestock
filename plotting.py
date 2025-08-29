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
):
    """結果をプロットし、チャートをファイルに保存します。"""
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

    plt.plot(plot_df["record_date"], plot_df["close"], label="実際の株価", color="blue")
    plt.plot(
        future_dates,
        predictions,
        label=f"将来の予測 ({model_type.upper()})",
        linestyle="--",
        color="orange",
    )

    # バックテストのプロット
    if backtest_data:
        plt.plot(
            backtest_data["dates"],
            backtest_data["predictions"],
            label="バックテスト予測",
            linestyle=":",
            color="green",
        )

    # 最高値・最安値の注釈
    plot_data_frames = [(plot_df, "blue")]
    if predictions.size > 0:
        plot_data_frames.append(
            (
                pd.DataFrame(
                    {"record_date": future_dates, "close": predictions.flatten()}
                ),
                "orange",
            )
        )
    if backtest_data and backtest_data["predictions"].size > 0:
        plot_data_frames.append(
            (
                pd.DataFrame(
                    {
                        "record_date": backtest_data["dates"],
                        "close": backtest_data["predictions"].flatten(),
                    }
                ),
                "green",
            )
        )

    for data, color in plot_data_frames:
        if not data.empty:
            for func, va in [
                (data["close"].idxmax, "bottom"),
                (data["close"].idxmin, "top"),
            ]:
                idx = func()
                row = data.loc[idx]
                plt.text(
                    row["record_date"],
                    row["close"],
                    f" {row['close']:.2f}",
                    va=va,
                    ha="left",
                    color=color,
                    fontsize=9,
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
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=7))
    plt.gcf().autofmt_xdate()

    output_dir = Path("stock_prediction_charts")
    output_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = (
        output_dir / f"{stock_code}_prediction_{model_type}_{time_frame}_{now}.png"
    )
    plt.savefig(chart_filename, dpi=150)
    print(f"グラフを {chart_filename} として保存しました。")
