import os
import argparse
import sys
from pathlib import Path
from uuid import UUID
from io import BytesIO

import pandas as pd
from sqlalchemy import text, Engine, Connection

# Import from existing modules
from db_utils import get_db_engine

# PDF出力用のライブラリ
REPORTLAB_AVAILABLE = False
_reportlab_import_error = None
try:
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        Table,
        TableStyle,
        PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    REPORTLAB_AVAILABLE = True
except ImportError as e:
    # エラーを捕捉するが、表示はPDF生成が要求された場合のみ行う
    _reportlab_import_error = e


def register_japanese_font():
    """日本語フォントを登録します。"""
    font_path = "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("TakaoPGothic", font_path))
        return "TakaoPGothic"
    else:
        print(
            f"警告: 日本語フォント '{font_path}' が見つかりません。PDFの日本語が文字化けする可能性があります。"
        )
        return "Helvetica"  # デフォルトフォント


def get_latest_prediction_batch_id(engine: Engine) -> UUID | None:
    """データベースから最新の prediction_batch_id を取得します。"""
    try:
        with engine.connect() as connection:
            query = text(
                "SELECT prediction_batch_id FROM prediction_runs ORDER BY executed_at DESC LIMIT 1"
            )
            result = connection.execute(query).scalar_one_or_none()
            if isinstance(result, UUID):
                return result
            return UUID(result) if result else None
    except Exception as e:
        print(f"最新のバッチIDの取得中にエラーが発生しました: {e}")
        return None

def get_stock_name(connection: Connection, stock_code: str) -> str | None:
    """指定された銘柄コードの最新の銘柄名を取得します。"""
    query = text(
        "SELECT stock_name FROM stocks WHERE code = :stock_code ORDER BY record_date DESC LIMIT 1"
    )
    result = connection.execute(query, {"stock_code": stock_code}).scalar_one_or_none()
    return result


def get_prediction_run_info(connection: Connection, batch_id: UUID) -> pd.Series | None:
    """指定されたバッチIDの実行情報を取得します。"""
    query = text("SELECT * FROM prediction_runs WHERE prediction_batch_id = :batch_id")
    df = pd.read_sql_query(query, connection, params={"batch_id": str(batch_id)})
    return df.iloc[0] if not df.empty else None


def get_evaluation_metrics(connection: Connection, batch_id: UUID) -> pd.DataFrame:
    """指定されたバッチIDの評価指標を取得します。"""
    query = text(
        "SELECT model_name, rmse, mae, r2_score FROM prediction_run_evaluations WHERE prediction_batch_id = :batch_id ORDER BY model_name"
    )
    return pd.read_sql_query(query, connection, params={"batch_id": str(batch_id)})


def get_predictions(connection: Connection, batch_id: UUID) -> pd.DataFrame:
    """指定されたバッチIDの予測結果を取得します。"""
    query = text(
        """SELECT model_name, prediction_target_date, predicted_value, predicted_volume, actual_value FROM stock_predictions WHERE prediction_batch_id = :batch_id ORDER BY model_name, prediction_target_date"""
    )
    return pd.read_sql_query(query, connection, params={"batch_id": str(batch_id)})


def get_charts(connection: Connection, batch_id: UUID) -> list[tuple[str, bytes]]:
    """指定されたバッチIDのグラフを取得します。"""
    query = text(
        "SELECT model_name, chart_image FROM prediction_charts WHERE prediction_batch_id = :batch_id ORDER BY model_name"
    )
    result = connection.execute(query, {"batch_id": str(batch_id)})
    return result.fetchall()


def generate_text_report(
    run_info: pd.Series,
    eval_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    charts: list,
    output_dir: str,
    batch_id: UUID,
    stock_name: str | None = None,
):
    """テキスト形式のレポートを生成し、グラフを保存します。"""
    print("\n" + "=" * 80)
    print(" " * 30 + "株価予測レポート")
    print("=" * 80)
    print(f"■ 実行バッチID: {batch_id}")
    print(f"■ 実行日時: {run_info['executed_at']}")
    stock_code_display = run_info['stock_code']
    if stock_name:
        stock_code_display += f" ({stock_name})"
    print(f"■ 銘柄コード: {stock_code_display}")
    print(f"■ タイムフレーム: {run_info['time_frame']}")
    print(f"■ 予測期間: {run_info['fut_pred']} 日分")
    print(f"■ 使用モデル: {run_info['model_type'] or '全モデル'}")
    print("-" * 80)

    if not eval_metrics.empty:
        print("\n■ モデル評価指標:")
        print(eval_metrics.to_string(index=False))
        print("\n--- 指標の説明 ---")
        print("RMSE (二乗平均平方根誤差): 予測と実際の値の差の二乗の平均の平方根。小さいほど良い。")
        print("MAE (平均絶対誤差): 予測と実際の値の差の絶対値の平均。小さいほど良い。")
        print("R2 Score (決定係数): 実際の値の変動を予測モデルがどれだけ説明できるかを示す指標。1に近いほど良い。")

        best_model = eval_metrics.loc[eval_metrics['r2_score'].idxmax()]
        print("\n--- 最も精度の良いモデル ---")
        print(f"R2 Scoreが最も高いモデルは '{best_model['model_name']}' (R2: {best_model['r2_score']:.4f}) です。")
    else:
        print("\n■ モデル評価指標: データがありません。")

    if not predictions.empty:
        print("\n■ 予測結果詳細:")
        
        # Create a multi-index pivot table
        pivot_df = predictions.pivot_table(
            index='prediction_target_date',
            columns='model_name',
            values=['predicted_value', 'predicted_volume']
        )
        
        # Format the output
        if 'predicted_value' in pivot_df.columns:
            pivot_df['predicted_value'] = pivot_df['predicted_value'].round(2)
        if 'predicted_volume' in pivot_df.columns:
            pivot_df['predicted_volume'] = pivot_df['predicted_volume'].fillna(0).astype(int)

        pivot_df.index = pd.to_datetime(pivot_df.index).strftime('%Y-%m-%d')
        print(pivot_df.to_string())
    else:
        print("\n■ 予測結果詳細: データがありません。")

    if charts:
        execution_date = pd.to_datetime(run_info['executed_at']).strftime('%Y-%m-%d')
        output_path = Path(output_dir) / execution_date
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\n■ グラフの保存:")
        for model_name, chart_image in charts:
            chart_file = output_path / f"prediction_chart_{model_name}_{batch_id}.png"
            with open(chart_file, "wb") as f:
                f.write(chart_image)
            print(f"  - {chart_file} に保存しました。")
    else:
        print("\n■ グラフ: データがありません。")

    print("\n" + "=" * 80)
    print("レポート生成が完了しました。")


def generate_pdf_report(
    run_info: pd.Series,
    eval_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    charts: list,
    output_dir: str,
    batch_id: UUID,
    stock_name: str | None = None,
):
    """PDF形式のレポートを生成します。"""
    execution_date = pd.to_datetime(run_info['executed_at']).strftime('%Y-%m-%d')
    output_path = Path(output_dir) / execution_date
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_file = output_path / f"report_{batch_id}.pdf"

    doc = SimpleDocTemplate(str(pdf_file), pagesize=A4)
    styles = getSampleStyleSheet()

    # 日本語フォントの設定
    font_name = register_japanese_font()
    styles.add(
        ParagraphStyle(name="Japan", fontName=font_name, fontSize=10, leading=14)
    )
    styles.add(
        ParagraphStyle(
            name="JapanTitle",
            fontName=font_name,
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="JapanH2",
            fontName=font_name,
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        )
    )

    story = []

    # 1. タイトル
    story.append(Paragraph("株価予測レポート", styles["JapanTitle"]))

    # 2. 実行情報
    story.append(Paragraph("実行情報", styles["JapanH2"]))
    stock_code_display = run_info['stock_code']
    if stock_name:
        stock_code_display += f" ({stock_name})"
    info_data = [
        ["実行バッチID:", str(batch_id)],
        ["実行日時:", str(run_info["executed_at"])],
        ["銘柄コード:", stock_code_display],
        ["タイムフレーム:", run_info["time_frame"]],
        ["予測期間:", f"{run_info['fut_pred']} 日分"],
        ["使用モデル:", run_info["model_type"] or "全モデル"],
    ]
    info_table = Table(info_data, colWidths=[100, 350])
    info_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), font_name, 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 12))

    # 3. モデル評価指標
    if not eval_metrics.empty:
        story.append(Paragraph("モデル評価指標", styles["JapanH2"]))
        eval_data = [eval_metrics.columns.tolist()] + eval_metrics.round(
            4
        ).values.tolist()
        eval_table = Table(eval_data, hAlign="LEFT")
        eval_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), font_name, 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ]
            )
        )
        story.append(eval_table)
        story.append(Spacer(1, 12))

        # 指標の説明
        story.append(Paragraph("指標の説明", styles["JapanH2"]))
        explanation_texts = [
            "<b>RMSE (二乗平均平方根誤差):</b> 予測と実際の値の差の二乗の平均の平方根。小さいほど良い。",
            "<b>MAE (平均絶対誤差):</b> 予測と実際の値の差の絶対値の平均。小さいほど良い。",
            "<b>R2 Score (決定係数):</b> 実際の値の変動を予測モデルがどれだけ説明できるかを示す指標。1に近いほど良い。"
        ]
        for text in explanation_texts:
            story.append(Paragraph(text, styles["Japan"]))
        
        story.append(Spacer(1, 12))

        # 最も精度の良いモデル
        best_model = eval_metrics.loc[eval_metrics['r2_score'].idxmax()]
        best_model_text = f"<b>最も精度の良いモデル:</b> R2 Scoreが最も高いモデルは <b>{best_model['model_name']}</b> (R2: {best_model['r2_score']:.4f}) です。"
        story.append(Paragraph(best_model_text, styles["Japan"]))
        story.append(Spacer(1, 12))

    # 4. 予測結果詳細 (Pivot Table)
    if not predictions.empty:
        story.append(Paragraph("予測結果詳細", styles["JapanH2"]))
        
        # --- 予測価格と出来高の統合テーブル ---
        pivot_df = predictions.pivot_table(
            index='prediction_target_date',
            columns='model_name',
            values=['predicted_value', 'predicted_volume']
        )

        # カラムの順序を調整 (value -> volume)
        pivot_df = pivot_df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
        
        # フォーマット
        for model in pivot_df.columns.levels[0]:
            pivot_df[(model, 'predicted_value')] = pivot_df[(model, 'predicted_value')].round(2)
            pivot_df[(model, 'predicted_volume')] = pivot_df[(model, 'predicted_volume')].fillna(0).astype(int)

        # ReportLabのテーブルデータを作成
        pivot_df.index = pd.to_datetime(pivot_df.index).strftime('%Y-%m-%d')
        header1 = ["Date"] + [col[0] for col in pivot_df.columns]
        header2 = [""] + [col[1].replace('predicted_', '').capitalize() for col in pivot_df.columns]
        
        # ユニークなモデル名を取得し、ヘッダー1をマージ
        unique_models = sorted(list(set(h for h in header1 if h != "Date")))
        spans = []
        for model in unique_models:
            start = header1.index(model)
            end = len(header1) - 1 - header1[::-1].index(model)
            if start != end:
                spans.append(('SPAN', (start, 0), (end, 0)))

        data = [header1, header2] + [
            [idx] + list(row) for idx, row in zip(pivot_df.index, pivot_df.values)
        ]
        
        pred_table = Table(data, hAlign="LEFT")
        pred_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), font_name, 10),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 1), colors.lightgrey),
                    ("BACKGROUND", (0, 2), (0, -1), colors.lightgrey),
                    ("ALIGN", (0, 0), (-1, 1), "CENTER"),
                    ("ALIGN", (1, 2), (-1, -1), "RIGHT"),
                ] + spans
            )
        )
        story.append(pred_table)
        story.append(Spacer(1, 12))

    # 5. グラフ
    if charts:
        story.append(PageBreak())
        story.append(Paragraph("予測グラフ", styles["JapanH2"]))
        for model_name, chart_image in charts:
            img = Image(BytesIO(chart_image), width=450, height=300)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDFレポートを {pdf_file} に保存しました。")




def main():
    """レポート生成のメイン処理"""
    parser = argparse.ArgumentParser(
        description="データベースから予測結果のレポートを生成します。"
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        help="レポート対象の prediction_batch_id。指定しない場合は最新のものが使われます。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="レポートとグラフの出力先ディレクトリ。",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "pdf"],
        default="pdf",
        help="出力形式 (text または pdf)。",
    )
    args = parser.parse_args()

    if args.format == "pdf" and not REPORTLAB_AVAILABLE:
        print("エラー: PDF出力には reportlab ライブラリが必要です。", file=sys.stderr)
        print(
            "以下のコマンドを実行して、必要なライブラリをインストールしてください:",
            file=sys.stderr,
        )
        print("pip install -r requirements.txt", file=sys.stderr)
        if _reportlab_import_error:
            print(
                "\n[デバッグ情報] インポート時に以下のエラーが発生しました:",
                file=sys.stderr,
            )
            print(f"  {_reportlab_import_error}", file=sys.stderr)
        sys.exit(1)

    engine = get_db_engine()
    if not engine:
        sys.exit(1)

    batch_id = None
    if args.batch_id:
        try:
            batch_id = UUID(args.batch_id)
        except ValueError:
            print(f"エラー: 無効なUUID形式のbatch_idです: {args.batch_id}")
            sys.exit(1)
    else:
        print("batch_idが指定されていないため、最新の予測実行を探します...")
        batch_id = get_latest_prediction_batch_id(engine)
        if not batch_id:
            print("エラー: レポート対象の予測実行が見つかりませんでした。")
            sys.exit(1)

    print(f"レポートを生成中 (Batch ID: {batch_id})...")

    with engine.connect() as connection:
        run_info = get_prediction_run_info(connection, batch_id)
        if run_info is None:
            print(
                f"エラー: 指定されたbatch_id ({batch_id}) の実行情報が見つかりません。"
            )
            sys.exit(1)

        stock_name = get_stock_name(connection, run_info['stock_code'])

        eval_metrics = get_evaluation_metrics(connection, batch_id)
        predictions = get_predictions(connection, batch_id)
        charts = get_charts(connection, batch_id)

    if args.format == "pdf":
        generate_pdf_report(
            run_info, eval_metrics, predictions, charts, args.output_dir, batch_id, stock_name
        )
    else:
        generate_text_report(
            run_info, eval_metrics, predictions, charts, args.output_dir, batch_id, stock_name
        )


if __name__ == "__main__":
    main()
