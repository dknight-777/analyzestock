import os
import sys
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
import argparse

import pandas as pd
from sqlalchemy import create_engine, text, Engine, Connection

# データベース接続情報
# --- Database Connection Details ---
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")


def get_db_engine() -> Optional[Engine]:
    """データベース接続エンジンを作成して返します。接続に失敗した場合はNoneを返します。"""
    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        # 接続テスト
        with engine.connect():
            pass
        return engine
    except Exception as e:
        print(f"データベース接続に失敗しました: {e}")
        return None


def save_prediction_run(
    connection: Connection, batch_id: UUID, args: argparse.Namespace, updated_by: str
):
    """実行時の引数をprediction_runsテーブルに保存します。"""
    query = text(
        """
        INSERT INTO prediction_runs (prediction_batch_id, stock_code, model_type, time_frame, epochs, seq_length, fut_pred, device, updated_by)
        VALUES (:prediction_batch_id, :stock_code, :model_type, :time_frame, :epochs, :seq_length, :fut_pred, :device, :updated_by)
        """
    )
    connection.execute(
        query,
        {
            "prediction_batch_id": batch_id,
            "stock_code": args.stock_code,
            "model_type": args.model_type,
            "time_frame": args.time_frame,
            "epochs": args.epochs,
            "seq_length": args.seq_length,
            "fut_pred": args.fut_pred,
            "device": args.device,
            "updated_by": updated_by,
        },
    )
    print(f"実行情報 (Batch ID: {batch_id}) を prediction_runs テーブルに保存しました。")


def get_stock_data(engine: Engine, code: str) -> pd.DataFrame:
    """指定された銘柄コードの株価データをデータベースから取得します。"""
    try:
        query = text(
            """
            SELECT
                sv.record_date,
                sv.close,
                s.stock_name
            FROM stock_values sv
            LEFT JOIN (
                SELECT code, stock_name, ROW_NUMBER() OVER(PARTITION BY code ORDER BY record_date DESC) as rn
                FROM stocks
            ) s ON sv.code = s.code AND s.rn = 1
            WHERE sv.code = :code
            ORDER BY sv.record_date
        """
        )
        df = pd.read_sql_query(query, engine, params={"code": code})
        return df
    except Exception as e:
        print(f"データベースからのデータ取得中にエラーが発生しました: {e}")
        sys.exit(1)


def save_prediction_chart(
    connection: Connection, batch_id: UUID, model_name: str, chart_image: bytes, updated_by: str
) -> Optional[int]:
    """予測グラフを保存し、そのIDを返します。"""
    query = text(
        """
        INSERT INTO prediction_charts (prediction_batch_id, model_name, chart_image, updated_by, updated_at)
        VALUES (:batch_id, :model_name, :chart_image, :updated_by, :updated_at)
        RETURNING chart_id
        """
    )
    result = connection.execute(
        query,
        {
            "batch_id": batch_id,
            "model_name": model_name,
            "chart_image": chart_image,
            "updated_by": updated_by,
            "updated_at": datetime.now(),
        },
    )
    chart_id = result.scalar_one_or_none()
    print(f"グラフを prediction_charts テーブルに保存しました (chart_id: {chart_id})。")
    return chart_id


def save_stock_predictions(
    connection: Connection,
    batch_id: UUID,
    chart_id: int,
    model_name: str,
    code: str,
    predictions_df: pd.DataFrame,
    updated_by: str,
):
    """株価の予測結果をまとめて保存します。"""
    records_to_insert = []
    for index, row in predictions_df.iterrows():
        records_to_insert.append(
            {
                "prediction_batch_id": batch_id,
                "chart_id": chart_id,
                "model_name": model_name,
                "prediction_target_date": row["date"],
                "code": code,
                "predicted_value": float(row["prediction"]) if row.get("prediction") is not None else None,
                "updated_by": updated_by,
                "updated_at": datetime.now(),
            }
        )
    
    if records_to_insert:
        query = text(
            """
            INSERT INTO stock_predictions (prediction_batch_id, chart_id, model_name, prediction_target_date, code, predicted_value, updated_by, updated_at)
            VALUES (:prediction_batch_id, :chart_id, :model_name, :prediction_target_date, :code, :predicted_value, :updated_by, :updated_at)
            """
        )
        connection.execute(query, records_to_insert)
        print(f"{len(records_to_insert)}件の予測結果を stock_predictions テーブルに保存しました。")


def save_run_evaluations(
    connection: Connection, batch_id: UUID, evaluation_results: List[Dict[str, Any]], updated_by: str
):
    """モデルの評価結果をまとめて保存します。"""
    records_to_insert = []
    for result in evaluation_results:
        records_to_insert.append(
            {
                "prediction_batch_id": batch_id,
                "model_name": result["model"],
                "rmse": float(result["RMSE"]) if result.get("RMSE") is not None else None,
                "mae": float(result["MAE"]) if result.get("MAE") is not None else None,
                "r2_score": float(result["R2 Score"]) if result.get("R2 Score") is not None else None,
                "updated_by": updated_by,
                "updated_at": datetime.now(),
            }
        )

    if records_to_insert:
        query = text(
            """
            INSERT INTO prediction_run_evaluations (prediction_batch_id, model_name, rmse, mae, r2_score, updated_by, updated_at)
            VALUES (:prediction_batch_id, :model_name, :rmse, :mae, :r2_score, :updated_by, :updated_at)
            """
        )
        connection.execute(query, records_to_insert)
        print(f"{len(records_to_insert)}件の評価結果を prediction_run_evaluations テーブルに保存しました。")