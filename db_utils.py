import os
import sys
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text, Engine

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


def get_stock_data(engine: Engine, code: str) -> pd.DataFrame:
    """指定された銘柄コードの株価データをデータベースから取得します。"""
    try:
        # SQLインジェクション対策としてパラメータ化クエリを使用
        # 同時にstocksテーブルから最新の銘柄名を取得
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
