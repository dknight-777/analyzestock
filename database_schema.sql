-- Table: stocks
CREATE TABLE stocks (
    record_date VARCHAR(8) NOT NULL,
    code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(255),
    market_product_segment VARCHAR(50),
    sector_code_1 VARCHAR(10),
    sector_name_1 VARCHAR(50),
    sector_code_2 VARCHAR(10),
    sector_name_2 VARCHAR(50),
    scale_code VARCHAR(10),
    scale_name VARCHAR(50),
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (record_date, code)
);

COMMENT ON COLUMN stocks.record_date IS 'データの記録日。日次データなどを管理する際のキーとして使用。';
COMMENT ON COLUMN stocks.code IS '銘柄を識別するためのコード（例：株式コード）。';
COMMENT ON COLUMN stocks.stock_name IS '企業名または銘柄の名称。';
COMMENT ON COLUMN stocks.market_product_segment IS '上場している市場（例：東証プライム、東証グロースなど）や商品区分。';
COMMENT ON COLUMN stocks.stock_name IS '企業名または銘柄の名称。';
COMMENT ON COLUMN stocks.market_product_segment IS '上場している市場（例：東証プライム、東証グロースなど）や商品区分。';
COMMENT ON COLUMN stocks.sector_code_1 IS '主要な業種のコード。';
COMMENT ON COLUMN stocks.sector_name_1 IS '主要な業種の区分名。';
COMMENT ON COLUMN stocks.sector_code_2 IS '補助的な業種のコード（必要に応じて設定）。';
COMMENT ON COLUMN stocks.sector_name_2 IS '補助的な業種の区分名（必要に応じて設定）。';
COMMENT ON COLUMN stocks.scale_code IS '企業の規模を示すコード。';
COMMENT ON COLUMN stocks.scale_name IS '企業の規模区分名。';
COMMENT ON COLUMN stocks.updated_by IS 'このレコードを更新したプログラム名。';
COMMENT ON COLUMN stocks.updated_at IS 'このレコードの更新日時。';

-- Table: stock_values
CREATE TABLE stock_values (
    record_date DATE NOT NULL,
    code VARCHAR(10) NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    dividends NUMERIC,
    stock_splits NUMERIC,
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (record_date, code),
    CREATE INDEX idx_stock_values_code_recorddate ON stock_values (code, record_date),
    CREATE INDEX idx_stock_values_recorddate ON stock_values (record_date)
);

-- Table: usdjpy_values
CREATE TABLE usdjpy_values (
    record_date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (record_date)
);

COMMENT ON COLUMN usdjpy_values.record_date IS '為替レートデータの記録日。';
COMMENT ON COLUMN usdjpy_values.open IS '始値。';
COMMENT ON COLUMN usdjpy_values.high IS '高値。';
COMMENT ON COLUMN usdjpy_values.low IS '安値。';
COMMENT ON COLUMN usdjpy_values.close IS '終値。';
COMMENT ON COLUMN usdjpy_values.updated_by IS 'このレコードを更新したプログラム名。';
COMMENT ON COLUMN usdjpy_values.updated_at IS 'このレコードの更新日時。';

COMMENT ON COLUMN stock_values.record_date IS '株価データの記録日。';
COMMENT ON COLUMN stock_values.code IS '銘柄を識別するためのコード。';
COMMENT ON COLUMN stock_values.open IS '始値。';
COMMENT ON COLUMN stock_values.high IS '高値。';
COMMENT ON COLUMN stock_values.low IS '安値。';
COMMENT ON COLUMN stock_values.close IS '終値。';
COMMENT ON COLUMN stock_values.volume IS '出来高。';
COMMENT ON COLUMN stock_values.dividends IS '配当。';
COMMENT ON COLUMN stock_values.stock_splits IS '株式分割。';
COMMENT ON COLUMN stock_values.updated_by IS 'このレコードを更新したプログラム名。';
COMMENT ON COLUMN stock_values.updated_at IS 'このレコードの更新日時。';

-- Table: prediction_runs
CREATE TABLE prediction_runs (
    prediction_batch_id UUID PRIMARY KEY,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    stock_code VARCHAR(10) NOT NULL,
    model_type VARCHAR(50),
    time_frame VARCHAR(20),
    epochs INTEGER,
    seq_length INTEGER,
    fut_pred INTEGER,
    device VARCHAR(20),
    updated_by VARCHAR(255)
);

COMMENT ON TABLE prediction_runs IS '予測バッチの実行情報（引数）を格納するテーブル。';
COMMENT ON COLUMN prediction_runs.prediction_batch_id IS 'この実行バッチを一位に識別するID。';
COMMENT ON COLUMN prediction_runs.executed_at IS 'スクリプトが実行された日時。';
COMMENT ON COLUMN prediction_runs.stock_code IS '引数で指定された銘柄コード。';
COMMENT ON COLUMN prediction_runs.model_type IS '引数で指定されたモデル名。全モデル実行時はNULL。';
COMMENT ON COLUMN prediction_runs.time_frame IS '引数で指定された時間枠（daily/weekly）。';
COMMENT ON COLUMN prediction_runs.epochs IS '引数で指定されたエポック数。';
COMMENT ON COLUMN prediction_runs.seq_length IS '引数で指定されたシーケンス長。';
COMMENT ON COLUMN prediction_runs.fut_pred IS '引数で指定された予測期間。';
COMMENT ON COLUMN prediction_runs.device IS '引数で指定された実行デバイス（cpu/cuda）。';
COMMENT ON COLUMN prediction_runs.updated_by IS 'このレコードを登録したプログラム名。';

-- Table: prediction_charts
CREATE TABLE prediction_charts (
    chart_id SERIAL PRIMARY KEY,
    prediction_batch_id UUID NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    chart_image BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE,
    UNIQUE (prediction_batch_id, model_name),
    FOREIGN KEY (prediction_batch_id) REFERENCES prediction_runs (prediction_batch_id)
);

COMMENT ON TABLE prediction_charts IS '予測結果のグラフ画像を格納するテーブル。';
COMMENT ON COLUMN prediction_charts.chart_id IS 'グラフを一位に識別するための自動採番ID。';
COMMENT ON COLUMN prediction_charts.prediction_batch_id IS 'このグラフが属する実行バッチのID。';
COMMENT ON COLUMN prediction_charts.model_name IS 'このグラフを生成したモデルの名称。';
COMMENT ON COLUMN prediction_charts.chart_image IS '予測結果を可視化したグラフの画像データ（バイナリ形式）。';
COMMENT ON COLUMN prediction_charts.created_at IS 'このレコードがデータベースに作成された日時。';
COMMENT ON COLUMN prediction_charts.updated_by IS 'このレコードを最後に更新したプログラムやユーザーの名前。';
COMMENT ON COLUMN prediction_charts.updated_at IS 'このレコードが最後に更新された日時。';

-- Table: stock_predictions
CREATE TABLE stock_predictions (
    prediction_id SERIAL PRIMARY KEY,
    prediction_batch_id UUID NOT NULL,
    chart_id INTEGER,
    model_name VARCHAR(255) NOT NULL,
    prediction_target_date DATE NOT NULL,
    code VARCHAR(10) NOT NULL,
    predicted_value NUMERIC,
    predicted_volume BIGINT,
    actual_value NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (prediction_batch_id) REFERENCES prediction_runs (prediction_batch_id),
    FOREIGN KEY (chart_id) REFERENCES prediction_charts (chart_id)
);

COMMENT ON TABLE stock_predictions IS '個別の株価予測結果（数値）を格納するテーブル。';
COMMENT ON COLUMN stock_predictions.prediction_id IS '予測結果を一位に識別するための自動採番ID。';
COMMENT ON COLUMN stock_predictions.prediction_batch_id IS '一回の実行で生成された複数の予測をグループ化するためのID。';
COMMENT ON COLUMN stock_predictions.chart_id IS '関連するグラフ画像のID。prediction_chartsテーブルと紐づく。';
COMMENT ON COLUMN stock_predictions.model_name IS '予測に使用した機械学習モデルの名称（例: "lstm", "gru"）。';
COMMENT ON COLUMN stock_predictions.prediction_target_date IS 'この予測が対象とする未来の日付。';
COMMENT ON COLUMN stock_predictions.code IS '予測対象の銘柄コード。';
COMMENT ON COLUMN stock_predictions.predicted_value IS 'モデルによって算出された予測株価。';
COMMENT ON COLUMN stock_predictions.predicted_volume IS 'モデルによって算出された予測出来高。';
COMMENT ON COLUMN stock_predictions.actual_value IS '予測対象日の実際の株価。後から実績を記録するために使用。';
COMMENT ON COLUMN stock_predictions.created_at IS 'このレコードがデータベースに作成された日時。';
COMMENT ON COLUMN stock_predictions.updated_by IS 'このレコードを最後に更新したプログラムやユーザーの名前。';
COMMENT ON COLUMN stock_predictions.updated_at IS 'このレコードが最後に更新された日時。';

-- Table: prediction_run_evaluations
CREATE TABLE prediction_run_evaluations (
    evaluation_id SERIAL PRIMARY KEY,
    prediction_batch_id UUID NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    rmse NUMERIC,
    mae NUMERIC,
    r2_score NUMERIC,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (prediction_batch_id) REFERENCES prediction_runs (prediction_batch_id)
);

COMMENT ON TABLE prediction_run_evaluations IS 'モデルの予測精度に関する評価指標を格納するテーブル。';
COMMENT ON COLUMN prediction_run_evaluations.evaluation_id IS '評価結果を一位に識別するための自動採番ID。';
COMMENT ON COLUMN prediction_run_evaluations.prediction_batch_id IS '評価対象の予測が属する実行バッチのID。';
COMMENT ON COLUMN prediction_run_evaluations.model_name IS '評価対象のモデル名。';
COMMENT ON COLUMN prediction_run_evaluations.rmse IS '二乗平均平方根誤差 (Root Mean Squared Error)。予測精度を示す指標の一つ。';
COMMENT ON COLUMN prediction_run_evaluations.mae IS '平均絶対誤差 (Mean Absolute Error)。予測精度を示す指標の一つ。';
COMMENT ON COLUMN prediction_run_evaluations.r2_score IS '決定係数 (R2 Score)。予測が実績値の変動をどれだけ説明できるかを示す指標。';
COMMENT ON COLUMN prediction_run_evaluations.created_at IS 'このレコードがデータベースに作成された日時。';
COMMENT ON COLUMN prediction_run_evaluations.updated_by IS 'このレコードを最後に更新したプログラムやユーザーの名前。';
COMMENT ON COLUMN prediction_run_evaluations.updated_at IS 'このレコードが最後に更新された日時。';
