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