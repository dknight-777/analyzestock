# 株価予測プロジェクト

## 概要

このアプリケーションは、過去の株価データを分析し、機械学習モデル（LSTM, GRU, NN）を用いて将来の株価を予測し、結果をチャートで可視化します。データはPostgreSQLデータベースに保存されます。

## 特徴

- 複数の機械学習モデル（LSTM, GRU, NN）による株価予測
- 過去のデータと将来の予測値を結合したチャートを生成
- 証券コード、モデル、時間枠（日足/週足/月足）などをコマンドライン引数で指定可能

---

## 動作要件

- Python 3.10 以降
- PostgreSQL
- 日本語フォント（Takao-PGothic）

---

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/mame-777/analyzestock.git
cd analyzestock
```

### 2. システム依存関係のインストール (Linux)

`pip` でPythonライブラリをインストールする際、コンパイルに必要なシステムパッケージをあらかじめインストールしておきます。

```bash
# Debian / Ubuntu の場合
sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev
```

### 3. Python依存関係のインストール

プロジェクトに必要なPythonライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### 4. 日本語フォントのインストール (Linux)

グラフの日本語表示のために、Takao-PGothicフォントをインストールします。

```bash
# Debian / Ubuntu の場合
sudo apt-get update && sudo apt-get install -y fonts-takao-pgothic
```

もしフォントのインストール後に文字化けが解消されない場合は、Matplotlibのキャッシュを削除してみてください。

```bash
rm -rf ~/.cache/matplotlib
```

### 5. データベースのセットアップ

1.  **PostgreSQLのインストールと起動**
    お使いの環境にPostgreSQLをインストールし、データベースサーバーを起動してください。

2.  **データベースとテーブルの作成**
    `database_schema.sql` を使って、予測データの保存に必要なデータベースとテーブルを作成します。

    ```bash
    # psqlコマンドでデータベース 'stock_db' を作成
    psql -U postgres -c "CREATE DATABASE stock_db;"

    # 作成したデータベースに接続し、テーブルを作成
    psql -U postgres -d stock_db -f database_schema.sql
    ```
    *上記は `postgres` ユーザーの例です。ご自身の環境に合わせてユーザー名を変更してください。*

3.  **株価データの準備**
    予測の元となる過去の株価データを `stock_values` テーブルに、銘柄情報を `stocks` テーブルに格納してください。

### 6. 環境変数の設定

プロジェクトのルートディレクトリに `.env` ファイルを作成し、ローカルのデータベース接続情報を記述します。

`.env` ファイルの例：
```env
# PostgreSQL 接続情報
POSTGRES_DB=stock_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password # ご自身のパスワードに変更してください
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

---

## 実行方法

### 株価予測の実行 (create_stock_prediction_chart.py)

`create_stock_prediction_chart.py` を実行して、株価予測と評価、チャート生成を行います。

**基本コマンド:** 
```bash
python3 create_stock_prediction_chart.py [オプション]
```

**主なオプション:** 

| オプション | 説明 | デフォルト値 |
| :--- | :--- | :--- |
| `--stock_code` | 予測対象の証券コード。 | `9432` |
| `--model_type` | `lstm`, `nn`, `gru` からモデルを選択。 | `nn` |
| `--time_frame` | `daily` (日足), `weekly` (週足), `monthly` (月足) を選択。 | `daily` |
| `--epochs` | 学習のエポック数。 | `150` |
| `--seq_length` | 予測に用いる過去データのシーケンス長。 | `30` |
| `--fut_pred` | 何営業日先まで予測するか。 | `5` |
| `--hidden_unit_size` | 隠れユニット数 (LSTM, GRU用)。 | `128` |
| `--num_layers` | RNN層の数 (LSTM, GRU用)。 | `2` |
| `--nn_layer_units` | NNモデルの各隠れ層のユニット数をスペース区切りで指定。 | `100 50` |
| `--device` | `cpu` または `cuda` を指定。 | 自動検出 |

**実行例:** 
```bash
# 銘柄コード9432をデフォルトのNNモデルで予測
python3 create_stock_prediction_chart.py --stock_code 9432

# LSTMモデルで予測を実行
python3 create_stock_prediction_chart.py --model_type lstm

# NNモデルの隠れ層を256, 128ユニットで実行
python3 create_stock_prediction_chart.py --model_type nn --nn_layer_units 256 128
```

### 予測レポートの生成 (generate_report.py)

`create_stock_prediction_chart.py` の実行後、データベースに保存された結果を用いて、詳細なレポートをPDFまたはテキスト形式で生成できます。

**基本コマンド:** 
```bash
python3 generate_report.py [オプション]
```

**主なオプション:** 

| オプション | 説明 | デフォルト値 | 例 |
| :--- | :--- | :--- | :--- |
| `--batch_id` | レポート対象の実行ID (UUID)。指定しない場合は最新のものが自動で選択されます。 | `None` | `--batch_id ...` |
| `--format` | `pdf` または `text` から出力形式を選択。 | `pdf` | `--format text` |
| `--output_dir` | レポートの出力先ディレクトリ。 | `reports` | `--output_dir ./my_reports` |

**実行例:** 
```bash
# 最新の予測実行からPDFレポートを生成
python3 generate_report.py

# 最新の予測実行からテキストレポートを生成
python3 generate_report.py --format text
```

---

## ライセンス

このプロジェクトは Apache License, Version 2.0 の下でライセンスされています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。
