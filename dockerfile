# 軽量なベースイメージを指定
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 環境変数の設定（Pythonのバッファリング無効化）
ENV PYTHONUNBUFFERED=1

# 必要なパッケージとNode.jsをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        gnupg \
        wget \
        unzip \
        openssh-client \
        iputils-ping \
        fonts-takao-gothic fonts-takao-mincho fonts-takao-pgothic \
        build-essential python3-dev libffi-dev libssl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app/src/

# 依存関係ファイルをコピーしてインストール
COPY ./requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip cache purge && \
    pip install --timeout=600 --no-cache-dir -r requirements.txt

# コンテナを終了させずに起動し続ける設定
ENTRYPOINT ["tail", "-f", "/dev/null"]