# 1. ベースとなるOSとPythonの環境を選ぶ
FROM python:3.11-slim

# 2. SudachiPyのビルドに必要な「Rust」をインストールする
#    まず、システムのパッケージリストを更新し、Rustをインストールするためのツールを導入
RUN apt-get update && apt-get install -y curl build-essential
#    次に、Rustコンパイラ自体をインストール
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#    インストールしたRustを使えるようにPATHを通す
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. アプリケーションの作業ディレクトリを作成
WORKDIR /app

# 4. 必要なPythonライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. アプリケーションのコードをコピー
COPY . .

# 6. アプリケーションを起動するコマンド
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]