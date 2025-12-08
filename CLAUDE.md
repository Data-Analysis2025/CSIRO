# CLAUDE.md

このファイルは、このリポジトリでコードを扱う際の Claude Code (claude.ai/code) への指針を提供します。

## プロジェクト概要

これは「Hull Tactical Market Prediction」の Kaggle コンペティションプロジェクトです。S&P 500の超過リターンを予測し、120%のボラティリティ制約内でベンチマークをアウトパフォームする機械学習モデルを実装しています。

詳細なコンペティション情報については、[COMPETITION_OVERVIEW.md](./COMPETITION_OVERVIEW.md)を参照してください。

## セットアップ

### 依存関係のインストール

```bash
# pipを使用
pip install -r requirements.txt

# または、Makefileを使用
make install
```

### Kaggleデータのダウンロード

```bash
# 方法1: Makefileを使用（推奨）
make download

# 方法2: 直接コマンドを実行
kaggle competitions download -c hull-tactical-market-prediction -p data/raw
unzip data/raw/hull-tactical-market-prediction.zip -d data/raw/

# ファイルの確認
ls data/raw/
# → train.csv, test.csv, sample_submission.csv が存在することを確認
```

## モデルの学習と予測

### 基本的なワークフロー

```bash
# 方法1: Makefileを使用（推奨）
make train       # モデルの学習
make predict     # 予測の生成

# 方法2: 直接Pythonスクリプトを実行
python scripts/train.py --config configs/default.yaml
python scripts/predict.py --config configs/default.yaml

# 生成されたファイルの確認
ls models/        # → default_fold_*.pkl, default_metadata.json
ls submissions/   # → default_submission.csv
```

### シード値の変更

再現性を保ちつつ異なる乱数シードで実験する場合：

```bash
python scripts/train.py --config configs/default.yaml --seed 123
python scripts/predict.py --config configs/default.yaml --seed 123
```

### カスタム設定での実験

```bash
# 新しい設定ファイルを作成
cp configs/default.yaml configs/experiment01.yaml

# experiment01.yamlを編集（モデルタイプ、ハイパーパラメータなど）

# 新しい設定で学習と予測
python scripts/train.py --config configs/experiment01.yaml
python scripts/predict.py --config configs/experiment01.yaml
```

## Kaggleデータセットへのアップロード

学習済みモデルをKaggle Notebookで使用するため、Kaggleデータセットとしてアップロードします。

### 方法1: Makefileを使用（推奨）

```bash
# 初回アップロード（新しいデータセットを作成）
make upload

# 2回目以降（既存データセットを更新）
make upload-update
```

### 方法2: シェルスクリプトを使用

```bash
# 初回アップロード
./scripts/upload_models.sh

# データセット更新
./scripts/upload_models.sh --update

# カスタムメッセージで更新
./scripts/upload_models.sh --update "Added new features and retrained models"
```

### 方法3: Pythonスクリプトを直接使用

```bash
# 初回アップロード（models/ と src/ をアップロード）
python upload_kaggle_dataset.py \
    --dataset-id htmp-baseline-models \
    --dirs models src

# データセット更新
python upload_kaggle_dataset.py \
    --dataset-id htmp-baseline-models \
    --dirs models src \
    --update \
    --message "Updated models with new features"

# configs/ も含めてアップロード
python upload_kaggle_dataset.py \
    --dataset-id htmp-baseline-models \
    --dirs models src configs \
    --update

# Kaggleユーザー名を明示的に指定
python upload_kaggle_dataset.py \
    --dataset-id htmp-baseline-models \
    --dirs models src \
    --username your-username
```

### アップロードされるファイル

デフォルトでは以下のディレクトリがアップロードされます：

- **models/**: 学習済みモデルファイル（`*.pkl`）とメタデータ（`*.json`）
- **src/**: 推論時に必要なPythonモジュール（`config.py`, `features.py`, `model.py`）
- **configs/** (オプション): 実験設定ファイル

### アップロード後の確認

```bash
# データセットURLが表示される
# https://www.kaggle.com/datasets/your-username/htmp-baseline-models

# Kaggle Notebookで使用する際は、以下を追加データとして指定：
# your-username/htmp-baseline-models
```

## Kaggle Notebookでの提出

### 提出用Notebookの構造

`notebooks/hull-tactical-market-prediction-demo-submission.ipynb` を基に、以下の構造で実装：

1. **モデルの読み込み**（グローバルスコープで1回のみ）
   - Kaggleデータセットから学習済みモデルをロード
   - 全foldのモデルとscalerを準備

2. **predict関数の実装**
   - 各取引日ごとに呼び出される
   - 入力: Polars DataFrameの特徴量データ
   - 出力: S&P 500への配分比率（0.0～2.0の範囲）
   - 制約: 5分以内に応答が必要

3. **inference_serverの起動**
   - Kaggle評価API経由でリアルタイム予測を提供
   - 初回呼び出しは15分以内に完了する必要あり

詳細な実装例はREADMEを参照してください。

## コードアーキテクチャ

```
htmp-baseline/
├── configs/                 # 実験設定ファイル（YAML形式）
│   └── default.yaml         # デフォルト設定
├── data/
│   ├── raw/                 # Kaggleからダウンロードした生データ
│   ├── processed/           # 特徴量エンジニアリング後のデータ
│   └── kaggle_evaluation/   # Kaggle評価API（protobuf）
├── logs/                    # 学習ログと実験メタデータ
├── models/                  # 学習済みモデルの保存先
├── notebooks/               # Kaggle提出用ノートブック
├── scripts/                 # CLIエントリーポイント
│   ├── train.py             # モデル学習スクリプト
│   └── predict.py           # 予測生成スクリプト
├── src/                     # 再利用可能なPythonモジュール
│   ├── config.py            # 設定読み込みユーティリティ
│   ├── features.py          # 特徴量抽出と前処理
│   └── model.py             # モデルファクトリとCV
├── submissions/             # 生成されたKaggle提出ファイル
└── tutorials/               # データサイエンス学習用チュートリアル
    └── ds_handmade_ml/      # MNIST hands-onチュートリアル
```

## 主要な技術的詳細

### データとターゲット

- **入力データ**: 数十年分の日次市場データ（古いデータには欠損値が多い）
- **特徴量カテゴリ**:
  - `M*` - Market Dynamics/Technical features
  - `E*` - Macro Economic features
  - `I*` - Interest Rate features
  - `P*` - Price/Valuation features
  - `V*` - Volatility features
  - `S*` - Sentiment features
  - `MOM*` - Momentum features
  - `D*` - Dummy/Binary features
- **ターゲット**: `market_forward_excess_returns` - 5年移動平均を差し引き、MADでwinsorizeした超過リターン

### 特徴量エンジニアリング

`src/features.py`の`SimpleFeatureExtractor`で実装：

- **欠損値補完**: mean, median, zero fill
- **スケーリング**: StandardScaler（訓練データでfit、テストデータに適用）
- **ローリング統計**: 時系列・グループ別の移動平均・標準偏差（オプション）
- **特徴量交互作用**: 乗算による組み合わせ特徴量（オプション）

### クロスバリデーション

- **TimeSeriesSplit**: 時系列データの特性を考慮（デフォルト）
- **KFold**: 標準的なk分割交差検証

### モデル

現在実装されているモデル（`src/model.py`）:

- **Ridge回帰** (デフォルト): L2正則化線形回帰
- **Linear回帰**: 標準線形回帰

設定ファイルで簡単に切り替え可能。将来的にLightGBM、XGBoostなどの追加が推奨されます。

### 評価指標

- **学習中**: RMSE (Root Mean Squared Error)
- **Kaggleコンペ**: Modified Sharpe Ratio（120%ボラティリティ制約付き）

### Kaggle評価API

- **プロトコル**: Protocol Buffers (gRPC)
- **応答時間制限**: 各バッチ5分以内（初回は15分）
- **予測単位**: 1取引日ごとにリアルタイム予測

## 開発ツール

- **Python**: 3.8以上推奨
- **依存パッケージ**:
  - numpy >= 1.24
  - pandas >= 2.0
  - scikit-learn >= 1.3
  - pyyaml >= 6.0
  - joblib >= 1.3
- **Kaggle CLI**: データダウンロードとデータセットアップロードに必要

## 設定ファイルの構造

`configs/default.yaml` の主要セクション:

```yaml
run_name: default              # 実験名
seed: 42                       # 乱数シード

paths:                         # ディレクトリパス
  input_dir: data/raw
  processed_dir: data/processed
  models_dir: models
  submissions_dir: submissions
  logs_dir: logs

files:                         # ファイル名
  train: train.csv
  test: test.csv
  sample_submission: sample_submission.csv

target:                        # ターゲット設定
  column: market_forward_excess_returns
  prediction_column: market_forward_excess_returns

cv:                            # クロスバリデーション
  strategy: time_series        # 'time_series' or 'kfold'
  n_splits: 5
  time_column: date_id

features:                      # 特徴量エンジニアリング
  drop_columns: [date_id, ...]
  imputation_strategy: median  # 'mean', 'median', 'zero'
  scale: true
  rolling_windows: null        # 例: [5, 10, 20]
  enable_interactions: false

model:                         # モデル設定
  type: ridge                  # 'ridge' or 'linear_regression'
  params:
    alpha: 1.0                 # Ridge正則化パラメータ
  fit_intercept: true

inference:                     # 推論設定
  average_folds: true          # fold予測の平均化
  output_filename: default_submission.csv
```

## トラブルシューティング

### データが見つからない

```bash
# data/raw/ ディレクトリが存在し、CSVファイルが配置されているか確認
ls -l data/raw/
```

### モデルが見つからない

```bash
# train.pyを実行してからpredict.pyを実行
python scripts/train.py --config configs/default.yaml
```

### Kaggle API認証エラー

```bash
# ~/.kaggle/kaggle.json に認証情報が配置されているか確認
ls -l ~/.kaggle/kaggle.json

# ファイルのパーミッションを確認（600である必要あり）
chmod 600 ~/.kaggle/kaggle.json
```

## 参考リンク

- [Kaggleコンペページ](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- [Efficient Market Hypothesis (Wikipedia)](https://en.wikipedia.org/wiki/Efficient-market_hypothesis)
- [Sharpe Ratio (Investopedia)](https://www.investopedia.com/terms/s/sharperatio.asp)
