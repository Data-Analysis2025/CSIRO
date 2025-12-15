# CSIRO Biomass ベースライン

CSIRO Biomass コンペ向けに、htmp-baseline と同じ構造（configs / scripts / src）でローカル学習 → Kaggle データセット公開 → Notebook から推論・提出までをまとめたテンプレートです。

## プロジェクト構成

```
CSIRO/
├── configs/                 # 実験設定（csiro_biomass.yaml を編集）
├── data/                    # Kaggle から取得したデータを配置（data/csiro_biomass/）
├── logs/                    # 学習ログ
├── models/                  # 保存される学習済みモデル
├── notebooks/               # 提出用 Notebook サンプル
├── scripts/                 # train/predict/列確認・Kaggleデータセット作成などのスクリプト
├── src/                     # 前処理・特徴量・モデル実装
└── requirements.txt
```

## セットアップ

```bash
cd /home/yamazono/DAS25/CSIRO
pip install -r requirements.txt  # またはお好みの仮想環境で
```

## データのダウンロード（Kaggle CLI）

```bash
# 事前準備: Kaggle APIキーを配置
# cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
# chmod 600 ~/.kaggle/kaggle.json

mkdir -p data/csiro_biomass
cd data/csiro_biomass
kaggle competitions download -c csiro-biomass
unzip csiro-biomass.zip
# 不要なら zip を削除
rm csiro-biomass.zip
```

期待する配置例:
```
data/csiro_biomass/
  ├── train.csv
  ├── test.csv           # 公開テストがある場合
  └── sample_submission.csv
```

## 列確認スクリプト（ターゲット列・drop_columns を決める）

```bash
python scripts/inspect_columns.py --train data/csiro_biomass/train.csv
```

## コンフィグ編集のガイド（configs/csiro_biomass.yaml）

`configs/csiro_biomass.yaml` を開いて、まず以下を実データに合わせて更新してください。

```yaml
target:
  column: TARGET_COLUMN_NAME      # 目的変数

cv:
  strategy: time_series           # 時系列でなければ kfold に変更
  time_column: date_id            # 時系列列名（無ければ削除）

features:
  drop_columns: []                # target や ID 列を追加
files:
  sample_submission: sample_submission.csv # 実際のファイル名に合わせる
```

- 時系列でない場合は `cv.strategy: kfold` に変更し、`time_column` は削除/コメントアウト。
- `drop_columns` にはターゲット列・ID列・不要列を入れる。
- Optuna はデフォルト無効 (`optuna.n_trials: 0`)。探索する場合は値を増やし、学習時に `--skip-optuna` を外します。

## 学習コマンド（ローカル）

```bash
python scripts/train.py --config configs/csiro_biomass.yaml --seed 42 --skip-optuna
```

成果物は `models/` と `logs/` に保存されます。Optuna も使う場合は `--skip-optuna` を付けずに実行してください。

## モデルとコードを Kaggle データセットにアップロード

```bash
python scripts/upload_kaggle_dataset.py \
  --dataset-id csiro-biomass-models \
  --dirs models src configs/csiro_biomass.yaml \
  --update \
  --message "CSIRO biomass model update"
```

- 初回は `--update` を外してください。
- `--dataset-id` は自分の名前空間に合わせて変更可能。
- kaggle CLI が一時ディレクトリを作るので、実行後に URL が出力されます。

## Notebook で推論して Submit する手順

1. Kaggle の `csiro-biomass` ページで **Code → New Notebook** を開く。
2. 右側の **Input** パネルで **+ Add Data** → 上で作ったデータセット（例: `YOUR_USERNAME/csiro-biomass-models`）を追加。
3. このリポジトリの `kaggle_submission_fixed.ipynb` をアップロードするか、中身をコピー。必要に応じてモデル読み込みパスを合わせる。
4. **Run All** で実行し、出力を確認。
5. 問題なければ **Submit to Competition** を押して提出。

## よく使うコマンドのまとめ

- 列確認: `python scripts/inspect_columns.py --train data/csiro_biomass/train.csv`
- 学習: `python scripts/train.py --config configs/csiro_biomass.yaml --seed 42 --skip-optuna`
- Kaggle データセット更新: `python scripts/upload_kaggle_dataset.py --dataset-id csiro-biomass-models --dirs models src configs/csiro_biomass.yaml --update --message "update"`

Happy Kaggling!
