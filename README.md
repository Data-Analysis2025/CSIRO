# Hull Tactical Market Prediction - ベースラインモデル

このリポジトリは、[Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) Kaggleコンペティションのベースライン実装です。

## 📊 プロジェクト概要

- **モデル**: Ridge回帰（5-fold時系列交差検証）
- **OOF RMSE**: 0.01503
- **特徴量**: 94個（元データから特定の列を除外）
- **予測**: トレーディングシグナル（0.0～2.0の範囲）

## 🏗️ プロジェクト構造

```
htmp-baseline/
├── configs/
│   └── default.yaml          # 実験設定ファイル
├── data/
│   ├── train.csv            # 訓練データ
│   ├── test.csv             # テストデータ（ローカル用サンプル）
│   └── kaggle_evaluation/   # Kaggle評価システム
├── models/                  # 訓練済みモデル（.pklファイル）
├── src/
│   ├── config.py           # 設定読み込みユーティリティ
│   ├── features.py         # 特徴量エンジニアリング
│   └── model.py            # モデル定義
├── scripts/
│   ├── train.py            # モデル訓練スクリプト
│   └── predict.py          # ローカル予測スクリプト
├── submissions/            # 提出ファイル
├── logs/                   # ログファイル
├── upload_kaggle_dataset.py  # Kaggleデータセットアップロード
└── kaggle_submission_fixed.ipynb  # Kaggle提出用Notebook（最終版）
```

## 🚀 セットアップ

### 1. 環境構築

```bash
# Condaまたはvenv環境を作成（推奨：Python 3.11）
conda create -n kaggle python=3.11
conda activate kaggle

# 必要なパッケージをインストール
pip install pandas numpy scikit-learn==1.2.2 pyyaml joblib polars
pip install kaggle  # Kaggle API用
```

### 2. データのダウンロード

```bash
# Kaggle APIの設定（~/.kaggle/kaggle.jsonが必要）
# https://www.kaggle.com/docs/api から取得

# コンペデータをダウンロード
cd data
kaggle competitions download -c hull-tactical-market-prediction
unzip hull-tactical-market-prediction.zip
```

## 🎯 ローカルでのモデル訓練

### 訓練の実行

```bash
# デフォルト設定でモデルを訓練
python scripts/train.py --config configs/default.yaml

# 出力例:
# Fold 0: RMSE=0.01971
# Fold 1: RMSE=0.01384
# Fold 2: RMSE=0.01935
# Fold 3: RMSE=0.01329
# Fold 4: RMSE=0.01308
# Overall OOF RMSE: 0.01503
```

### 訓練されるもの

- `models/default_fold_0.pkl` ～ `models/default_fold_4.pkl`: 各Foldのモデル
- `models/default_metadata.json`: メタデータ（設定、スコアなど）
- `logs/default_train.json`: 訓練ログ

## 📤 Kaggleへの提出手順（重要）

このコンペティションは **Code Competition** 形式です。CSVファイルではなく、**Notebookを提出**します。

### 全体の流れ

```
ローカル環境でモデル訓練
    ↓
Kaggleデータセットにアップロード
    ↓
Kaggle Notebookで推論コード作成
    ↓
Notebookを提出（Submit to Competition）
    ↓
スコア確認
```

---

### ステップ1: ローカルでモデルを訓練

```bash
# sklearn 1.2.2をインストール（Kaggle環境に合わせる）
pip install scikit-learn==1.2.2

# モデルを訓練
python scripts/train.py --config configs/default.yaml
```

**重要**: Kaggle環境のsklearnバージョン（1.2.2）に合わせることで、バージョン不一致の警告を回避できます。

---

### ステップ2: モデルとコードをKaggleデータセットにアップロード

```bash
# 初回アップロード（新規作成）
python upload_kaggle_dataset.py \
  --dataset-id htmp-baseline-models \
  --dirs models src

# 更新（既存データセットを更新する場合）
python upload_kaggle_dataset.py \
  --dataset-id htmp-baseline-models \
  --dirs models src \
  --update \
  --message "Updated models with sklearn 1.2.2"
```

**成功すると以下のように表示されます:**
```
✅ Dataset uploaded successfully!
Dataset URL: https://www.kaggle.com/datasets/YOUR_USERNAME/htmp-baseline-models
```

---

### ステップ3: Kaggle Notebookを作成

#### 3-1. 新しいNotebookを作成

1. [Kaggleのコンペティションページ](https://www.kaggle.com/competitions/hull-tactical-market-prediction)にアクセス
2. 上部メニューの **"Code"** タブをクリック
3. **"New Notebook"** ボタンをクリック

#### 3-2. Notebookの内容をコピー

`kaggle_submission_fixed.ipynb` の内容を全てコピーして、新しいNotebookに貼り付けます。

**または**、ローカルの `.ipynb` ファイルを直接アップロードすることもできます：
- Kaggle Notebookエディタで **"File" → "Upload Notebook"** を選択

---

### ステップ4: データセットを追加（重要）

Notebook右側の **"Input"** パネルで：

1. **"+ Add Data"** ボタンをクリック
2. 検索バーに **"htmp-baseline-models"** と入力
3. あなたのデータセット（`YOUR_USERNAME/htmp-baseline-models`）を選択
4. **最新バージョン**を選択（通常は最も大きいバージョン番号）
5. **"Add"** ボタンをクリック

**確認**: 右側のInputパネルに以下が表示されることを確認
```
📁 htmp-baseline-models
   ├── models/
   └── src/
```

---

### ステップ5: Notebookを実行（テスト）

提出前にローカルテストを実行します：

1. Notebook上部の **"Run All"** ボタンをクリック
2. 実行完了を待つ（約20秒）
3. 出力を確認：

```
✅ sklearn: 1.2.2
✅ All models loaded: 5 folds
Signal conversion function defined
Example: ret=0.006 -> signal=2.000
🧪 Running local gateway for testing...
```

**エラーがないことを確認してください。**

---

### ステップ6: Notebookを提出（本番）⭐

#### 6-1. Notebookを保存

1. Notebook右上の **"Save Version"** ボタンをクリック
2. ドロップダウンメニューが表示されるので、以下を選択：
   - **"Save & Run All (Commit)"** を選択
   - または **"Quick Save"** → **"Save & Run All"**
3. 実行完了を待つ（約20秒）

#### 6-2. 提出ボタンをクリック

実行が完了すると、Notebook右上に **"Submit to Competition"** ボタンが表示されます。

1. **"Submit to Competition"** ボタンをクリック
2. 確認ダイアログが表示される
3. **"Submit"** ボタンを再度クリック

**重要**:
- 「Submit」ボタンを押さないと提出されません
- 「Run All」だけではローカルテストのみで、提出にはなりません

---

### ステップ7: 提出結果を確認

#### 7-1. 提出履歴を確認

1. コンペティションページの上部タブ **"My Submissions"** をクリック
2. 提出履歴が表示される
3. 以下を確認：
   - **Status**: "Complete"（成功）または "Error"（失敗）
   - **Score**: スコアが表示される
   - **Daily Submissions**: `X / 5 used` （1日5回まで）

#### 7-2. スコアの確認

提出が成功すると：
```
Latest Score: X.XX
Best Score: X.XX
Daily Submissions: 1 / 5 used
```

#### 7-3. エラーが出た場合

エラーが出た場合：
1. 提出履歴で該当の提出をクリック
2. **"Logs"** タブをクリック
3. エラーメッセージを確認
4. 下記の「トラブルシューティング」を参照

---

## ⚙️ 設定ファイル（configs/default.yaml）

```yaml
# 実験名
run_name: default

# ランダムシード
seed: 42

# パス設定
paths:
  input_dir: data
  processed_dir: data/processed
  models_dir: models
  submissions_dir: submissions
  logs_dir: logs

# 目的変数
target:
  column: market_forward_excess_returns

# 交差検証
cv:
  strategy: time_series  # 時系列分割
  n_splits: 5
  time_column: date_id

# 特徴量エンジニアリング
features:
  drop_columns:
    - date_id
    - forward_returns
    - risk_free_rate
    - market_forward_excess_returns
    - lagged_forward_returns
    - lagged_risk_free_rate
    - lagged_market_forward_excess_returns
    - is_scored
  imputation_strategy: median
  scale: true

# モデル
model:
  type: ridge
  params:
    alpha: 1.0
```

## 🔑 重要なポイント

### 1. ⭐ シグナル変換が必須

このコンペティションで最も重要なポイントです。

**❌ 間違い（生のリターン予測を返す）:**
```python
def predict(test: pl.DataFrame) -> float:
    raw_pred = model.predict(X)
    return raw_pred  # 0.004 など → エラー！
```

**✅ 正しい（シグナルに変換して返す）:**
```python
def convert_ret_to_signal(ret_pred: float) -> float:
    """リターン予測をシグナルに変換

    0.0 = ショートポジション（市場下落予想）
    1.0 = ニュートラル（ポジションなし）
    2.0 = ロングポジション（市場上昇予想）
    """
    signal = ret_pred * 400.0 + 1.0
    return np.clip(signal, 0.0, 2.0)

def predict(test: pl.DataFrame) -> float:
    raw_pred = model.predict(X)
    signal = convert_ret_to_signal(raw_pred)  # シグナルに変換
    return signal  # 0.0～2.0
```

**シグナルの意味:**
- `0.0`: 市場が大きく下落すると予想（ショートポジション）
- `1.0`: 中立（ポジションなし）
- `2.0`: 市場が大きく上昇すると予想（ロングポジション）

---

### 2. scikit-learnのバージョン

**Kaggle環境のデフォルトバージョンに合わせる:**
```bash
pip install scikit-learn==1.2.2
```

**理由:**
- Kaggle環境: sklearn 1.2.2
- バージョンが異なると、モデルロード時に警告やエラーが発生
- 警告があると予測が不安定になる可能性

---

### 3. Code Competition形式

このコンペは **Code Competition** です：

| 項目 | 説明 |
|-----|------|
| **ローカルのtest.csv** | 開発用サンプル（10行のみ） |
| **本番のテストデータ** | Kaggle上で評価システム経由で提供（非公開） |
| **提出方法** | Notebookを提出（CSVファイルではない） |
| **評価** | Notebookが本番環境で実行され、リアルタイムで予測 |

---

### 4. 予測関数の仕様

```python
def predict(test: pl.DataFrame) -> float:
    """1行ずつ予測する関数

    Args:
        test (pl.DataFrame): 1行のPolars DataFrame
            - 各date_idごとに1回呼ばれる
            - 列: date_id, D1, D2, ..., lagged_*

    Returns:
        float: トレーディングシグナル（0.0～2.0）
    """
    # testは1行のDataFrame
    # 必ず0.0～2.0の範囲の値を返す
```

**重要:**
- `predict()` は各date_idごとに1回呼ばれる
- 入力は1行のDataFrame
- 返り値は必ず `0.0 <= signal <= 2.0`

---

## 🐛 トラブルシューティング

### エラー1: "Submission Scoring Error: incorrect format"

**原因**: 予測関数が生のリターン（例: 0.004）を返している

**解決策**: シグナル変換を追加
```python
signal = convert_ret_to_signal(raw_pred)
return signal  # 0.0～2.0
```

---

### エラー2: sklearn バージョン警告

**症状:**
```
UserWarning: Trying to unpickle estimator Ridge from version 1.7.1 when using version 1.2.2
```

**原因**: ローカルとKaggleでsklearnバージョンが異なる

**解決策**:
```bash
# ローカル環境を1.2.2にダウングレード
pip install scikit-learn==1.2.2

# モデルを再トレーニング
python scripts/train.py --config configs/default.yaml

# Kaggleデータセットを更新
python upload_kaggle_dataset.py --dataset-id htmp-baseline-models --dirs models src --update
```

---

### エラー3: "Dataset not found"

**症状:**
```
❌ Dataset not found at /kaggle/input/htmp-baseline-models
```

**原因**: Kaggle Notebookでデータセットを追加していない

**解決策**:
1. Notebook右側の **"Input"** パネル
2. **"+ Add Data"** をクリック
3. あなたのデータセット（`YOUR_USERNAME/htmp-baseline-models`）を追加
4. **最新バージョン**を選択

---

### エラー4: サーバー起動タイムアウト（15分制限超過）

**症状:**
```
RuntimeWarning: 1500 seconds elapsed before server startup.
This exceeds the startup time limit of 900 seconds
```

**原因**: pip installに時間がかかりすぎる、またはモデルロードが遅い

**解決策**:
1. pip installを削除（Kaggle環境のデフォルトバージョンを使う）
2. モデルをサーバー起動前にロード
```python
# サーバー起動前にモデルをロード
load_models()

# 予測関数ではロード済みモデルを使用
def predict(test):
    # モデルは既にロード済み
    ...

# サーバー起動
inference_server.serve()
```

---

### エラー5: "Submit to Competition" ボタンが見つからない

**原因**: Notebookの実行が完了していない、またはバージョン保存していない

**解決策**:
1. **"Save Version"** をクリック
2. **"Save & Run All (Commit)"** を選択
3. 実行完了を待つ
4. 画面右上に **"Submit to Competition"** ボタンが表示される

---

## 📈 スコア改善のアイデア

- [ ] **特徴量エンジニアリング**
  - ラグ特徴量（過去N日のデータ）
  - ローリング統計量（移動平均、移動標準偏差など）
  - 特徴量の交互作用項

- [ ] **モデルの改善**
  - ハイパーパラメータチューニング（alphaの最適化）
  - 他のモデルを試す（ElasticNet、LightGBM、XGBoostなど）
  - アンサンブル（複数モデルの予測を組み合わせ）

- [ ] **シグナル変換の最適化**
  - `SIGNAL_MULTIPLIER`（現在400.0）の調整
  - `MIN_SIGNAL`, `MAX_SIGNAL`の範囲を変更

- [ ] **交差検証の改善**
  - Fold数の調整
  - Purged K-Fold（時系列データ用の特殊なCV）

---

## 📚 参考資料

- [コンペティションページ](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- [公式ディスカッション](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion)
- [Code Competition の仕組み](https://www.kaggle.com/docs/competitions#code-competitions)
- [Kaggle API ドキュメント](https://github.com/Kaggle/kaggle-api)

---

## 📝 よくある質問（FAQ）

**Q1: 提出したはずなのにスコアが表示されない**

A: 以下を確認してください：
- "Run All" だけでなく、"Submit to Competition" ボタンを押しましたか？
- 提出履歴（My Submissions）に記録がありますか？
- Status が "Error" になっていませんか？

---

**Q2: ローカルのtest.csvで予測が動作するのに、本番でエラーが出る**

A: よくある原因：
- シグナル変換を忘れている
- sklearnバージョンが異なる
- 本番データの構造がローカルと若干異なる

---

**Q3: 1日に何回まで提出できますか？**

A: 1日5回までです。Daily Submissionsで確認できます。

---

## 🤝 貢献

改善案やバグ報告は Issue または Pull Request でお願いします。

---

## 📄 ライセンス

このテンプレートは教育目的で提供されています（Data Analysis 2025セミナー）。

---

**Happy Kaggling! 🚀**

---

## 🌱 CSIRO Biomass コンペ用の流れ（同じ構造で再利用）

手元でモデルを学習し、Kaggle Notebookで推論する方式をそのまま流用できます。まずはデータのダウンロードと列確認から。

### 1. データ取得（Kaggle CLI）
```
# Kaggle APIセットアップ後
mkdir -p data/csiro_biomass
cd data/csiro_biomass
kaggle competitions download -c csiro-biomass
unzip csiro-biomass.zip
```

### 2. 列とターゲットの確認
```
python scripts/inspect_columns.py --train data/csiro_biomass/train.csv
```
出力を見て、ターゲット列名や時系列列を把握します。

### 3. コンフィグを編集
`configs/csiro_biomass.yaml` を開き、以下を必ず調整してください。
- `target.column`: 目的変数の列名
- `cv.time_column`: 時系列列がある場合はその列名。ない場合は `strategy: kfold` に変更
- `features.drop_columns`: ターゲット列やID列を追加

デフォルトでは Optuna は `n_trials: 0` で無効化されています。必要なら値を増やし、`--skip-optuna`を外して実行してください。

### 4. 学習の実行（Optunaなし）
```
python scripts/train.py --config configs/csiro_biomass.yaml --seed 42 --skip-optuna
```
GPUが使える環境なら自動で `device=gpu` を指定します。Optunaを使う場合は `--skip-optuna` を付けずに、`optuna.n_trials` を増やしてください。

### 5. Kaggleデータセットへのアップロード
```
python upload_kaggle_dataset.py \
  --dataset-id csiro-biomass-models \
  --dirs models src configs/csiro_biomass.yaml \
  --update \
  --message "CSIRO biomass model update"
```
（初回は `--update` を外す）

### 6. Kaggle Notebookで推論
1. Kaggleの`csiro-biomass`コンペページ → Code → New Notebook
2. 右側Inputパネルで `+ Add Data` → 自分の `csiro-biomass-models` データセットを追加
3. ローカルの推論ノートブック（例: `kaggle_submission_fixed.ipynb` を流用し、モデル/特徴読み込み部分をパスに合わせて修正）を貼り付け
4. Run All → Submit to Competition

### 7. 補足
- 進捗バーを見やすくするため、Optunaを使わない場合はCVのみのバーが出ます。
- データ列やタスク（回帰/分類）が異なるため、`configs/csiro_biomass.yaml`の`model`・`metric`・`drop_columns`は実データに合わせて微調整してください。
