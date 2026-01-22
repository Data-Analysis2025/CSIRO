# CSIRO Image2Biomass（ローカル学習 + Kaggle提出）

このリポジトリは、**ローカルで画像モデルを学習**し、重みを **Kaggle Dataset にアップロード**して、**Kaggle Notebook で submission.csv を生成・提出**するための最小構成テンプレートです。

---

## 改善の流れと現状の処理概要

- **Baseline (ResNet 画像のみ)**  
  - `scripts/train_image.py` で `custom_resnet18/34` をゼロから学習し、`scripts/predict_image.py` + Notebook で提出。  
  - 画像パス + target を単純 pivot しただけの最小パイプラインを README に沿って再現可能。

- **データ配置のフレンドリー化**  
  - すべてのスクリプトと設定が `data/` 直下または旧来の `data/csiro_biomass/` のどちらでも自動認識するよう `path_utils.py` を追加。  
  - Kaggle からの zip 展開先を変えても FileNotFound にならないよう統一済み。

- **高性能埋め込みの導入**  
  - `scripts/extract_features_large.py` で SigLIP + DINOv2 Large + CLIP をまとめて推論し、TTA + L2 正規化した 2688 次元特徴を `train/test_embeddings_large.csv` に保存。  
  - safetensors のみを使用し、ローカルキャッシュがあれば再ダウンロードを避けるよう改善。

- **ツリー系モデルでのアンサンブル**  
  - `scripts/train_large_models.py` が上記埋め込みを特徴量に、LGBM/XGB/CatBoost を MultiOutput + 5fold で学習。  
  - `models/large_ensemble/` に fold ごとの scaler + estimator を保存し、OOF を競技スコアで評価するところまで自動化済み。

- **ViT/CrossPVT 系の再利用**  
  - `scripts/run_crosspvt_inference.py` や `src/inference_crosspvt.py` を Kaggle Notebook から呼び出し、Fold 平均で最終 submission を作成。  
  - モデルと `src/`、`scripts/` を Kaggle Dataset にまとめてアップロードして再現。

- **timm 製 Conv モデルの追加 (New!)**  
  - `scripts/train_image.py` / `scripts/predict_image.py` が `convnextv2_large`, `efficientnetv2_l` など timm モデルを直接受け付けるよう拡張。  
  - ViT 系モデルと畳み込み系モデルを並列学習→`scripts/ensemble.py` などでブレンドする構成を推奨。

現状は **「画像特徴を SigLIP/DINO/CLIP で抽出 → 木系モデルのアンサンブル → Kaggle で CrossPVT も併用可能」** という構成です。Baseline との違いが README だけで把握できるよう、この節に最新の改善内容を追記していきます。

---

## 1. セットアップ

```bash
pip install -r requirements.txt
```

---

## 2. データの取得

```bash
kaggle competitions download -c csiro-biomass -p data
cd data
unzip csiro-biomass.zip
```

データ構成：

```
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/        # 画像
└── test/         # 公開テスト画像
```

> NOTE: 以前のテンプレートでは `data/csiro_biomass` 配下を前提としていましたが、全スクリプトは自動で `data` / `data/csiro_biomass` のどちらかを検出するよう更新済みです。上記のように `data/` 直下へ展開すればそのまま動作します。

---

## 3. ローカル学習（自作ResNet）

以下は `custom_resnet34` を **ゼロから学習**する例です。

```bash
python scripts/train_image.py \
  --data-dir data \
  --out-dir models \
  --model-name custom_resnet34 \
  --image-size 128 \
  --epochs 10 \
  --batch-size 8
```

学習後に `models/image_custom_resnet34.pt` が生成されます。

---

## 4. ローカル推論（任意）

```bash
python scripts/predict_image.py \
  --data-dir data \
  --model-path models/image_custom_resnet34.pt \
  --output submissions/submission.csv
```

---

## 5. Kaggle用にモデルとコードをアップロード

```bash
python scripts/upload_kaggle_dataset.py \
  --dataset-id YOUR_USERNAME/csiro-biomass-models \
  --dirs models src \
  --update \
  --message "custom_resnet34 weights"
```

※ `YOUR_USERNAME` は自分の Kaggle ユーザー名に変更してください。

---

## 6. Kaggle Notebookで提出

1. Kaggleで新規 Notebook を作成  
2. Input に以下を追加  
   - `csiro-biomass`（コンペデータ）  
   - `YOUR_USERNAME/csiro-biomass-models`（アップロードした models + src）
3. `notebooks/kaggle_image_inference.ipynb` をアップロード  
4. **Run All → Output から `submission.csv` を提出**

Notebookは以下に出力します：

```
/kaggle/working/submission.csv
```

---

## 7. TIMM Conv モデル（ConvNeXt/EfficientNet 等）での追加学習

timm のモデル名をそのまま `--model-name` に渡すだけで、ImageNet 事前学習済みの畳み込みモデルを fine-tune できます。例：ConvNeXtV2 Large（入力解像度 384）。

```bash
python scripts/train_image.py \
  --data-dir data \
  --out-dir models \
  --model-name convnextv2_large \
  --image-size 576 \
  --batch-size 8 \
  --epochs 10 \
  --folds 5 \
  --lr 5e-5 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --mixup-alpha 0.4 \
  --use-albumentations \
  --pretrained
```

Albumentations ベースの強い増強（RandomResizedCrop, ShiftScaleRotate, CoarseDropout 等）、Mixup/CutMix、Cosine LR などを備えており、Fold ごとの RMSE/R² を確認しながら ViT 系と近い条件で fine-tune できます。1-fold の場合も `fold_0` 配下に保存され、互換用に `models/image_convnextv2_large.pt` が自動コピーされます。

推論も同じスクリプトで OK です。チェックポイント内に `model_source` を記録しているので自動的に正しいアーキテクチャを復元します。

```bash
python scripts/predict_image.py \
  --data-dir data \
  --model-path models/convnextv2_large/fold_0/best.pt \
  --output submissions/submission_convnext.csv
```

ViT 系（CrossPVT）や SigLIP/DINO 埋め込みで得た CSV と、この Conv 系モデルの予測を `scripts/ensemble.py` などで重み付き平均すると、相関の低さを活かして LB/OOF をさらに押し上げられます。

### 7.1 Conv 特徴量 → LGBM/XGB/CatBoost への展開

ConvNeXt などで fine-tune した checkpoint を使い、SigLIP/DINO と同様に「特徴量化 → 木系モデル3種」を回せます。

1. **埋め込み作成**（複数 Fold を平均する例）
   ```bash
   python scripts/extract_conv_features.py \
     --data-dir data \
     --checkpoint-paths models/convnextv2_large/fold_0/best.pt models/convnextv2_large/fold_1/best.pt \
     --output-prefix convnext_ft
   ```
   `data/train_embeddings_convnext_ft.csv` / `data/test_embeddings_convnext_ft.csv` が生成されます。

2. **木系モデルの学習**
   ```bash
   python scripts/train_large_models.py \
     --train-embeddings data/train_embeddings_convnext_ft.csv \
     --models-dir models/large_ensemble_convnext
   ```
   既存の SigLIP+DINO+CLIP と同じく LGBM/XGB/CatBoost の 5-fold 学習と OOF 評価が走ります。

---

## 8. CrossPVT / DINOv2 Fine-tuning & 推論

`scripts/train_crosspvt.py` は DINOv2 Small をバックボーンにした CrossPVT_T2T_MambaDINO を 5-fold で学習します。実行例（学習には GPU が必須です）:

```bash
python scripts/train_crosspvt.py \
  --train-csv data/train.csv \
  --image-dir data/train \
  --out-dir models \
  --run-name crosspvt_imgauto_bs8_lr0.0001_$(date +%Y%m%d_%H%M%S) \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --folds 5 \
  --seed 42 \
  --num-workers 4 \
  --dropout 0.1 \
  --hidden-ratio 0.35 \
  --img-size 576
```

Fold ごとの checkpoint（`models/<run-name>/fold_X/checkpoints/best_wr2.pt`）を Kaggle Dataset に含め、Notebook から以下で推論します:

```bash
python scripts/run_crosspvt_inference.py \
  --checkpoint-root /kaggle/input/YOUR_DATASET/models \
  --run-name crosspvt_imgauto_bs8_lr0.0001_20260113_130321 \
  --competition-root /kaggle/input/csiro-biomass \
  --folds 0 1 2 3 4 \
  --ckpt-filename best_wr2.pt \
  --output submissions/submission_crosspvt.csv
```

Conv 系（Section 7）や SigLIP/DINO 埋め込みモデルと合わせて重み付き平均すれば、ViT / Conv / ツリー系の三系統アンサンブルが完成します。

---

## 9. アンサンブルの作成

`scripts/ensemble.py` を使うと任意の submission CSV を重み付き平均できます。デフォルトでは LGBM/XGB/CatBoost を対象にしていますが、ConvNeXt（`submission_convnext.csv`）や CrossPVT（`submission_crosspvt.csv`）なども自由に追加可能です。

```bash
python scripts/ensemble.py \
  --models lgbm xgb cat convnext crosspvt \
  --weights 0.25 0.25 0.2 0.15 0.15 \
  --output submission_blend.csv
```

`alias=path/to/file.csv` の形式で任意のファイルを直接指定できます。`--weights` を省略すると均等重みが自動で適用されます。

---

## ファイル説明

- `scripts/train_image.py`：ローカル学習
- `scripts/predict_image.py`：ローカル推論
- `scripts/extract_conv_features.py`：Conv fine-tuned checkpoint から埋め込み作成
- `src/models_image.py`：自作ResNet定義
- `notebooks/kaggle_image_inference.ipynb`：Kaggle推論用Notebook

---

## 備考

- 出力は **5つのターゲットを同時回帰**します  
  `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`
- `custom_resnet18` も使用可能です
### 7.2 Conv FT → 木系モデルまで一括実行するスクリプト（New）

以下のシェルスクリプトで、ConvNeXt などの 5-fold fine-tune → 特徴抽出 → LGBM/XGB/CatBoost 学習までをまとめて回せます（Albumentations + Mixup/CutMix + Cosine LR など、上記オプションをデフォルトで有効化）。

```bash
bash scripts/run_conv_ft_pipeline.sh
```

環境変数で各種パラメータを上書きできます（例: `MODEL_NAME=convnextv2_large FOLDS=3 MIXUP_ALPHA=0.2 bash scripts/run_conv_ft_pipeline.sh`）。
