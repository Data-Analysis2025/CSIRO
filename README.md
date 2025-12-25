# CSIRO Image2Biomass（ローカル学習 + Kaggle提出）

このリポジトリは、**ローカルで画像モデルを学習**し、重みを **Kaggle Dataset にアップロード**して、**Kaggle Notebook で submission.csv を生成・提出**するための最小構成テンプレートです。

---

## 1. セットアップ

```bash
pip install -r requirements.txt
```

---

## 2. データの取得

```bash
kaggle competitions download -c csiro-biomass -p data/csiro_biomass
cd data/csiro_biomass
unzip csiro-biomass.zip
```

データ構成：

```
data/csiro_biomass/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/        # 画像
└── test/         # 公開テスト画像
```

---

## 3. ローカル学習（自作ResNet）

以下は `custom_resnet34` を **ゼロから学習**する例です。

```bash
python scripts/train_image.py \
  --data-dir data/csiro_biomass \
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
  --data-dir data/csiro_biomass \
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

## ファイル説明

- `scripts/train_image.py`：ローカル学習
- `scripts/predict_image.py`：ローカル推論
- `src/models_image.py`：自作ResNet定義
- `notebooks/kaggle_image_inference.ipynb`：Kaggle推論用Notebook

---

## 備考

- 出力は **5つのターゲットを同時回帰**します  
  `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`
- `custom_resnet18` も使用可能です
