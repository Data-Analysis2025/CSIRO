import os
import sys
import glob
import torch
import gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from pathlib import Path

# プロジェクトルートの設定
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_utils import resolve_data_dir

DATA_DIR = resolve_data_dir(PROJECT_ROOT / "data" / "csiro_biomass")

# 設定
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings_for_split(image_paths, model, processor, model_type):

# ロード済みのモデルを使って、リストアップされた画像の埋め込みを作成する

    embeddings = []

    # TTA抽出
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"Extract {model_type}"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except:
                images.append(Image.new("RGB", (224, 224)))
        
        if not images: continue

        # TTA: 元画像 + 反転画像
        images_flipped = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        combined = images + images_flipped

        with torch.no_grad():
            if model_type == "clip":
                inputs = processor(images=combined, return_tensors="pt", padding=True).to(DEVICE)
                out = model.get_image_features(**inputs)
            else:
                inputs = processor(images=combined, return_tensors="pt").to(DEVICE)
                if model_type == "siglip":
                    out = model.get_image_features(**inputs)
                else:
                    outputs = model(**inputs)
                    out = outputs.last_hidden_state[:, 0, :]

        # 正規化 & 平均
        emb = out / out.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy()
        n = len(images)
        emb_avg = (emb[:n] + emb[n:]) / 2.0
        embeddings.append(emb_avg)

    return np.vstack(embeddings)

def process_one_model(model_name, model_type, train_paths, test_paths):
    """
    1つのモデルをロードし、TrainとTest両方の特徴量を抽出して返す。
    終わったらモデルをメモリから消去する。
    """
    print(f"\nLoading {model_name}...")

    # モデルロード
def _load_with_cache(component_cls, model_name: str, **kwargs):
    """Load HuggingFace components preferring the local cache to avoid downloads."""
    try:
        return component_cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except OSError:
        print(f"Cache miss for {model_name}; downloading once...")
        return component_cls.from_pretrained(model_name, **kwargs)

try:
    if model_type == "clip":
        processor = _load_with_cache(CLIPProcessor, model_name, use_safetensors=True)
        model = _load_with_cache(CLIPModel, model_name, use_safetensors=True).to(DEVICE).eval()
    else:
        processor = _load_with_cache(AutoImageProcessor, model_name, use_safetensors=True)
        model = _load_with_cache(AutoModel, model_name, use_safetensors=True).to(DEVICE).eval()
except ValueError as exc:
    raise RuntimeError(
        "Failed to load pretrained weights with safetensors. "
        "Please ensure the model repository provides safetensors files "
        "or upgrade torch >= 2.6."
    ) from exc

    print(f"--> Extracting Train ({len(train_paths)} images)...")
    emb_train = get_embeddings_for_split(train_paths, model, processor, model_type)

    print(f"--> Extracting Test ({len(test_paths)} images)...")
    emb_test = get_embeddings_for_split(test_paths, model, processor, model_type)

    # メモリ解放
    print("--> Cleaning up memory...")
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    return emb_train, emb_test

def main():
    print(f"Using device: {DEVICE}")

    # 1. 画像リストの取得
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_paths = [str(DATA_DIR / p) for p in train_df["image_path"]]
    test_paths = [str(DATA_DIR / p) for p in test_df["image_path"]]

    unique_train = sorted(list(set(train_paths)))
    unique_test = sorted(list(set(test_paths)))

    # ==========================================
    # モデル定義
    # ==========================================
    siglip_model = "google/siglip-so400m-patch14-384"
    dino_model = "facebook/dinov2-large"
    clip_model = "openai/clip-vit-large-patch14-336"

    # --- 1. SigLIP ---
    train_sig, test_sig = process_one_model(siglip_model, "siglip", unique_train, unique_test)

    # --- 2. DINOv2 Large ---
    train_dino, test_dino = process_one_model(dino_model, "dinov2", unique_train, unique_test)

    # --- 3. CLIP ---
    train_clip, test_clip = process_one_model(clip_model, "clip", unique_train, unique_test)

    # ==========================================
    # 結合と保存
    # ==========================================
    print("\nMerging features...")

    # Train保存
    train_combined = np.hstack([train_sig, train_dino, train_clip])
    df_train = pd.DataFrame(train_combined, columns=[f"emb_{i}" for i in range(train_combined.shape[1])])
    rel_train_paths = [str(Path(p).relative_to(DATA_DIR)) for p in unique_train]
    df_train.insert(0, "image_path", rel_train_paths)
    df_train.to_csv(DATA_DIR / "train_embeddings_large.csv", index=False)
    print("Saved train_embeddings_large.csv")

    # Test保存
    test_combined = np.hstack([test_sig, test_dino, test_clip])
    df_test = pd.DataFrame(test_combined, columns=[f"emb_{i}" for i in range(test_combined.shape[1])])
    rel_test_paths = [str(Path(p).relative_to(DATA_DIR)) for p in unique_test]
    df_test.insert(0, "image_path", rel_test_paths)
    df_test.to_csv(DATA_DIR / "test_embeddings_large.csv", index=False)
    print("Saved test_embeddings_large.csv")

if __name__ == "__main__":
    main()
