import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "csiro_biomass"

# 設定
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings_v4(image_paths, model, processor, model_type):
    embeddings = []
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"Extract {model_type} (No TTA, CLS+AVG)"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except:
                images.append(Image.new("RGB", (224, 224)))
        
        if not images: continue

        with torch.no_grad():
            if model_type == "clip":
                inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
                out = model.get_image_features(**inputs)
            else:
                inputs = processor(images=images, return_tensors="pt").to(DEVICE)
                if model_type == "siglip":
                    out = model.get_image_features(**inputs)
                else:
                    # DINOv2: CLSトークンと平均プーリングを結合する
                    outputs = model(**inputs)
                    last_hidden = outputs.last_hidden_state # [Batch, Seq, Dim]
                    
                    # 1. CLS Token (先頭)
                    cls_token = last_hidden[:, 0, :]
                    
                    # 2. Average Pooling (残り全ての平均)
                    avg_pool = last_hidden[:, 1:, :].mean(dim=1)
                    
                    # 3. 結合 (Dim * 2 になる)
                    out = torch.cat([cls_token, avg_pool], dim=1)

        # 正規化
        emb = out / out.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)

def process_one_model(model_name, model_type, train_paths, test_paths):
    print(f"\nLoading {model_name}...")
    
    if model_type == "clip":
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    
    print(f"--> Extracting Train...")
    emb_train = get_embeddings_v4(train_paths, model, processor, model_type)
    
    print(f"--> Extracting Test...")
    emb_test = get_embeddings_v4(test_paths, model, processor, model_type)
    
    del model, processor
    torch.cuda.empty_cache()
    return emb_train, emb_test

def main():
    print(f"Using device: {DEVICE}")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    train_paths = [str(DATA_DIR / p) for p in train_df["image_path"]]
    test_paths = [str(DATA_DIR / p) for p in test_df["image_path"]]
    unique_train = sorted(list(set(train_paths)))
    unique_test = sorted(list(set(test_paths)))
    
    # モデル定義
    siglip_model = "google/siglip-so400m-patch14-384"
    dino_model = "facebook/dinov2-large"
    clip_model = "openai/clip-vit-large-patch14-336"

    # 1. SigLIP (1152)
    train_sig, test_sig = process_one_model(siglip_model, "siglip", unique_train, unique_test)
    
    # 2. DINOv2 (2048)
    train_dino, test_dino = process_one_model(dino_model, "dinov2", unique_train, unique_test)
    
    # 3. CLIP (768)
    train_clip, test_clip = process_one_model(clip_model, "clip", unique_train, unique_test)

    print("\nMerging features...")
    # 1152 + 2048 + 768 = 3968次元
    train_combined = np.hstack([train_sig, train_dino, train_clip])
    df_train = pd.DataFrame(train_combined, columns=[f"emb_{i}" for i in range(train_combined.shape[1])])
    rel_train_paths = [str(Path(p).relative_to(DATA_DIR)) for p in unique_train]
    df_train.insert(0, "image_path", rel_train_paths)
    df_train.to_csv(DATA_DIR / "train_embeddings_large_v4.csv", index=False)
    
    test_combined = np.hstack([test_sig, test_dino, test_clip])
    df_test = pd.DataFrame(test_combined, columns=[f"emb_{i}" for i in range(test_combined.shape[1])])
    rel_test_paths = [str(Path(p).relative_to(DATA_DIR)) for p in unique_test]
    df_test.insert(0, "image_path", rel_test_paths)
    df_test.to_csv(DATA_DIR / "test_embeddings_large_v4.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()