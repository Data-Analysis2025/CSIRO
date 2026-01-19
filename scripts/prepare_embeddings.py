"""
Extract SigLIP embeddings from images and save as a new CSV/Parquet.
Usage: python scripts/prepare_embeddings.py --data-dir data/csiro_biomass
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def split_image(image, patch_size=520, overlap=16):
    """Splits an image into patches."""
    h, w, c = image.shape
    stride = patch_size - overlap
    patches = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1, y2, x2 = y, x, y + patch_size, x + patch_size
            patch = image[y1:y2, x1:x2, :]
            # Pad if needed
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0,pad_h), (0,pad_w), (0,0)), mode='reflect')
            patches.append(patch)
    return patches

def get_model(model_name="google/siglip-so400m-patch14-384", device='cpu'):
    print(f"Loading model: {model_name}")
    try:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load from Hub. Trying local path if Kaggle... {e}")
        # Local fallback logic (omitted for standard usage, assuming internet or cached)
        raise e
        
    return model.eval().to(device), processor

def compute_embeddings(df, root_dir, model_name, patch_size=520, batch_size=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = get_model(model_name, device)
    
    embeddings_list = []
    image_paths = []
    
    print(f"Extracting embeddings for {len(df)} images...")
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = row['image_path']
        full_path = root_dir / rel_path
        
        # Read image
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Could not read {full_path}. Skipping/Zeroing.")
            embeddings_list.append(np.zeros(model.config.vision_config.hidden_size)) # Placeholder
            image_paths.append(rel_path)
            continue

        patches = split_image(img, patch_size=patch_size)
        pil_images = [Image.fromarray(p).convert("RGB") for p in patches]
        
        # Batch processing patches
        patch_embeds = []
        # Process patches in small batches to avoid OOM
        for k in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[k : k + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                # SigLIP specific: get_image_features
                if hasattr(model, "get_image_features"):
                    features = model.get_image_features(**inputs)
                else:
                    # Fallback for generic ViT
                    features = model(**inputs).pooler_output
            patch_embeds.append(features.cpu().numpy())
            
        if patch_embeds:
            full_embed = np.concatenate(patch_embeds, axis=0)
            mean_embed = full_embed.mean(axis=0) # Average over patches
            embeddings_list.append(mean_embed)
        else:
            embeddings_list.append(np.zeros(model.config.vision_config.hidden_size))
            
        image_paths.append(rel_path)

    embeddings_matrix = np.stack(embeddings_list, axis=0)
    
    # Create DataFrame
    emb_cols = [f"emb{i}" for i in range(embeddings_matrix.shape[1])]
    emb_df = pd.DataFrame(embeddings_matrix, columns=emb_cols)
    emb_df['image_path'] = image_paths
    
    return emb_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/csiro_biomass")
    parser.add_argument("--model-name", type=str, default="google/siglip-so400m-patch14-384")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found.")

    # Process Train
    print("Processing Train Data...")
    df_train = pd.read_csv(train_csv)
    # Pivot if necessary (ensure one row per image)
    if 'target_name' in df_train.columns:
        df_train_unique = df_train[['image_path']].drop_duplicates().reset_index(drop=True)
    else:
        df_train_unique = df_train
        
    emb_train = compute_embeddings(df_train_unique, data_dir, args.model_name)
    emb_train_path = data_dir / "train_embeddings_siglip.csv"
    emb_train.to_csv(emb_train_path, index=False)
    print(f"Saved train embeddings to {emb_train_path}")

    # Process Test
    if test_csv.exists():
        print("Processing Test Data...")
        df_test = pd.read_csv(test_csv)
        df_test_unique = df_test[['image_path']].drop_duplicates().reset_index(drop=True)
        emb_test = compute_embeddings(df_test_unique, data_dir, args.model_name)
        emb_test_path = data_dir / "test_embeddings_siglip.csv"
        emb_test.to_csv(emb_test_path, index=False)
        print(f"Saved test embeddings to {emb_test_path}")

if __name__ == "__main__":
    main()