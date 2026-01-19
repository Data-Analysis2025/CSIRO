"""
Extract DINOv2 embeddings from images.
Requires: pip install transformers torch pillow
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_dinov2():
    print("Loading DINOv2 model (facebook/dinov2-base)...")
    # 0.69 Notebook uses 'facebook/dinov2-base' usually
    model_name = "facebook/dinov2-base"
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
    except Exception as e:
        print(f"Error loading model from Hub: {e}")
        print("Please ensure you have internet access or the model is cached.")
        sys.exit(1)
        
    model.eval()
    return processor, model

def get_embeddings(image_paths, processor, model, batch_size=32):
    embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting DINOv2"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_indices = []
        
        for idx, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {p}: {e}")
        
        if not images:
            embeddings.append(np.zeros((len(batch_paths), 768))) # DINOv2-base is 768 dim
            continue
            
        try:
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                # DINOv2 outputs: last_hidden_state. We take the [CLS] token (index 0)
                # output shape: (batch, sequence, hidden) -> (batch, 0, 768)
                last_hidden_states = outputs.last_hidden_state
                emb = last_hidden_states[:, 0, :]
                
            # Normalize (L2) - crucial for embedding combining
            emb = emb / emb.norm(dim=-1, keepdim=True)
            
            # Fill batch result
            batch_emb = np.zeros((len(batch_paths), 768))
            batch_emb[valid_indices] = emb.cpu().numpy()
            embeddings.append(batch_emb)
            
        except Exception as e:
            print(f"Inference error: {e}")
            embeddings.append(np.zeros((len(images), 768)))
            
    return np.vstack(embeddings)

def main():
    parser = argparse.ArgumentParser()
    # configはパス取得のためだけに使う
    parser.add_argument("--config", type=str, default="configs/lgbm_siglip.yaml")
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    paths = config.get("paths")
    files = config.get("files")
    
    input_dir = Path(paths["input_dir"])
    
    # Process Train and Test
    for split in ["train", "test"]:
        print(f"\nProcessing {split} data...")
        csv_file = files[split]
        df = pd.read_csv(input_dir / csv_file)
        
        # Unique images only
        unique_df = df[['image_path']].drop_duplicates().reset_index(drop=True)
        image_paths = [input_dir / p for p in unique_df['image_path']]
        
        # Load Model
        if split == "train": # Load once
            processor, model = load_dinov2()
            
        # Extract
        emb_matrix = get_embeddings(image_paths, processor, model)
        
        # Save as DataFrame
        col_names = [f"emb_dino_{i}" for i in range(emb_matrix.shape[1])]
        emb_df = pd.DataFrame(emb_matrix, columns=col_names)
        emb_df["image_path"] = unique_df["image_path"]
        
        # Output filename
        out_name = f"{split}_embeddings_dino.csv"
        out_path = input_dir / out_name
        
        print(f"Saving to {out_path}...")
        emb_df.to_csv(out_path, index=False)
        print("Done.")

if __name__ == "__main__":
    main()