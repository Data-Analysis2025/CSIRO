import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings(image_paths, processor, model, batch_size=32):
    embeddings = []
    # バッチサイズごとに処理
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except:
                images.append(Image.new("RGB", (224, 224)))
        
        if not images: continue
        
        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
        emb = outputs / outputs.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
            
    return np.vstack(embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lgbm_embedding.yaml")
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    input_dir = Path(config.get("paths", "input_dir"))
    
    # モデル読み込み (ViT-Large)
    model_name = "openai/clip-vit-large-patch14-336"
    print(f"Loading {model_name}...")
    # 修正後
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        # use_safetensors=True を追加して、ブロックされているpickle形式を回避する
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # TrainとTestの両方を作成
    files = config.get("files")
    for split in ["train", "test"]:
        print(f"Processing {split}...")
        csv_file = files[split] # train.csv or test.csv
        df = pd.read_csv(input_dir / csv_file)
        
        unique_df = df[['image_path']].drop_duplicates().reset_index(drop=True)
        paths_list = [input_dir / p for p in unique_df['image_path']]
        
        emb = get_embeddings(paths_list, processor, model)
        
        # 保存
        out_df = pd.DataFrame(emb, columns=[f"emb_clip_{i}" for i in range(emb.shape[1])])
        out_df["image_path"] = unique_df["image_path"]
        
        out_path = input_dir / f"{split}_embeddings_clip.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()