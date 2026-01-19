"""
Merge features: SigLIP + DINOv2 + CLIP (No Semantic)
Total Dims: 1152 + 768 + 768 = 2688
"""
import sys
from pathlib import Path
import pandas as pd

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

def main():
    # Config読み込み
    config_path = "configs/cat_embedding.yaml"
    config = Config.from_yaml(config_path)
    
    input_dir = Path(config.get("paths", "input_dir"))
    
    targets = [
        ("train", "train_embeddings_siglip.csv", "train_embeddings_dino.csv", "train_embeddings_clip.csv"),
        ("test",  "test_embeddings_siglip.csv",  "test_embeddings_dino.csv",  "test_embeddings_clip.csv")
    ]
    
    for split, f_sig, f_dino, f_clip in targets:
        print(f"--- Merging {split} ---")
        
        # 1. SigLIP 読み込み
        p_sig = input_dir / f_sig
        if not p_sig.exists():
            print(f"Error: {p_sig} not found.")
            continue
        df = pd.read_csv(p_sig)
        
        # 2. DINOv2 結合
        p_dino = input_dir / f_dino
        if p_dino.exists():
            df_dino = pd.read_csv(p_dino)
            df = pd.merge(df, df_dino, on="image_path", how="inner")
        else:
            print(f"Warning: {p_dino} not found.")

        # 3. CLIP 結合
        p_clip = input_dir / f_clip
        if p_clip.exists():
            df_clip = pd.read_csv(p_clip)
            df = pd.merge(df, df_clip, on="image_path", how="inner")
        else:
            print(f"Warning: {p_clip} not found.")
            
        # 次元数確認
        feat_dim = df.shape[1] - 1 # image_pathの分を引く
        print(f"Final Shape: {df.shape} (Features: {feat_dim})")
        
        if feat_dim == 2688:
            print("✅ OK: 2688 dims (SigLIP+DINO+CLIP)")
        else:
            print(f"⚠️ Warning: Dimension is {feat_dim} (Expected 2688)")

        # 保存
        out_path = input_dir / f"{split}_embeddings_combined.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()