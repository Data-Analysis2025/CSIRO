import pandas as pd
from pathlib import Path

from path_utils import resolve_data_dir

DATA_DIR = resolve_data_dir("data/csiro_biomass", required_files=[])
train_path = DATA_DIR / "train_embeddings_combined.csv"
test_path = DATA_DIR / "test_embeddings_combined.csv"

def check_file(path):
    print(f"--- Checking {path} ---")
    if not Path(path).exists():
        print("File not found!")
        return

    # 最初の1行だけ読む
    df = pd.read_csv(path, nrows=1)
    
    total_cols = df.shape[1]
    cols = df.columns.tolist()
    
    # image_path などを除いた純粋な特徴量数を計算
    non_feature_cols = ["image_path", "sample_id", "fold", "target"]
    feature_cols = [c for c in cols if c not in non_feature_cols]
    
    print(f"Total Columns in CSV: {total_cols}")
    print(f"Feature Columns (Approx): {len(feature_cols)}")
    
    # 内訳の確認
    n_siglip = len([c for c in cols if "siglip" in c or ("emb" in c and "dino" not in c and "clip" not in c)])
    n_dino = len([c for c in cols if "dino" in c])
    # Semantic特徴量は特定の名前リストか、上記以外
    n_semantic = len(feature_cols) - n_siglip - n_dino
    
    print(f"  - SigLIP-like: {n_siglip} (Expect ~1152)")
    print(f"  - DINOv2-like: {n_dino}   (Expect ~768)")
    print(f"  - Semantic:    {n_semantic}     (Expect 13)")
    print("-" * 30)

check_file(train_path)
check_file(test_path)
