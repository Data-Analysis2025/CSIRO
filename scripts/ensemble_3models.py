"""
Ensemble script: Blends predictions from LightGBM, XGBoost, and CatBoost.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser()
    # 重み付けの設定
    parser.add_argument("--weights", type=float, nargs=3, default=[0.34, 0.33, 0.33], 
                        help="Weights for LGBM, XGB, CatBoost (must sum to 1)")
    args = parser.parse_args()
    
    # パス設定
    sub_dir = Path("submissions")
    output_path = sub_dir / "submission_ensemble.csv"
    
    # 読み込むファイル
    files = {
        "lgbm": sub_dir / "submission_lgbm.csv",
        "xgb":  sub_dir / "submission_xgb.csv",
        "cat":  sub_dir / "submission_cat.csv"
    }
    
    # ファイル読み込み
    dfs = {}
    for model, path in files.items():
        if not path.exists():
            print(f"Error: {path} not found.")
            return
        
        # sample_id順に並び替えて読み込む
        df = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
        dfs[model] = df
        print(f"Loaded {model}: {len(df)} rows")

    base_ids = dfs["lgbm"]["sample_id"]
    if not (dfs["xgb"]["sample_id"].equals(base_ids) and dfs["cat"]["sample_id"].equals(base_ids)):
        print("Error: Sample IDs do not match across submission files!")
        return

    # 重み付け平均の計算
    w_lgbm, w_xgb, w_cat = args.weights
    print(f"Ensembling with weights -> LGBM: {w_lgbm}, XGB: {w_xgb}, Cat: {w_cat}")

    # ベースとなるデータフレームを作成
    final_df = dfs["lgbm"][["sample_id"]].copy()
    
    # 加重平均
    final_df["target"] = (
        dfs["lgbm"]["target"] * w_lgbm +
        dfs["xgb"]["target"]  * w_xgb +
        dfs["cat"]["target"]  * w_cat
    )
    
    # 保存
    sub_dir.mkdir(exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Done! Ensemble submission saved to {output_path}")
    print(final_df.head())

if __name__ == "__main__":
    main()