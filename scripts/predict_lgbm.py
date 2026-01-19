"""
Inference script for LightGBM with Embeddings.
Loads models and creates submission.csv.
"""
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.features_embedding import SupervisedEmbeddingEngine # Needed for unpickling

TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

def post_process_biomass(y_pred_df):
    ordered_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    Y = y_pred_df[ordered_cols].values.T
    C = np.array([[1, 1, 0, -1,  0], [0, 0, 1,  1, -1]])
    P = np.eye(5) - C.T @ np.linalg.inv(C @ C.T) @ C
    Y_reconciled = (P @ Y).T.clip(min=0)
    df_out = y_pred_df.copy()
    df_out[ordered_cols] = Y_reconciled
    return df_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    paths = config.get("paths")
    files = config.get("files")
    target_cfg = config.get("target")
    
    # Load Test Data & Embeddings
    input_dir = Path(paths["input_dir"])
    test_df = pd.read_csv(input_dir / files["test"])
    emb_df = pd.read_csv(input_dir / files["test_embeddings"])
    
    # Merge (careful with duplicates in test.csv)
    test_unique = test_df[['image_path']].drop_duplicates().reset_index(drop=True)
    full_df = pd.merge(test_unique, emb_df, on="image_path", how="left")
    
    emb_cols = [c for c in full_df.columns if c.startswith("emb")]
    X_test = full_df[emb_cols].values
    max_vals = np.array([target_cfg["max_values"][t] for t in TARGET_NAMES])
    
    # Load Models & Predict
    models_dir = Path(paths["models_dir"]) / config.get("run_name")
    n_splits = config.get("cv", "n_splits")
    
    test_preds_accum = np.zeros((len(X_test), 5))
    
    print(f"Predicting with {n_splits} folds...")
    
    for fold in range(n_splits):
        # Load Engine and Models
        engine = joblib.load(models_dir / f"engine_fold_{fold}.pkl")
        models = joblib.load(models_dir / f"models_fold_{fold}.pkl")
        
        # Transform Test Data
        X_test_eng = engine.transform(X_test)
        
        # Predict for each target
        fold_preds = []
        for i, model in enumerate(models):
            # Predict and scale back
            p = model.predict(X_test_eng) * max_vals[i]
            fold_preds.append(p)
            
        test_preds_accum += np.column_stack(fold_preds)
        
    # Average folds
    test_preds_avg = test_preds_accum / n_splits
    test_preds_avg = np.maximum(0, test_preds_avg)
    
    # Format to DataFrame
    pred_df_wide = pd.DataFrame(test_preds_avg, columns=TARGET_NAMES)
    pred_df_wide["image_path"] = full_df["image_path"]
    
    # Post Process
    if config.get("inference", "post_process"):
        pred_df_wide = post_process_biomass(pred_df_wide)
        
    # Create Submission (Melt to long format)
    # Submission format: sample_id, target
    # sample_id = image_name + "__" + target_name (usually)
    # But checking sample_submission structure is safer.
    
    # Let's use the provided test.csv to map back
    submission = test_df[["sample_id", "image_path", "target_name"]].copy()
    
    # Melt our predictions
    pred_long = pred_df_wide.melt(
        id_vars=["image_path"], 
        value_vars=TARGET_NAMES, 
        var_name="target_name", 
        value_name="pred_target"
    )
    
    # Merge
    final_sub = pd.merge(submission, pred_long, on=["image_path", "target_name"], how="left")
    final_sub["target"] = final_sub["pred_target"].fillna(0)
    final_sub = final_sub[["sample_id", "target"]]
    
    # Save
    out_path = Path(paths["submissions_dir"]) / config.get("inference", "output_filename")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_sub.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

if __name__ == "__main__":
    main()