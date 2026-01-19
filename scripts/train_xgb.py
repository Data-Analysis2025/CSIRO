"""
Train XGBoost using Pre-computed Embeddings + PCA/PLS/GMM.
"""
import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.features_embedding import SupervisedEmbeddingEngine

# Constants
TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
WEIGHTS = {
    'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5,
}

def competition_metric(y_true, y_pred):
    y_weighted = 0
    ss_res = 0
    ss_tot = 0
    for i, label in enumerate(TARGET_NAMES):
        w = WEIGHTS[label]
        y_weighted += y_true[:, i].mean() * w
    for i, label in enumerate(TARGET_NAMES):
        w = WEIGHTS[label]
        ss_res += ((y_true[:, i] - y_pred[:, i])**2).mean() * w
        ss_tot += ((y_true[:, i] - y_weighted)**2).mean() * w
    return 1 - (ss_res / ss_tot)

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
    model_cfg = config.get("model")
    feat_cfg = config.get("features")
    
    # Load Data
    input_dir = Path(paths["input_dir"])
    train_df = pd.read_csv(input_dir / files["train"])
    emb_df = pd.read_csv(input_dir / files["train_embeddings"])
    
    if 'target_name' in train_df.columns:
        train_wide = train_df.pivot_table(
            index="image_path", columns="target_name", values="target", aggfunc="mean"
        ).reset_index()
    else:
        train_wide = train_df
        
    full_df = pd.merge(train_wide, emb_df, on="image_path", how="inner")
    emb_cols = [c for c in full_df.columns if c.startswith("emb")]
    X = full_df[emb_cols].values
    y = full_df[TARGET_NAMES].values
    max_vals = np.array([target_cfg["max_values"][t] for t in TARGET_NAMES])
    
    kf = KFold(n_splits=config.get("cv", "n_splits"), shuffle=True, random_state=config.get("seed"))
    
    oof_preds = np.zeros_like(y)
    eng_params = feat_cfg.get("params", {})
    
    # Setup save directory
    models_dir = Path(paths["models_dir"]) / config.get("run_name")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training XGBoost. Saving models to {models_dir}")

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        y_train_scaled = y_train / max_vals
        
        # Fit Engine
        engine = SupervisedEmbeddingEngine(**eng_params, random_state=config.get("seed"))
        engine.fit(X_train, y_train_scaled)
        
        # Transform
        X_train_eng = engine.transform(X_train)
        X_valid_eng = engine.transform(X_valid)
        
        # Train & Save Models
        fold_models = []
        fold_preds = np.zeros_like(y_valid)
        
        for i in range(len(TARGET_NAMES)):
            model = xgb.XGBRegressor(**model_cfg["params"])
            model.fit(X_train_eng, y_train_scaled[:, i])
            fold_models.append(model)
            fold_preds[:, i] = model.predict(X_valid_eng) * max_vals[i]
            
        # Save Artifacts
        joblib.dump(engine, models_dir / f"engine_fold_{fold}.pkl")
        joblib.dump(fold_models, models_dir / f"models_fold_{fold}.pkl")
        
        fold_preds = np.maximum(0, fold_preds)
        oof_preds[valid_idx] = fold_preds
        print(f"Fold {fold} Score: {competition_metric(y_valid, fold_preds):.5f}")

    raw_score = competition_metric(y, oof_preds)
    oof_proc = post_process_biomass(pd.DataFrame(oof_preds, columns=TARGET_NAMES))
    proc_score = competition_metric(y, oof_proc.values)
    
    print(f"Overall OOF (Processed): {proc_score:.5f}")
    oof_proc.to_csv(Path(paths["processed_dir"]) / f"{config.get('run_name')}_oof.csv", index=False)

if __name__ == "__main__":
    main()