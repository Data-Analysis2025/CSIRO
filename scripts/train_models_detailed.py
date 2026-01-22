import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV

# GBDT Libraries
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "csiro_biomass"
MODELS_DIR = PROJECT_ROOT / "models" / "detailed_ensemble"

TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
MAX_VALS = np.array([71.7865, 83.8407, 157.9836, 185.70, 157.9836])

# 評価関数
WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}

def competition_metric(y_true, y_pred):
    y_weighted = 0
    ss_res = 0
    ss_tot = 0
    for i, label in enumerate(TARGET_NAMES):
        w = WEIGHTS[label]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        y_weighted += yt.mean() * w
        ss_res += ((yt - yp)**2).mean() * w
        ss_tot += ((yt - y_weighted)**2).mean() * w
    return 1 - (ss_res / (ss_tot + 1e-6))

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. データ読み込み
    print("Loading Features (v4)...")
    emb_path = DATA_DIR / "train_embeddings_large_v4.csv"
    if not emb_path.exists():
        print(f"⚠️ {emb_path} not found. Please run extract_features_v4.py first.")
        return

    df_feat = pd.read_csv(emb_path)
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    
    if "Dry_Green_g" not in df_train.columns:
        df_train = df_train.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="mean").reset_index()
        
    df = pd.merge(df_train, df_feat, on="image_path", how="inner")
    
    exclude = ["image_path", "sample_id", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"] + TARGET_NAMES
    feat_cols = [c for c in df.columns if c not in exclude]
    print(f"Features: {len(feat_cols)} dim")

    X = df[feat_cols].values
    y = df[TARGET_NAMES].values
    y_norm = y / MAX_VALS
    
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_bin = (y > 0).astype(int)

    # 保存用DataFrame
    oof_total = df[["image_path"] + TARGET_NAMES].copy()
    
    # ==========================================
    # 1. Ridge
    # ==========================================
    print(f"\n{'='*20} Training RIDGE (Auto-Tuned) {'='*20}")
    ridge_preds = np.zeros_like(y)
    
    save_dir = MODELS_DIR / "ridge"
    save_dir.mkdir(exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y_bin)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_norm[train_idx], y_norm[val_idx]
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        model.fit(X_train_sc, y_train)
        
        val_p = model.predict(X_val_sc) * MAX_VALS
        ridge_preds[val_idx] = val_p
        
        joblib.dump(scaler, save_dir / f"engine_fold_{fold}.pkl")
        joblib.dump(model, save_dir / f"models_fold_{fold}.pkl")
        
        score = competition_metric(y[val_idx], val_p)
        print(f"Ridge Fold {fold}: Score {score:.5f} (Best Alpha: {model.alpha_})")

    for i, c in enumerate(TARGET_NAMES): oof_total[f"pred_ridge_{c}"] = ridge_preds[:, i]
    
    # ==========================================
    # 2. GBDT Models
    # ==========================================
    gbdt_models = {
        "lgbm": LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1),
        "xgb": XGBRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=6, 
            colsample_bytree=0.8, subsample=0.8, 
            tree_method='hist', device='cpu', 
            early_stopping_rounds=50,
            random_state=42, n_jobs=-1
        ),
        "cat": CatBoostRegressor(
            iterations=2000, learning_rate=0.03, depth=6, 
            task_type="GPU", 
            early_stopping_rounds=50,
            random_state=42, verbose=0
        )
    }

    for name, base_model in gbdt_models.items():
        print(f"\n{'='*20} Training {name.upper()} (Detailed Log) {'='*20}")
        save_dir = MODELS_DIR / name
        save_dir.mkdir(exist_ok=True)
        
        model_preds = np.zeros_like(y)
        
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y_bin)):
            print(f"\n--- Fold {fold} ---")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_raw, y_val_raw = y_norm[train_idx], y_norm[val_idx]
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)
            joblib.dump(scaler, save_dir / f"engine_fold_{fold}.pkl")
            
            fold_estimators = []
            
            for i, target_name in enumerate(TARGET_NAMES):
                print(f"  [Target: {target_name}]")
                yt_train = y_train_raw[:, i]
                yt_val = y_val_raw[:, i]
                
                # Clone model
                if name == "lgbm":
                    model = LGBMRegressor(**base_model.get_params())
                    callbacks = [
                        log_evaluation(period=50), 
                        early_stopping(stopping_rounds=50, verbose=True)
                    ]
                    model.fit(
                        X_train_sc, yt_train,
                        eval_set=[(X_train_sc, yt_train), (X_val_sc, yt_val)],
                        eval_names=['Train', 'Valid'],
                        eval_metric='rmse',
                        callbacks=callbacks
                    )
                
                elif name == "xgb":
                    model = XGBRegressor(**base_model.get_params())
                    model.fit(
                        X_train_sc, yt_train,
                        eval_set=[(X_train_sc, yt_train), (X_val_sc, yt_val)],
                        verbose=50 
                    )

                elif name == "cat":
                    model = CatBoostRegressor(**base_model.get_params())
                    model.fit(
                        X_train_sc, yt_train,
                        eval_set=(X_val_sc, yt_val),
                        verbose=50
                    )
                
                fold_estimators.append(model)
                
                # Best Iterationでの予測
                val_p = model.predict(X_val_sc) * MAX_VALS[i]
                model_preds[val_idx, i] = val_p
                
                r2 = r2_score(y[val_idx, i], val_p)
                rmse = np.sqrt(mean_squared_error(y[val_idx, i], val_p))
                print(f"    -> Best RMSE: {rmse:.4f}, R2: {r2:.4f}")

            joblib.dump(fold_estimators, save_dir / f"models_fold_{fold}.pkl")
            
        for i, c in enumerate(TARGET_NAMES): oof_total[f"pred_{name}_{c}"] = model_preds[:, i]
        
        score = competition_metric(y, model_preds)
        print(f"\n✅ {name.upper()} Overall Score: {score:.5f}")

    oof_total.to_csv(MODELS_DIR / "oof_detailed_all.csv", index=False)
    print("\nTraining Finished. All models saved.")

if __name__ == "__main__":
    main()