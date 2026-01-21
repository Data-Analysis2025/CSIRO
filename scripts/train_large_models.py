import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "csiro_biomass"
MODELS_DIR = PROJECT_ROOT / "models" / "large_ensemble"

TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
MAX_VALS = np.array([71.7865, 83.8407, 157.9836, 185.70, 157.9836])

# ==========================================
# スコア計算用関数
# ==========================================
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
        # 形状ズレ防止のためreshape
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ss_res += ((yt - yp)**2).mean() * w
        ss_tot += ((yt - y_weighted)**2).mean() * w
    return 1 - (ss_res / (ss_tot + 1e-6))

def post_process_biomass(y_pred_df):
    ordered_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    # カラムが足りない場合のガード
    if not all(col in y_pred_df.columns for col in ordered_cols):
        return y_pred_df
        
    Y = y_pred_df[ordered_cols].values.T
    C = np.array([[1, 1, 0, -1,  0], [0, 0, 1,  1, -1]])
    P = np.eye(5) - C.T @ np.linalg.inv(C @ C.T) @ C
    Y_reconciled = (P @ Y).T.clip(min=0)
    df_out = y_pred_df.copy()
    df_out[ordered_cols] = Y_reconciled
    return df_out

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading Large Embeddings...")
    df_feat = pd.read_csv(DATA_DIR / "train_embeddings_large.csv")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    
    if "Dry_Green_g" not in df_train.columns:
        df_train = df_train.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="mean").reset_index()
        
    df = pd.merge(df_train, df_feat, on="image_path", how="inner")
    
    # 特徴量カラム
    exclude = ["image_path", "sample_id", "Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"] + TARGET_NAMES
    feat_cols = [c for c in df.columns if c not in exclude]
    print(f"Features: {len(feat_cols)} dim (SigLIP+DINO_L+CLIP)")

    X = df[feat_cols].values
    y = df[TARGET_NAMES].values
    
    # 正規化
    y_norm = y / MAX_VALS
    
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_bin = (y > 0).astype(int)

    # モデル定義
    models_def = {
        "lgbm": LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, colsample_bytree=0.8, subsample=0.8, random_state=42, verbose=-1),
        "xgb": XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, colsample_bytree=0.8, subsample=0.8, tree_method='hist', device='cpu', random_state=42),
        "cat": CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, task_type="GPU", random_state=42, verbose=0)
    }

    # 各モデル学習ループ
    for name, base_model in models_def.items():
        print(f"\n=== Training {name.upper()} (Large) ===")
        save_dir = MODELS_DIR / name
        save_dir.mkdir(exist_ok=True)
        
        # OOF保存用
        oof_df = df[["image_path"] + TARGET_NAMES].copy()
        pred_cols = [f"pred_{c}" for c in TARGET_NAMES]
        for c in pred_cols: oof_df[c] = 0.0
        
        # Foldループ
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y_bin)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_norm[train_idx], y_norm[val_idx]
            
            # Scaler
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)
            
            # Train
            model = MultiOutputRegressor(base_model)
            model.fit(X_train_sc, y_train)
            
            # Predict (OOF)
            val_preds = model.predict(X_val_sc)
            val_preds_restored = val_preds * MAX_VALS
            
            # OOF格納
            oof_df.loc[val_idx, pred_cols] = val_preds_restored
            
            # Save Model
            joblib.dump(scaler, save_dir / f"engine_fold_{fold}.pkl")
            joblib.dump(model.estimators_, save_dir / f"models_fold_{fold}.pkl")
            
            # FoldごとのOOF csv保存
            fold_oof = oof_df.iloc[val_idx].copy()
            fold_oof.to_csv(save_dir / f"oof_fold_{fold}.csv", index=False)
            
            # スコア計算
            y_true_fold = y[val_idx]
            score = competition_metric(y_true_fold, val_preds_restored)
            print(f"Fold {fold} Score: {score:.5f}")

        # モデルごとの全体スコア計算
        y_true_all = df[TARGET_NAMES].values
        y_pred_all = oof_df[pred_cols].values
        
        # Raw Score
        raw_score = competition_metric(y_true_all, y_pred_all)
        
        # Processed Score
        temp_pred_df = pd.DataFrame(y_pred_all, columns=TARGET_NAMES)
        processed_df = post_process_biomass(temp_pred_df)
        proc_score = competition_metric(y_true_all, processed_df.values)
        
        print("-" * 30)
        print(f"Overall OOF (Raw):       {raw_score:.5f}")
        print(f"Overall OOF (Processed): {proc_score:.5f}")
        print("-" * 30)

    print("\n✅ All Training Completed!")

if __name__ == "__main__":
    main()