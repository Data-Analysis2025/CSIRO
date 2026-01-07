"""Training script for CrossPVT_T2T_MambaDINO.

This script trains the CrossPVT model on CSIRO biomass images, saves
fold-wise checkpoints with cfg attached, and keeps directory structure
organized for later inference/upload.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data as data_utils
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast

# Ensure project root is on sys.path so that `src` can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CFG, CrossPVT_T2T_MambaDINO, update_cfg_from_checkpoint  # noqa: E402


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def load_and_pivot(train_csv: Path) -> Tuple[List[str], np.ndarray]:
    import pandas as pd

    df = pd.read_csv(train_csv)
    # Pivot to wide: one row per image_path
    wide = df.pivot_table(index="image_path", columns="target_name", values="target")
    # Ensure consistent column order
    cols = list(CFG.ALL_TARGET_COLS)
    wide = wide[cols]
    paths = wide.index.to_list()
    targets = wide.to_numpy(dtype=np.float32)
    return paths, targets


class BiomassDataset(data.Dataset):
    def __init__(self, image_paths: List[str], targets: np.ndarray, image_dir: Path, transform: A.Compose):
        self.image_paths = image_paths
        self.targets = targets
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        rel_path_clean = rel_path.lstrip("./")
        base_name = self.image_dir.name
        prefix = f"{base_name}/"
        # If rel_path already includes the image_dir name (e.g., train/IDxxx.jpg) and image_dir points to train/,
        # drop the duplicate to avoid constructing train/train/IDxxx.jpg.
        if rel_path_clean.startswith(prefix):
            rel_path_clean = rel_path_clean[len(prefix) :]
        full_path = self.image_dir / rel_path_clean
        img = cv2.imread(str(full_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        target = torch.from_numpy(self.targets[idx])
        return left_t, right_t, target


def save_checkpoint(state: Dict, base_dir: Path, fold: int, tag: str = "best"):
    ckpt_dir = base_dir / f"fold_{fold}" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}_wr2.pt"
    torch.save(state, ckpt_path)
    return ckpt_path


def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    total_loss = 0.0
    for left, right, target in loader:
        left = left.to(device)
        right = right.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            out = model(x_left=left, x_right=right)
            # Pack predictions to 5 targets
            green = out["green"]
            gdm = out["gdm"]
            total_pred = out["total"]
            clover = gdm - green
            dead = total_pred - gdm
            pred = torch.cat([green, dead, clover, gdm, total_pred], dim=1)
            loss = criterion(pred, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * left.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for left, right, target in loader:
            left = left.to(device)
            right = right.to(device)
            target = target.to(device)
            out = model(x_left=left, x_right=right)
            green = out["green"]
            gdm = out["gdm"]
            total_pred = out["total"]
            clover = gdm - green
            dead = total_pred - gdm
            pred = torch.cat([green, dead, clover, gdm, total_pred], dim=1)
            loss = criterion(pred, target)
            total_loss += loss.item() * left.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train CrossPVT_T2T_MambaDINO")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to crosspvt_<YYYYMMDD_HHMMSS>",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden-ratio", type=float, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)

    paths, targets = load_and_pivot(args.train_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up run directory for organized checkpoints
    run_name = args.run_name or f"crosspvt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints under: {run_dir}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Update CFG if custom hyperparams provided
    cfg_override = {}
    if args.dropout is not None:
        cfg_override["dropout"] = args.dropout
    if args.hidden_ratio is not None:
        cfg_override["hidden_ratio"] = args.hidden_ratio
    update_cfg_from_checkpoint(cfg_override)

    for fold, (train_idx, val_idx) in enumerate(kf.split(paths)):
        print(f"\n===== Fold {fold} / {args.folds} =====")
        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        train_tgts = targets[train_idx]
        val_tgts = targets[val_idx]

        # Build temporary model to get input resolution
        temp_model = CrossPVT_T2T_MambaDINO(dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        img_size = getattr(temp_model, "input_res", 518)
        transform = build_transforms(img_size)

        train_ds = BiomassDataset(train_paths, train_tgts, args.image_dir, transform)
        val_ds = BiomassDataset(val_paths, val_tgts, args.image_dir, transform)

        train_loader = data_utils.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = data_utils.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model = temp_model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scaler = GradScaler(enabled=True)
        criterion = nn.SmoothL1Loss()

        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
            val_loss = validate(model, val_loader, device, criterion)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            state = {
                "model_state": model.state_dict(),
                "cfg": asdict(CFG),
                "epoch": epoch,
                "fold": fold,
                "val_loss": val_loss,
            }
            ckpt_path = save_checkpoint(state, run_dir, fold, tag="last")
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_path = save_checkpoint(state, run_dir, fold, tag="best")
                print(f"  Saved new best: {ckpt_path} (val_loss={val_loss:.4f})")

        # free memory per fold
        del model
        torch.cuda.empty_cache()

    # Save meta info
    meta = {
        "args": vars(args),
        "cfg": asdict(CFG),
        "run_dir": str(run_dir),
    }
    meta_path = run_dir / "crosspvt_training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
