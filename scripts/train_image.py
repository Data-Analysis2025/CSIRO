"""Train a lightweight image baseline for CSIRO Biomass."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_utils import resolve_data_dir
from src.model_factory import create_model_from_name, detect_model_source
DEFAULT_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


class AlbumentationsWrapper:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img)
        augmented = self.transform(image=arr)
        return augmented["image"]


class BiomassImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, targets: List[str], tfms):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.targets = targets
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.tfms(image)
        target = row[self.targets].values.astype("float32")
        return image, torch.tensor(target)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rand_bbox(height: int, width: int, lam: float):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return y1, x1, y2, x2


def apply_mixup_cutmix(
    imgs: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if imgs.size(0) == 1:
        return imgs, targets
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    shuffled_imgs = imgs[perm]
    shuffled_targets = targets[perm]

    if cutmix_alpha > 0.0 and (mixup_alpha <= 0.0 or random.random() < 0.5):
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        y1, x1, y2, x2 = rand_bbox(imgs.size(2), imgs.size(3), lam)
        imgs[:, :, y1:y2, x1:x2] = shuffled_imgs[:, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (imgs.size(-1) * imgs.size(-2)))
        targets = targets * lam + shuffled_targets * (1 - lam)
    elif mixup_alpha > 0.0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        imgs = lam * imgs + (1 - lam) * shuffled_imgs
        targets = lam * targets + (1 - lam) * shuffled_targets
    return imgs, targets


def build_transforms(image_size: int, use_alb: bool):
    normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    if use_alb:
        train_tf = AlbumentationsWrapper(
            A.Compose([
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.85, 1.15), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.ColorJitter(0.1, 0.1, 0.1, 0.02, p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    min_holes=1,
                    max_height=int(0.1 * image_size),
                    max_width=int(0.1 * image_size),
                    min_height=1,
                    min_width=1,
                    fill_value=0,
                    mask_fill_value=None,
                    p=0.5,
                ),
                A.Normalize(mean=normalize["mean"], std=normalize["std"]),
                ToTensorV2(),
            ])
        )
        valid_tf = AlbumentationsWrapper(
            A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=normalize["mean"], std=normalize["std"]),
                ToTensorV2(),
            ])
        )
    else:
        norm = transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            norm,
        ])
        valid_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm,
        ])
    return train_tf, valid_tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    parser.add_argument("--image-size", type=int, default=576)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1, help="Used only when --folds=1.")
    parser.add_argument("--folds", type=int, default=1, help="Number of KFold splits. Use >1 for cross-validation.")
    parser.add_argument("--targets", type=str, default="")
    parser.add_argument("--use-albumentations", action="store_true", help="Use Albumentations augmentation pipeline.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup alpha (0 to disable).")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0, help="CutMix alpha (0 to disable).")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="none")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    train_csv = data_dir / "train.csv"

    train_df = pd.read_csv(train_csv)

    targets = DEFAULT_TARGETS
    if args.targets.strip():
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        unique_targets = sorted(train_df["target_name"].dropna().unique().tolist())
        if set(DEFAULT_TARGETS).issubset(set(unique_targets)):
            targets = DEFAULT_TARGETS
        else:
            targets = unique_targets

    if not targets:
        raise ValueError("No targets resolved. Check train.csv target_name values or pass --targets.")

    wide = train_df.pivot_table(
        index="image_path",
        columns="target_name",
        values="target",
        aggfunc="mean",
    ).reset_index()
    for t in targets:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[["image_path"] + targets]
    wide.fillna(0.0, inplace=True)

    train_tf, valid_tf = build_transforms(args.image_size, args.use_albumentations)

    rng = np.random.RandomState(args.seed)
    if args.folds > 1:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(wide))
    else:
        val_size = max(1, int(len(wide) * args.val_split))
        perm = rng.permutation(len(wide))
        val_idx = perm[:val_size]
        tr_idx = perm[val_size:]
        splits = [(tr_idx, val_idx)]

    model_source = detect_model_source(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_fold(fold_id: int, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
        tr_df = wide.iloc[train_idx].copy()
        va_df = wide.iloc[val_idx].copy()

        train_loader = DataLoader(
            BiomassImageDataset(tr_df, data_dir, targets, train_tf),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        valid_loader = DataLoader(
            BiomassImageDataset(va_df, data_dir, targets, valid_tf),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = create_model_from_name(
            args.model_name,
            num_outputs=len(targets),
            pretrained=args.pretrained,
            source=model_source,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = None
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        criterion = nn.SmoothL1Loss()
        best_rmse = float("inf")
        best_stats = {}

        fold_dir = out_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = fold_dir / "best.pt"

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            for imgs, targets_tensor in tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch} train", leave=False):
                imgs = imgs.to(device)
                targets_tensor = targets_tensor.to(device)
                if args.mixup_alpha > 0.0 or args.cutmix_alpha > 0.0:
                    imgs, targets_tensor = apply_mixup_cutmix(imgs, targets_tensor, args.mixup_alpha, args.cutmix_alpha)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, targets_tensor)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            valid_losses = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for imgs, targets_tensor in tqdm(valid_loader, desc=f"Fold {fold_id} Epoch {epoch} valid", leave=False):
                    imgs = imgs.to(device)
                    targets_tensor = targets_tensor.to(device)
                    preds = model(imgs)
                    loss = criterion(preds, targets_tensor)
                    valid_losses.append(loss.item())
                    val_preds.append(preds.cpu())
                    val_targets.append(targets_tensor.cpu())

            tr_loss = float(np.mean(train_losses)) if train_losses else 0.0
            va_loss = float(np.mean(valid_losses)) if valid_losses else 0.0
            y_true = torch.cat(val_targets).numpy()
            y_pred = torch.cat(val_preds).numpy()
            val_rmse = mean_squared_error(y_true, y_pred)
            val_rmse = float(val_rmse) ** 0.5
            val_r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
            print(f"Fold {fold_id} Epoch {epoch}/{args.epochs} | train {tr_loss:.4f} | valid {va_loss:.4f} | rmse {val_rmse:.4f} | r2 {val_r2:.4f}")

            if scheduler is not None:
                scheduler.step()

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_stats = {"rmse": val_rmse, "r2": val_r2, "loss": va_loss, "epoch": epoch}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_name": args.model_name,
                        "image_size": args.image_size,
                        "targets_order": targets,
                        "normalize": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                        "custom_model": model_source == "custom",
                        "model_source": model_source,
                        "fold": fold_id,
                        "best_metrics": best_stats,
                    },
                    ckpt_path,
                )
                meta_path = fold_dir / "metrics.json"
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(best_stats, f, indent=2)
                print(f"[Fold {fold_id}] Saved checkpoint to {ckpt_path}")

        return best_stats

    summary = []
    for fold_id, (tr_idx, val_idx) in enumerate(splits):
        stats = run_fold(fold_id, np.array(tr_idx), np.array(val_idx))
        summary.append({"fold": fold_id, **stats})

    if summary:
        print("\n=== Fold Summary ===")
        for item in summary:
            print(f"Fold {item['fold']} | rmse {item.get('rmse', float('inf')):.4f} | r2 {item.get('r2', float('-inf')):.4f}")

    if args.folds == 1:
        legacy_ckpt = Path(args.out_dir) / f"image_{args.model_name}.pt"
        src_ckpt = out_dir / "fold_0" / "best.pt"
        if src_ckpt.exists():
            shutil.copyfile(src_ckpt, legacy_ckpt)
            print(f"Copied fold_0 checkpoint to {legacy_ckpt} for backward compatibility.")


if __name__ == "__main__":
    main()
