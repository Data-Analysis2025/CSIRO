"""Training script for CrossPVT_T2T_MambaDINO.

This script trains the CrossPVT model on CSIRO biomass images, saves
fold-wise checkpoints with cfg attached, and keeps directory structure
organized for later inference/upload.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data as data_utils
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

# Ensure project root is on sys.path so that `src` can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CFG, CrossPVT_T2T_MambaDINO, SimpleViTRegressor, update_cfg_from_checkpoint  # noqa: E402

DEFAULT_TARGET_WEIGHTS = (0.1, 0.1, 0.1, 0.2, 0.5)
SUMMARY_FILENAME = "model_summary.txt"


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
    wide = (
        df.pivot_table(index="image_path", columns="target_name", values="target")
        .reset_index()
        .sort_values("image_path")
        .reset_index(drop=True)
    )
    # Ensure consistent column order
    cols = list(CFG.ALL_TARGET_COLS)
    wide = wide[["image_path", *cols]]
    paths = wide["image_path"].to_list()
    targets = wide[cols].to_numpy(dtype=np.float32)
    return paths, targets


class BiomassDataset(data.Dataset):
    def __init__(
        self,
        image_paths: List[str],
        targets: np.ndarray,
        image_dir: Path,
        transform: A.Compose,
        dual_input: bool = True,
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.image_dir = image_dir
        self.transform = transform
        self.dual_input = dual_input

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
        target = torch.from_numpy(self.targets[idx])
        if not self.dual_input:
            image_t = self.transform(image=img)["image"]
            return image_t, target

        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        return left_t, right_t, target


def save_checkpoint(state: Dict, base_dir: Path, fold: int, tag: str = "best"):
    ckpt_dir = base_dir / f"fold_{fold}" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}_wr2.pt"
    torch.save(state, ckpt_path)
    return ckpt_path


def _rmse_from_sse(sse: float, count: int) -> float:
    if count == 0:
        return 0.0
    return float(math.sqrt(max(0.0, sse) / count))


def _weighted_r2(
    sse_res: float,
    weighted_sum_y: float,
    weighted_sum_y2: float,
    total_weight: float,
) -> float:
    if total_weight <= 0:
        return 0.0
    y_bar = weighted_sum_y / total_weight
    sstot = weighted_sum_y2 - total_weight * (y_bar ** 2)
    if sstot <= 1e-12:
        return 0.0
    return float(1.0 - (sse_res / sstot))


def save_metric_plot(
    train_losses: List[float],
    val_losses: List[float],
    train_rmses: List[float],
    val_rmses: List[float],
    train_r2: List[float],
    val_r2: List[float],
    out_path: Path,
):
    epochs = np.arange(1, len(train_losses) + 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("SmoothL1Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_rmses, label="train")
    axes[1].plot(epochs, val_rmses, label="valid")
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()

    axes[2].plot(epochs, train_r2, label="train")
    axes[2].plot(epochs, val_r2, label="valid")
    axes[2].set_title("Weighted R²")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R²")
    axes[2].legend()

    fig.suptitle("Training Progress")
    fig.savefig(out_path)
    plt.close(fig)


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, weight_sum: float):
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    weighted = loss * weights
    per_sample = weighted.sum(dim=1) / max(weight_sum, 1e-8)
    return per_sample.mean()


def assemble_predictions(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "pred" in out:
        return out["pred"]
    green = out["green"]
    gdm = out["gdm"]
    total = out["total"]
    clover = gdm - green
    dead = total - gdm
    return torch.cat([green, dead, clover, gdm, total], dim=1)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    target_weights: torch.Tensor,
    dual_input: bool,
    desc: str = "train",
):
    model.train()
    total_loss = 0.0
    seen = 0
    sq_err = 0.0
    elem_count = 0
    weights = target_weights.view(1, -1).to(device)
    weights_sum = float(target_weights.sum().item())
    weighted_y_sum = 0.0
    weighted_y2_sum = 0.0
    weighted_sse = 0.0
    total_weight = 0.0
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        if dual_input:
            left, right, target = batch
            left = left.to(device)
            right = right.to(device)
        else:
            left, target = batch
            left = left.to(device)
            right = None
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            out = model(x_left=left, x_right=right)
            pred = assemble_predictions(out)
            loss = _weighted_smooth_l1(pred, target, weights, weights_sum)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * left.size(0)
        mse_batch = F.mse_loss(pred, target, reduction="sum")
        sq_err += mse_batch.item()
        elem_count += target.numel()
        seen += left.size(0)
        diff = (pred - target) ** 2
        weighted_sse += (diff * weights).sum().item()
        weighted_y_sum += (target * weights).sum().item()
        weighted_y2_sum += (target.pow(2) * weights).sum().item()
        total_weight += weights_sum * left.size(0)
        avg_loss = total_loss / max(1, seen)
        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            rmse=f"{_rmse_from_sse(sq_err, elem_count):.4f}",
        )
    r2 = _weighted_r2(weighted_sse, weighted_y_sum, weighted_y2_sum, total_weight)
    return total_loss / len(loader.dataset), _rmse_from_sse(sq_err, elem_count), r2


def validate(model, loader, device, target_weights: torch.Tensor, dual_input: bool, desc: str = "valid"):
    model.eval()
    total_loss = 0.0
    sq_err = 0.0
    elem_count = 0
    seen = 0
    weights = target_weights.view(1, -1).to(device)
    weights_sum = float(target_weights.sum().item())
    weighted_y_sum = 0.0
    weighted_y2_sum = 0.0
    weighted_sse = 0.0
    total_weight = 0.0
    progress = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in progress:
            if dual_input:
                left, right, target = batch
                left = left.to(device)
                right = right.to(device)
            else:
                left, target = batch
                left = left.to(device)
                right = None
            target = target.to(device)
            out = model(x_left=left, x_right=right)
            pred = assemble_predictions(out)
            loss = _weighted_smooth_l1(pred, target, weights, weights_sum)
            total_loss += loss.item() * left.size(0)
            mse_batch = F.mse_loss(pred, target, reduction="sum")
            sq_err += mse_batch.item()
            elem_count += target.numel()
            seen += left.size(0)
            diff = (pred - target) ** 2
            weighted_sse += (diff * weights).sum().item()
            weighted_y_sum += (target * weights).sum().item()
            weighted_y2_sum += (target.pow(2) * weights).sum().item()
            total_weight += weights_sum * left.size(0)
            avg_loss = total_loss / max(1, seen)
            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                rmse=f"{_rmse_from_sse(sq_err, elem_count):.4f}",
            )
    r2 = _weighted_r2(weighted_sse, weighted_y_sum, weighted_y2_sum, total_weight)
    return total_loss / len(loader.dataset), _rmse_from_sse(sq_err, elem_count), r2


def main():
    parser = argparse.ArgumentParser(description="Train CrossPVT or ViT baselines")
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
    parser.add_argument(
        "--arch",
        type=str,
        default="crosspvt",
        choices=["crosspvt", "vit_simple"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Override CrossPVT dropout or vit_simple head dropout.",
    )
    parser.add_argument(
        "--hidden-ratio",
        type=float,
        default=0.3,
        help="CrossPVT MLP hidden ratio (ignored for vit_simple).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override input resolution (defaults to model's preferred size).",
    )
    parser.add_argument(
        "--vit-backbone",
        type=str,
        default="vit_small_patch14_dinov2",
        help="Backbone name for vit_simple architecture.",
    )
    parser.add_argument(
        "--target-weights",
        type=str,
        default="0.1,0.1,0.1,0.2,0.5",
        help="Comma-separated weights matching CFG.ALL_TARGET_COLS order.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop training if validation loss fails to improve for this many epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum increase in validation R² to qualify as an improvement.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    paths, targets = load_and_pivot(args.train_csv)
    if args.target_weights:
        weight_values = [float(x.strip()) for x in args.target_weights.split(",") if x.strip()]
    else:
        weight_values = list(DEFAULT_TARGET_WEIGHTS)
    if len(weight_values) != len(CFG.ALL_TARGET_COLS):
        raise ValueError(
            f"Expected {len(CFG.ALL_TARGET_COLS)} target weights, got {len(weight_values)}. "
            "Match CFG.ALL_TARGET_COLS order."
        )
    target_weights = torch.tensor(weight_values, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up run directory for organized checkpoints
    if args.run_name:
        run_name = args.run_name
    else:
        img_tag = args.img_size if args.img_size is not None else "auto"
        run_name = (
            f"{args.arch}_img{img_tag}_bs{args.batch_size}_lr{args.lr}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints under: {run_dir}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Update CFG if custom hyperparams provided
    dual_input = args.arch == "crosspvt"
    if dual_input:
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
        if args.arch == "crosspvt":
            temp_model = CrossPVT_T2T_MambaDINO(dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
        else:
            vit_dropout = args.dropout if args.dropout is not None else 0.1
            default_img = args.img_size or 384
            temp_model = SimpleViTRegressor(
                backbone=args.vit_backbone,
                img_size=default_img,
                dropout=vit_dropout,
            )
        img_size = args.img_size or getattr(temp_model, "input_res", 518)
        transform = build_transforms(img_size)

        train_ds = BiomassDataset(train_paths, train_tgts, args.image_dir, transform, dual_input=dual_input)
        val_ds = BiomassDataset(val_paths, val_tgts, args.image_dir, transform, dual_input=dual_input)

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
        summary_dir = run_dir / f"fold_{fold}"
        summary_path = summary_dir / SUMMARY_FILENAME
        summary_text = temp_model.summary() if hasattr(temp_model, "summary") else str(temp_model)
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print("=== Model Summary ===")
        print(summary_text)
        print(f"(Saved to {summary_path})")
        best_r2 = float("-inf")
        epochs_no_improve = 0
        train_losses: List[float] = []
        val_losses: List[float] = []
        train_rmses: List[float] = []
        val_rmses: List[float] = []
        train_r2_scores: List[float] = []
        val_r2_scores: List[float] = []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_rmse, train_r2 = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                target_weights,
                dual_input,
                desc=f"Fold {fold} Train (Epoch {epoch}/{args.epochs})",
            )
            val_loss, val_rmse, val_r2 = validate(
                model,
                val_loader,
                device,
                target_weights,
                dual_input,
                desc=f"Fold {fold} Valid (Epoch {epoch}/{args.epochs})",
            )
            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"train_loss={train_loss:.4f} train_rmse={train_rmse:.4f} train_r2={train_r2:.4f} | "
                f"val_loss={val_loss:.4f} val_rmse={val_rmse:.4f} val_r2={val_r2:.4f}"
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_rmses.append(train_rmse)
            val_rmses.append(val_rmse)
            train_r2_scores.append(train_r2)
            val_r2_scores.append(val_r2)

            state = {
                "model_state": model.state_dict(),
                "cfg": asdict(CFG),
                "epoch": epoch,
                "fold": fold,
                "val_loss": val_loss,
                "val_r2": val_r2,
            }
            ckpt_path = save_checkpoint(state, run_dir, fold, tag="last")
            if val_r2 > best_r2 + args.early_stop_min_delta:
                best_r2 = val_r2
                ckpt_path = save_checkpoint(state, run_dir, fold, tag="best")
                print(f"  Saved new best: {ckpt_path} (val_r2={val_r2:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                    print(
                        f"  Early stopping triggered at epoch {epoch} "
                        f"(no val_r2 improvement for {args.early_stop_patience} epochs)."
                    )
                    break

        metrics_path = run_dir / f"fold_{fold}" / "metrics.png"
        save_metric_plot(
            train_losses,
            val_losses,
            train_rmses,
            val_rmses,
            train_r2_scores,
            val_r2_scores,
            metrics_path,
        )
        print(f"  Saved metric plot: {metrics_path}")

        # free memory per fold
        del model
        torch.cuda.empty_cache()

    # Save meta info
    def _serialize_arg(value):
        if isinstance(value, Path):
            return str(value)
        return value

    meta = {
        "args": {k: _serialize_arg(v) for k, v in vars(args).items()},
        "cfg": asdict(CFG),
        "run_dir": str(run_dir),
    }
    meta_path = run_dir / "crosspvt_training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
