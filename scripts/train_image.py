"""Train a lightweight image baseline for CSIRO Biomass."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models_image import create_model as create_custom_model, is_custom_model
DEFAULT_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


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


def build_model(model_name: str, num_outputs: int, pretrained: bool) -> nn.Module:
    if is_custom_model(model_name):
        if pretrained:
            raise ValueError("Custom models do not support --pretrained.")
        return create_custom_model(model_name, num_outputs=num_outputs)
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model
    if model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model
    raise ValueError(f"Unsupported model_name: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/csiro_biomass")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--targets", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv at {train_csv}")

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

    val_size = max(1, int(len(wide) * args.val_split))
    perm = np.random.permutation(len(wide))
    val_idx = perm[:val_size]
    tr_idx = perm[val_size:]
    tr_df = wide.iloc[tr_idx].copy()
    va_df = wide.iloc[val_idx].copy()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        normalize,
    ])
    valid_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = BiomassImageDataset(tr_df, data_dir, targets, train_tf)
    valid_ds = BiomassImageDataset(va_df, data_dir, targets, valid_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, num_outputs=len(targets), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    best_loss = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"image_{args.model_name}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for imgs, targets_tensor in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
            imgs = imgs.to(device)
            targets_tensor = targets_tensor.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for imgs, targets_tensor in tqdm(valid_loader, desc=f"Epoch {epoch} valid", leave=False):
                imgs = imgs.to(device)
                targets_tensor = targets_tensor.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets_tensor)
                valid_losses.append(loss.item())

        tr_loss = float(np.mean(train_losses)) if train_losses else 0.0
        va_loss = float(np.mean(valid_losses)) if valid_losses else 0.0
        print(f"Epoch {epoch}/{args.epochs} | train {tr_loss:.4f} | valid {va_loss:.4f}")

        if va_loss < best_loss:
            best_loss = va_loss
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
                    "custom_model": is_custom_model(args.model_name),
                },
                ckpt_path,
            )
            meta_path = out_dir / f"image_{args.model_name}.json"
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump({"best_valid_loss": best_loss, "targets_order": targets}, f, indent=2)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
