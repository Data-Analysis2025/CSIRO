"""Run image-model inference and create a submission CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models_image import create_model as create_custom_model, is_custom_model

class ImageOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, tfms):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.tfms(image)
        return image, row["image_path"]


def build_model(model_name: str, num_outputs: int) -> nn.Module:
    if is_custom_model(model_name):
        return create_custom_model(model_name, num_outputs=num_outputs)
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model
    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
        return model
    raise ValueError(f"Unsupported model_name: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/csiro_biomass")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    test_csv = data_dir / "test.csv"
    sample_csv = data_dir / "sample_submission.csv"

    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test.csv at {test_csv}")
    if not sample_csv.exists():
        raise FileNotFoundError(f"Missing sample_submission.csv at {sample_csv}")

    test_df = pd.read_csv(test_csv)
    sample_submission = pd.read_csv(sample_csv)

    artifact = torch.load(args.model_path, map_location="cpu")
    targets: List[str] = artifact["targets_order"]
    model_name = artifact["model_name"]
    image_size = int(artifact.get("image_size", 224))
    norm = artifact.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})

    tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])

    unique_images = test_df[["image_path"]].drop_duplicates().reset_index(drop=True)
    ds = ImageOnlyDataset(unique_images, data_dir, tfms)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_outputs=len(targets)).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()

    image_to_pred: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            for p, pred in zip(paths, preds):
                image_to_pred[p] = pred

    pred_vals = []
    for _, row in test_df.iterrows():
        img_path = row["image_path"]
        tname = row["target_name"]
        if tname not in targets:
            raise KeyError(f"Unknown target_name: {tname}")
        idx = targets.index(tname)
        pred_vals.append(float(image_to_pred[img_path][idx]))

    submission = sample_submission.copy()
    submission["target"] = pred_vals
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
