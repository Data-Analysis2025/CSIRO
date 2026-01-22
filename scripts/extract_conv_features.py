"""Extract embeddings from fine-tuned CNN checkpoints (e.g., ConvNeXt)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from path_utils import resolve_data_dir
from src.model_factory import create_model_from_name


class ImagePathsDataset(Dataset):
    def __init__(self, image_paths: List[str], root_dir: Path, tfms):
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        rel_path = self.image_paths[idx]
        img = Image.open(self.root_dir / rel_path).convert("RGB")
        return self.tfms(img), rel_path


def build_transforms(image_size: int, normalize_cfg: dict) -> transforms.Compose:
    mean = normalize_cfg.get("mean", [0.485, 0.456, 0.406])
    std = normalize_cfg.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def gather_features(
    checkpoint_path: Path,
    image_paths: List[str],
    data_dir: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt["model_name"]
    model_source = ckpt.get("model_source", "timm")
    if model_source != "timm":
        raise ValueError(f"Embedding extraction currently supports only timm models, got {model_source}.")

    model = create_model_from_name(
        model_name,
        num_outputs=len(ckpt["targets_order"]),
        pretrained=False,
        source=model_source,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tfms = build_transforms(int(ckpt.get("image_size", 224)), ckpt.get("normalize", {}))
    dataset = ImagePathsDataset(image_paths, data_dir, tfms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_feats: List[torch.Tensor] = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"Embedding {checkpoint_path.stem}", leave=False):
            imgs = imgs.to(device)
            feats = model.forward_features(imgs) if hasattr(model, "forward_features") else model(imgs)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim == 4:
                feats = feats.mean(dim=[2, 3])
            all_feats.append(feats.cpu())
    return torch.cat(all_feats).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-paths", type=str, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-prefix", type=str, default="convnext")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    unique_train = sorted(train_df["image_path"].unique())
    unique_test = sorted(test_df["image_path"].unique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_for_split(split_paths: List[str], split_name: str) -> pd.DataFrame:
        agg_feats = None
        for ckpt in args.checkpoint_paths:
            feats = gather_features(Path(ckpt), split_paths, data_dir, args.batch_size, device)
            agg_feats = feats if agg_feats is None else agg_feats + feats
        agg_feats /= len(args.checkpoint_paths)
        cols = [f"emb_{i}" for i in range(agg_feats.shape[1])]
        df = pd.DataFrame(agg_feats, columns=cols)
        df.insert(0, "image_path", split_paths)
        output_path = data_dir / f"{split_name}_embeddings_{args.output_prefix}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {split_name} embeddings to {output_path}")
        return df

    extract_for_split(unique_train, "train")
    extract_for_split(unique_test, "test")


if __name__ == "__main__":
    main()
