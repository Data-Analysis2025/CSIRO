"""Utilities for running CrossPVT fold inference outside of notebooks."""
from __future__ import annotations

import gc
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .model import CFG, CrossPVT_T2T_MambaDINO, update_cfg_from_checkpoint


class BiomassInferenceDataset(Dataset):
    """Load split CSIRO test images and return left/right halves."""

    def __init__(self, image_paths: Iterable[str], base_dir: Path, transform: A.BasicTransform):
        self.image_paths: List[str] = [p.lstrip("./") for p in image_paths]
        self.base_dir = Path(base_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        rel_path = self.image_paths[idx]
        img_path = self.base_dir / rel_path
        if not img_path.exists():
            # Some entries may already include the directory name; fall back to filename only.
            img_path = self.base_dir / Path(rel_path).name
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left_t = self.transform(image=left)["image"]
        right_t = self.transform(image=right)["image"]
        return left_t, right_t, idx


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def _reset_train_cfg() -> None:
    update_cfg_from_checkpoint(asdict(CFG))


def load_fold_model(
    ckpt_path: Path,
    device: torch.device,
) -> CrossPVT_T2T_MambaDINO:
    state = torch.load(ckpt_path, map_location="cpu")
    _reset_train_cfg()
    update_cfg_from_checkpoint(state.get("cfg", {}))
    model = CrossPVT_T2T_MambaDINO(dropout=CFG.dropout, hidden_ratio=CFG.hidden_ratio)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_fold(model: CrossPVT_T2T_MambaDINO, loader: DataLoader, device: torch.device) -> np.ndarray:
    preds = np.zeros((len(loader.dataset), len(CFG.ALL_TARGET_COLS)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for left, right, indices in loader:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            out = model(x_left=left, x_right=right)
            batch = out["pred"].detach().cpu().numpy().astype(np.float32)
            preds[indices.cpu().numpy()] = batch
    return preds


def run_ensemble_inference(
    checkpoint_root: Path,
    run_name: str,
    fold_ids: Sequence[int],
    competition_root: Path,
    output_path: Path,
    ckpt_filename: str = "best_wr2.pt",
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
) -> Path:
    """Average specified folds and write submission.csv-compatible file."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comp_root = Path(competition_root)
    checkpoint_root = Path(checkpoint_root)
    output_path = Path(output_path)

    test_df = pd.read_csv(comp_root / "test.csv")
    sample_sub = pd.read_csv(comp_root / "sample_submission.csv")
    unique_paths = test_df["image_path"].drop_duplicates().tolist()

    dataset = None
    loader = None
    aggregated = None

    for fold_id in fold_ids:
        ckpt_path = checkpoint_root / run_name / f"fold_{fold_id}" / "checkpoints" / ckpt_filename
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        model = load_fold_model(ckpt_path, device=device)
        if dataset is None:
            img_size = getattr(model, "input_res", 518)
            transform = build_transforms(img_size)
            dataset = BiomassInferenceDataset(unique_paths, comp_root, transform)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            aggregated = np.zeros((len(dataset), len(CFG.ALL_TARGET_COLS)), dtype=np.float32)
        fold_preds = predict_fold(model, loader, device=device)
        aggregated += fold_preds
        del model
        torch.cuda.empty_cache()
        gc.collect()

    aggregated /= max(1, len(fold_ids))
    aggregated = np.clip(aggregated, 0.0, None)

    lookup = {}
    for rel_path, row in zip(unique_paths, aggregated):
        base = Path(rel_path).stem
        for name, value in zip(CFG.ALL_TARGET_COLS, row):
            lookup[f"{base}__{name}"] = float(value)

    submission = sample_sub.copy()
    submission["target"] = submission["sample_id"].map(lookup).astype(np.float32)
    if submission["target"].isna().any():
        missing = submission[submission["target"].isna()]["sample_id"].tolist()[:5]
        raise ValueError(f"Missing predictions for sample_ids: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path
