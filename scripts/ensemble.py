"""Flexible ensemble builder for arbitrary submission files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

DEFAULT_FILES = {
    "lgbm": "submission_lgbm.csv",
    "xgb": "submission_xgb.csv",
    "cat": "submission_cat.csv",
    "convnext": "submission_convnext.csv",
    "vit": "submission_vit.csv",
    "crosspvt": "submission_crosspvt.csv",
    "siglip": "submission_siglip.csv",
}


def parse_model_entry(entry: str, submissions_dir: Path) -> tuple[str, Path]:
    """
    Parse a model entry. Support either `name` (mapped via DEFAULT_FILES)
    or `alias=custom_path.csv`.
    """
    if "=" in entry:
        alias, path_str = entry.split("=", 1)
        return alias.strip(), Path(path_str.strip())
    alias = entry.strip()
    filename = DEFAULT_FILES.get(alias)
    if filename is None:
        raise ValueError(f"Unknown model alias '{alias}'. Provide alias=path.csv explicitly.")
    return alias, submissions_dir / filename


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing submission file: {path}")
    df = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
    print(f"Loaded {path.name}: {len(df)} rows")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "xgb", "cat"],
        help="Model aliases (lgbm/xgb/cat/convnext/crosspvt etc.) "
        "or custom entries alias=path/to/file.csv",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Weights for each model. Defaults to uniform.",
    )
    parser.add_argument("--output", type=str, default="submission_ensemble.csv")
    args = parser.parse_args()

    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)

    # Resolve paths and load
    entries: Dict[str, pd.DataFrame] = {}
    for entry in args.models:
        alias, path = parse_model_entry(entry, submissions_dir)
        entries[alias] = load_submission(path)

    # Validate IDs align
    ids = None
    for alias, df in entries.items():
        if ids is None:
            ids = df["sample_id"]
        elif not df["sample_id"].equals(ids):
            raise ValueError(f"Sample IDs mismatch in {alias}. Ensure identical ordering.")

    # Setup weights
    model_names: List[str] = list(entries.keys())
    if args.weights is None:
        weights = [1.0 / len(model_names)] * len(model_names)
    else:
        if len(args.weights) != len(model_names):
            raise ValueError("Number of weights must match number of models.")
        total = sum(args.weights)
        if total == 0:
            raise ValueError("Sum of weights must be > 0.")
        weights = [w / total for w in args.weights]

    print("Blending models:")
    for name, weight in zip(model_names, weights):
        print(f"  - {name}: weight={weight:.3f}")

    # Weighted sum
    blended = entries[model_names[0]][["sample_id"]].copy()
    blended["target"] = 0.0
    for name, weight in zip(model_names, weights):
        blended["target"] += entries[name]["target"] * weight

    output_path = submissions_dir / args.output
    blended.to_csv(output_path, index=False)
    print(f"Ensemble submission saved to {output_path}")
    print(blended.head())


if __name__ == "__main__":
    main()
