"""CLI wrapper to run CrossPVT ensemble inference without editing notebooks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_crosspvt import run_ensemble_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Dataset root that holds run directories.")
    parser.add_argument("--run-name", type=str, required=True, help="Run directory name containing fold_x checkpoints.")
    parser.add_argument("--competition-root", type=Path, required=True, help="Path containing Kaggle competition files.")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold IDs to average (default: 0 1 2 3 4).",
    )
    parser.add_argument("--ckpt-filename", type=str, default="best_wr2.pt", help="Checkpoint filename stored per fold.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Where to write the submission csv (default: submission.csv in CWD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_ensemble_inference(
        checkpoint_root=args.checkpoint_root,
        run_name=args.run_name,
        fold_ids=args.folds,
        competition_root=args.competition_root,
        output_path=args.output,
        ckpt_filename=args.ckpt_filename,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
