#!/usr/bin/env python
"""Quickly inspect a CSV to decide config values (target, drop columns)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--nrows", type=int, default=5, help="Rows to preview")
    args = parser.parse_args()

    path = Path(args.train)
    df = pd.read_csv(path)
    print(f"File: {path}  rows={len(df)}  cols={len(df.columns)}")
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head(args.nrows))


if __name__ == "__main__":
    main()
