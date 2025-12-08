"""
Upload trained models and source code to Kaggle Datasets.

Usage:
    # First upload (create new dataset)
    python upload_kaggle_dataset.py --dataset-id htmp-baseline-models --dirs models src

    # Update existing dataset
    python upload_kaggle_dataset.py --dataset-id htmp-baseline-models --dirs models src --update --message "Updated models with new features"

    # Include configs directory
    python upload_kaggle_dataset.py --dataset-id htmp-baseline-models --dirs models src configs
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload files to Kaggle Datasets.")
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="ID of the dataset (e.g., htmp-baseline-models).",
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories to upload (e.g., models src configs).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing dataset instead of creating new one.",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Updated dataset",
        help="Version message for dataset update.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Kaggle username (optional, will be read from kaggle.json if not provided).",
    )
    args = parser.parse_args()

    # Get Kaggle username
    if args.username is None:
        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json_path.exists():
            with open(kaggle_json_path, "r") as f:
                kaggle_config = json.load(f)
                args.username = kaggle_config.get("username")
        if args.username is None:
            raise ValueError(
                "Could not determine Kaggle username. Please provide --username or ensure ~/.kaggle/kaggle.json exists."
            )

    # 1. Create temporary directory for upload
    temp_dir = tempfile.mkdtemp(prefix="kaggle_upload_")
    print(f"Temporary directory created at: {temp_dir}")

    try:
        # 2. Copy specified directories to temp directory
        for dir_path in args.dirs:
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Directory {dir_path} does not exist.")

            dest_dir = os.path.join(temp_dir, os.path.basename(os.path.normpath(dir_path)))
            print(f"Copying {dir_path} to {dest_dir}")
            shutil.copytree(dir_path, dest_dir)

        # 3. Initialize metadata file
        cmd = ["kaggle", "datasets", "init", "-p", temp_dir]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # 4. Update metadata file
        metadata_path = os.path.join(temp_dir, "dataset-metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        metadata["title"] = args.dataset_id.replace("-", " ").title()
        metadata["id"] = f"{args.username}/{args.dataset_id}"
        metadata["licenses"] = [{"name": "CC0-1.0"}]
        metadata["keywords"] = ["finance", "machine-learning", "time-series", "hull-tactical"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata updated:")
        print(json.dumps(metadata, indent=2))

        # 5. Create or update dataset
        if args.update:
            cmd = ["kaggle", "datasets", "version", "-p", temp_dir, "-m", args.message, "-r", "zip"]
            print(f"\nUpdating existing dataset...")
        else:
            cmd = ["kaggle", "datasets", "create", "-p", temp_dir, "-r", "zip"]
            print(f"\nCreating new dataset...")

        print(f"Running command: {' '.join(cmd)}")

        # Set environment variable for temp directory
        env = os.environ.copy()
        tmpdir = os.path.expanduser("~/tmp")
        env["TMPDIR"] = tmpdir
        os.makedirs(tmpdir, exist_ok=True)

        subprocess.run(cmd, check=True, env=env)

        print(f"\n✅ Dataset uploaded successfully!")
        print(f"Dataset URL: https://www.kaggle.com/datasets/{args.username}/{args.dataset_id}")
        print(f"\nTo use in Kaggle Notebook, add this dataset:")
        print(f"  {args.username}/{args.dataset_id}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

    finally:
        # 6. Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nTemporary directory cleaned up: {temp_dir}")


if __name__ == "__main__":
    main()
