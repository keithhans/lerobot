#!/usr/bin/env python

"""Script to upload a previously saved dataset to the Hugging Face Hub."""

import argparse
from pathlib import Path

from datasets import load_from_disk
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import (
    push_lerobot_dataset_to_hub,
    from_dataset_to_lerobot_dataset,
)
from lerobot.common.datasets.compute_stats import compute_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a saved dataset to the Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID in format 'username/dataset_name'"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing the saved dataset"
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="Whether to compute dataset statistics before uploading"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags to add to the dataset on the Hub"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Construct paths
    local_dir = Path(args.root) / args.repo_id
    if not local_dir.exists():
        raise ValueError(f"Dataset directory not found at {local_dir}")

    # Load the saved dataset
    print(f"Loading dataset from {local_dir}...")
    hf_dataset = load_from_disk(str(local_dir / "train"))
    
    # Load metadata
    meta_data_dir = local_dir / "meta_data"
    if not meta_data_dir.exists():
        raise ValueError(f"Metadata directory not found at {meta_data_dir}")

    # Create LeRobotDataset instance
    videos_dir = local_dir / "videos"
    lerobot_dataset = LeRobotDataset(
        repo_id=args.repo_id,
        videos_dir=videos_dir,
    )
    lerobot_dataset.hf_dataset = hf_dataset

    # Compute statistics if requested
    if args.compute_stats:
        print("Computing dataset statistics...")
        lerobot_dataset.stats = compute_stats(lerobot_dataset)
    
    # Push to hub
    print(f"Pushing dataset to hub as {args.repo_id}...")
    push_lerobot_dataset_to_hub(lerobot_dataset, tags=args.tags)
    print("Dataset uploaded successfully!")


if __name__ == "__main__":
    main() 