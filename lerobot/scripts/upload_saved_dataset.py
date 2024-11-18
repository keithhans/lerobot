#!/usr/bin/env python

"""Script to upload a previously saved dataset to the Hugging Face Hub."""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
from lerobot.common.datasets.populate_dataset import push_lerobot_dataset_to_hub
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.utils import calculate_episode_data_index


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

    # Load info.json
    with open(meta_data_dir / "info.json", "r") as f:
        info = json.load(f)

    # Load episode_data_index.json if it exists
    episode_data_index_path = meta_data_dir / "episode_data_index.json"
    if episode_data_index_path.exists():
        with open(episode_data_index_path, "r") as f:
            episode_data_index = json.load(f)
    else:
        print("Computing episode data index...")
        episode_data_index = calculate_episode_data_index(hf_dataset)

    # Create LeRobotDataset instance
    videos_dir = local_dir / "videos"
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=args.repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

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