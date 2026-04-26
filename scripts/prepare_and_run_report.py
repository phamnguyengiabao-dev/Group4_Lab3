from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scan_local import load_or_extract_features
from scripts.bootstrap_checkpoints import ensure_checkpoints
from scripts.bootstrap_third_party import ensure_upstream_repo


DEFAULT_DATASETS = ["cifar-10", "cifar-20", "stl-10", "imagenet-10", "beans"]


def prepare_datasets(
    datasets: list[str],
    device: torch.device,
    force_recompute_features: bool = False,
) -> None:
    for dataset_name in datasets:
        print(f"[prepare] dataset={dataset_name} device={device.type}")
        (train_features, train_labels), (eval_features, eval_labels) = load_or_extract_features(
            dataset_name=dataset_name,
            device=device,
            force_recompute=force_recompute_features,
        )
        print(
            "[prepare] ready",
            f"dataset={dataset_name}",
            f"train={tuple(train_features.shape)}",
            f"eval={tuple(eval_features.shape)}",
            f"train_labels={tuple(train_labels.shape)}",
            f"eval_labels={tuple(eval_labels.shape)}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare report datasets and feature caches before training or table generation."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to preload before training/report generation.",
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Re-extract cached features even if .pt cache files already exist.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] project_root={PROJECT_ROOT}")
    print(f"[env] device={device}")
    print(f"[env] datasets={args.datasets}")

    ensure_upstream_repo()
    ensure_checkpoints()
    prepare_datasets(
        datasets=args.datasets,
        device=device,
        force_recompute_features=args.force_recompute_features,
    )
    print("[done] dataset preparation finished.")


if __name__ == "__main__":
    main()
