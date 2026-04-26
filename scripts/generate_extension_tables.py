from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scan_local import run_dataset_experiment


RESULT_DIR = PROJECT_ROOT / "data" / "scan_results"
ADDITIONAL_DIR = RESULT_DIR / "additional_datasets"


def run_beans_extension(device: torch.device) -> Path:
    ADDITIONAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_name, enable_pnp, k0, run_kwargs in [
        (
            "SCAN (local)",
            False,
            3,
            {
                "epochs": 15,
                "batch_size": 128,
                "lr": 1e-2,
                "entropy_weight": 5.0,
                "warmup_epochs": 0,
                "init_strategy": "kmeans",
            },
        ),
        (
            "Ours (local)",
            True,
            5,
            {
                "epochs": 60,
                "batch_size": 128,
                "lr": 5e-3,
                "entropy_weight": 5.0,
                "lambda_param": 2.0,
                "warmup_epochs": 20,
                "init_strategy": "kmeans",
            },
        ),
    ]:
        result = run_dataset_experiment(
            dataset_name="beans",
            k0=k0,
            enable_pnp=enable_pnp,
            method_name=method_name,
            device=device,
            **run_kwargs,
        )
        rows.append(
            {
                "Dataset": "beans",
                "Method": method_name,
                "K0": result.k0,
                "K*": result.inferred_k,
                "ACC(%)": result.acc * 100.0,
                "NMI(%)": result.nmi * 100.0,
                "ARI(%)": result.ari * 100.0,
                "Notes": "Small RGB benchmark outside the paper scope with 3 classes.",
            }
        )

    output_path = ADDITIONAL_DIR / "beans_comparison.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def run_init_ablation(device: torch.device) -> Path:
    rows = []
    configs = [
        (
            "SCAN (local)",
            False,
            10,
            "kmeans",
            {
                "epochs": 15,
                "batch_size": 2048,
                "lr": 1e-2,
                "entropy_weight": 5.0,
                "warmup_epochs": 0,
            },
        ),
        (
            "SCAN (local)",
            False,
            10,
            "random",
            {
                "epochs": 15,
                "batch_size": 2048,
                "lr": 1e-2,
                "entropy_weight": 5.0,
                "warmup_epochs": 0,
            },
        ),
        (
            "Ours (local)",
            True,
            20,
            "kmeans",
            {
                "epochs": 80,
                "batch_size": 2048,
                "lr": 5e-3,
                "entropy_weight": 5.0,
                "lambda_param": 2.0,
                "warmup_epochs": 30,
            },
        ),
        (
            "Ours (local)",
            True,
            20,
            "random",
            {
                "epochs": 80,
                "batch_size": 2048,
                "lr": 5e-3,
                "entropy_weight": 5.0,
                "lambda_param": 2.0,
                "warmup_epochs": 30,
            },
        ),
    ]

    for method_name, enable_pnp, k0, init_strategy, run_kwargs in configs:
        result = run_dataset_experiment(
            dataset_name="cifar-10",
            k0=k0,
            enable_pnp=enable_pnp,
            method_name=method_name,
            device=device,
            init_strategy=init_strategy,
            **run_kwargs,
        )
        rows.append(
            {
                "Dataset": "cifar-10",
                "Method": method_name,
                "K0": result.k0,
                "Init strategy": init_strategy,
                "K*": result.inferred_k,
                "ACC(%)": result.acc * 100.0,
                "NMI(%)": result.nmi * 100.0,
                "ARI(%)": result.ari * 100.0,
            }
        )

    output_path = RESULT_DIR / "table8_init_ablation.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beans_path = run_beans_extension(device)
    init_path = run_init_ablation(device)
    print(f"Saved: {beans_path}")
    print(f"Saved: {init_path}")


if __name__ == "__main__":
    main()
