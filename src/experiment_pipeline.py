from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

from src.data_pipeline import PROJECT_ROOT, load_or_extract_features, mine_neighbors
from src.pnp_training import train_scan_head
from src.utils import relative_deviation


SCAN_RESULTS_DIR = PROJECT_ROOT / "data" / "scan_results"


PAPER_TABLE1 = {
    "cifar-10": {
        "SCAN": {"NMI(%)": 71.2, "ACC(%)": 81.8, "ARI(%)": 66.5},
        "Ours (paper, K0=3)": {"NMI(%)": 71.9, "ACC(%)": 82.4, "ARI(%)": 67.5},
        "Ours (paper, K0=20)": {"NMI(%)": 71.1, "ACC(%)": 81.6, "ARI(%)": 66.2},
    },
    "cifar-20": {
        "SCAN": {"NMI(%)": 44.1, "ACC(%)": 42.2, "ARI(%)": 26.7},
        "Ours (paper, K0=3)": {"NMI(%)": 45.2, "ACC(%)": 43.8, "ARI(%)": 28.1},
        "Ours (paper, K0=30)": {"NMI(%)": 44.9, "ACC(%)": 43.1, "ARI(%)": 27.8},
    },
    "stl-10": {
        "SCAN": {"NMI(%)": 65.4, "ACC(%)": 75.5, "ARI(%)": 59.0},
        "Ours (paper, K0=3)": {"NMI(%)": 64.3, "ACC(%)": 74.5, "ARI(%)": 57.6},
        "Ours (paper, K0=20)": {"NMI(%)": 65.1, "ACC(%)": 74.7, "ARI(%)": 58.9},
    },
    "imagenet-10": {
        "SCAN": {"NMI(%)": 86.2, "ACC(%)": 92.0, "ARI(%)": 83.3},
        "Ours (paper, K0=3)": {"NMI(%)": 88.6, "ACC(%)": 91.2, "ARI(%)": 87.1},
        "Ours (paper, K0=20)": {"NMI(%)": 86.9, "ACC(%)": 91.8, "ARI(%)": 84.7},
    },
}

PAPER_TABLE2 = {
    "cifar-10": {3: 10.0, 20: 10.0},
    "cifar-20": {3: 19.7, 30: 19.8},
    "stl-10": {3: 9.7, 20: 10.3},
    "imagenet-10": {3: 10.0, 20: 10.3},
}

PAPER_TABLE3 = [
    {"Method": "SCAN (k=50)", "Inferred k": "-", "ACC(%)": "73.7±1.7", "NMI(%)": "79.7±0.6", "ARI(%)": "61.8±1.3"},
    {"Method": "SCAN (k=10)", "Inferred k": "-", "ACC(%)": "19.4±0.1", "NMI(%)": "60.6±0.4", "ARI(%)": "23.0±0.3"},
    {"Method": "DBSCAN", "Inferred k": "16.0", "ACC(%)": "24.0±0.0", "NMI(%)": "52.0±0.0", "ARI(%)": "9.0±0.0"},
    {"Method": "moVB", "Inferred k": "46.2±1.3", "ACC(%)": "55.0±2.0", "NMI(%)": "70.0±1.0", "ARI(%)": "43.0±1.0"},
    {"Method": "DPM Sampler", "Inferred k": "72.0±2.6", "ACC(%)": "57.0±1.0", "NMI(%)": "72.0±0.0", "ARI(%)": "43.0±1.0"},
    {"Method": "DeepDPM", "Inferred k": "55.3±1.5", "ACC(%)": "66.0±1.0", "NMI(%)": "77.0±0.0", "ARI(%)": "54.0±1.0"},
    {"Method": "Ours (paper)", "Inferred k": "50.6±1.7", "ACC(%)": "73.3±1.3", "NMI(%)": "80.1±0.7", "ARI(%)": "62.3±1.5"},
]

PAPER_TABLE5 = {
    "SCAN": {3: 28.6, 10: 82.2, 15: 58.8, 20: 45.9, 30: 33.4, 40: 25.9, 50: 21.9, 100: 11.9},
    "Ours (paper)": {3: 82.4, 10: 82.4, 15: 82.9, 20: 81.6, 30: 82.0, 40: 77.5, 50: 70.9, 100: 72.4},
    "K_inferred (paper)": {3: 10, 10: 10, 15: 10, 20: 10, 30: 10, 40: 11, 50: 12, 100: 12},
}

PAPER_TABLE6 = {
    "cifar-20": {1.5: 39.84, 1.8: 43.57, 2.0: 44.71, 2.2: 41.34, 2.5: 41.40},
    "stl-10": {1.5: 69.46, 1.8: 71.14, 2.0: 78.40, 2.2: 76.49, 2.5: 75.53},
}

PAPER_TABLE7 = {
    "No split/merge": {
        3: {"ACC(%)": 28.7, "NMI(%)": 46.2, "ARI(%)": 26.4},
        10: {"ACC(%)": 82.2, "NMI(%)": 71.6, "ARI(%)": 66.8},
        20: {"ACC(%)": 47.3, "NMI(%)": 62.3, "ARI(%)": 43.2},
    },
    "No split": {
        3: {"ACC(%)": 28.9, "NMI(%)": 46.1, "ARI(%)": 26.4},
        10: {"ACC(%)": 75.4, "NMI(%)": 68.9, "ARI(%)": 62.1},
        20: {"ACC(%)": 81.2, "NMI(%)": 70.9, "ARI(%)": 65.6},
    },
    "No merge": {
        3: {"ACC(%)": 79.7, "NMI(%)": 71.3, "ARI(%)": 66.9},
        10: {"ACC(%)": 82.1, "NMI(%)": 71.5, "ARI(%)": 66.8},
        20: {"ACC(%)": 47.1, "NMI(%)": 61.7, "ARI(%)": 42.6},
    },
    "No split loss": {
        3: {"ACC(%)": 28.6, "NMI(%)": 44.5, "ARI(%)": 25.3},
        10: {"ACC(%)": 77.6, "NMI(%)": 69.9, "ARI(%)": 63.3},
        20: {"ACC(%)": 81.7, "NMI(%)": 71.1, "ARI(%)": 66.4},
    },
    "Full method": {
        3: {"ACC(%)": 82.4, "NMI(%)": 72.2, "ARI(%)": 67.9},
        10: {"ACC(%)": 82.4, "NMI(%)": 71.7, "ARI(%)": 67.4},
        20: {"ACC(%)": 82.7, "NMI(%)": 72.3, "ARI(%)": 67.7},
    },
}


@dataclass
class ExperimentResult:
    dataset: str
    method: str
    k0: int
    inferred_k: int
    acc: float
    nmi: float
    ari: float

    def to_row(self) -> Dict[str, float | int | str]:
        return {
            "Dataset": self.dataset,
            "Method": self.method,
            "K0": self.k0,
            "K*": self.inferred_k,
            "ACC(%)": self.acc * 100.0,
            "NMI(%)": self.nmi * 100.0,
            "ARI(%)": self.ari * 100.0,
        }


def _reference_metrics(dataset_name: str, k0: int, method_key: str) -> Optional[Dict[str, float]]:
    refs = PAPER_TABLE1.get(dataset_name, {})
    if method_key == "SCAN":
        return refs.get("SCAN")
    if method_key in {"Ours-paper", "PnP-paper"}:
        label = f"Ours (paper, K0={k0})"
        return refs.get(label)
    return None


def compare_to_paper(dataset_name: str, local_result: ExperimentResult, paper_method: str) -> pd.DataFrame:
    ref = _reference_metrics(dataset_name, local_result.k0, paper_method)
    if ref is None:
        return pd.DataFrame()

    rows = []
    for metric in ["ACC(%)", "NMI(%)", "ARI(%)"]:
        our_value = local_result.to_row()[metric]
        rows.append(
            {
                "Dataset": dataset_name,
                "Paper method": "SCAN" if paper_method == "SCAN" else f"PnP (paper, K0={local_result.k0})",
                "Local method": local_result.method,
                "Metric": metric,
                "Paper result": ref[metric],
                "Our result": our_value,
                "Relative deviation(%)": relative_deviation(our_value, ref[metric]),
            }
        )
    return pd.DataFrame(rows)


def run_dataset_experiment(
    dataset_name: str,
    k0: int,
    enable_pnp: bool,
    method_name: str,
    device: Optional[torch.device] = None,
    force_recompute_features: bool = False,
    **train_kwargs,
) -> ExperimentResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train_features, train_labels), (eval_features, eval_labels) = load_or_extract_features(
        dataset_name,
        device=device,
        force_recompute=force_recompute_features,
    )
    neighbors = mine_neighbors(train_features, topk=20 if dataset_name != "imagenet-10" else 50)
    result = train_scan_head(
        train_features=train_features,
        train_labels=train_labels,
        eval_features=eval_features,
        eval_labels=eval_labels,
        neighbor_indices=neighbors,
        k0=k0,
        enable_pnp=enable_pnp,
        method_name=method_name,
        device=device,
        **train_kwargs,
    )
    result.dataset = dataset_name
    return result


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    SCAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCAN_RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path
