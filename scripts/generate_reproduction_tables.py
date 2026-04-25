from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scan_local import (
    PAPER_TABLE1,
    PAPER_TABLE2,
    PAPER_TABLE3,
    PAPER_TABLE5,
    PAPER_TABLE6,
    PAPER_TABLE7,
    SCAN_RESULTS_DIR,
    compare_to_paper,
    run_dataset_experiment,
    save_dataframe,
)


DATASET_CONFIG = {
    "cifar-10": {"scan_k": 10, "ours_k0s": [3, 20]},
    "cifar-20": {"scan_k": 20, "ours_k0s": [3, 30]},
    "stl-10": {"scan_k": 10, "ours_k0s": [3, 20]},
    "imagenet-10": {"scan_k": 10, "ours_k0s": [3, 20]},
}


def _paper_table1_rows() -> list[dict]:
    rows = []
    for dataset_name, methods in PAPER_TABLE1.items():
        for method_name, metrics in methods.items():
            normalized_method = method_name
            if method_name.startswith("Ours (paper"):
                normalized_method = method_name.replace("Ours (paper", "PnP (paper")
            k0 = ""
            if "K0=" in normalized_method:
                k0 = int(normalized_method.split("K0=")[1].rstrip(")"))
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Source": "paper",
                    "Method": normalized_method,
                    "K0": k0,
                    "K*": "",
                    **metrics,
                }
            )
    return rows


def run_table1_and_2(device: torch.device):
    table1_rows = _paper_table1_rows()
    comparison_rows = []
    inferred_rows = []

    for dataset_name, cfg in DATASET_CONFIG.items():
        scan_result = run_dataset_experiment(
            dataset_name,
            k0=cfg["scan_k"],
            enable_pnp=False,
            method_name="SCAN (local)",
            device=device,
            epochs=5,
            batch_size=1024 if dataset_name == "imagenet-10" else 2048,
            lr=1e-2,
            entropy_weight=5.0,
            warmup_epochs=0,
        )
        table1_rows.append({"Source": "local", "Method": "SCAN (local)", **scan_result.to_row()})
        comparison_rows.extend(compare_to_paper(dataset_name, scan_result, "SCAN").to_dict("records"))

        for k0 in cfg["ours_k0s"]:
            local_result = run_dataset_experiment(
                dataset_name,
                k0=k0,
                enable_pnp=True,
                method_name="ours",
                device=device,
                epochs=8,
                batch_size=1024 if dataset_name == "imagenet-10" else 2048,
                lr=5e-3,
                entropy_weight=5.0,
                lambda_param=2.0,
                warmup_epochs=2,
            )
            table1_rows.append({"Source": "local", "Method": "Ours", **local_result.to_row()})
            comparison_rows.extend(compare_to_paper(dataset_name, local_result, "PnP-paper").to_dict("records"))
            if dataset_name in PAPER_TABLE2 and k0 in PAPER_TABLE2[dataset_name]:
                inferred_rows.append(
                    {
                        "Dataset": dataset_name,
                        "K0": k0,
                        "Paper inferred K": PAPER_TABLE2[dataset_name][k0],
                        "Our inferred K": local_result.inferred_k,
                        "Relative deviation(%)": abs(local_result.inferred_k - PAPER_TABLE2[dataset_name][k0])
                        / max(PAPER_TABLE2[dataset_name][k0], 1e-8)
                        * 100.0,
                    }
                )

    table1_df = pd.DataFrame(table1_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    table2_df = pd.DataFrame(inferred_rows)
    save_dataframe(table1_df, "table1_main_comparison.csv")
    save_dataframe(comparison_df, "table1_paper_vs_local.csv")
    save_dataframe(table2_df, "table2_inferred_k.csv")


def run_table5(device: torch.device):
    k_values = [3, 10, 15, 20, 30, 40, 50, 100]
    rows = []
    for k0 in k_values:
        scan_result = run_dataset_experiment(
            "cifar-10",
            k0=k0,
            enable_pnp=False,
            method_name="SCAN (local)",
            device=device,
            epochs=4,
            batch_size=2048,
            lr=1e-2,
            entropy_weight=5.0,
            warmup_epochs=0,
        )
        ours_result = run_dataset_experiment(
            "cifar-10",
            k0=k0,
            enable_pnp=True,
            method_name="ours",
            device=device,
            epochs=6,
            batch_size=2048,
            lr=5e-3,
            entropy_weight=5.0,
            lambda_param=2.0,
            warmup_epochs=2,
        )
        rows.append(
            {
                "K0": k0,
                "SCAN (paper) ACC(%)": PAPER_TABLE5["SCAN"][k0],
                "SCAN (local) ACC(%)": scan_result.acc * 100.0,
                "PnP (paper) ACC(%)": PAPER_TABLE5["Ours (paper)"][k0],
                "Ours (local) ACC(%)": ours_result.acc * 100.0,
                "K_inferred (paper)": PAPER_TABLE5["K_inferred (paper)"][k0],
                "K_inferred (local)": ours_result.inferred_k,
            }
        )
    save_dataframe(pd.DataFrame(rows), "table5_k0_stability.csv")


def run_table6(device: torch.device):
    rows = []
    for dataset_name, k0 in [("cifar-20", 30), ("stl-10", 20)]:
        for lam in [1.5, 1.8, 2.0, 2.2, 2.5]:
            result = run_dataset_experiment(
                dataset_name,
                k0=k0,
                enable_pnp=True,
                method_name="ours",
                device=device,
                epochs=6,
                batch_size=2048 if dataset_name == "cifar-20" else 1024,
                lr=5e-3,
                entropy_weight=5.0,
                lambda_param=lam,
                warmup_epochs=2,
            )
            rows.append(
                {
                    "Dataset": dataset_name,
                    "lambda": lam,
                    "Paper ACC(%)": PAPER_TABLE6[dataset_name][lam],
                    "Ours (local) ACC(%)": result.acc * 100.0,
                }
            )
    save_dataframe(pd.DataFrame(rows), "table6_lambda_ablation.csv")


def run_table3_summary():
    rows = []
    for row in PAPER_TABLE3:
        rows.append(
            {
                "Source": "paper",
                "Dataset": "imagenet-50",
                "Method": row["Method"],
                "Inferred k": row["Inferred k"],
                "ACC(%)": row["ACC(%)"],
                "NMI(%)": row["NMI(%)"],
                "ARI(%)": row["ARI(%)"],
                "Local status": "",
            }
        )

    rows.append(
        {
            "Source": "local",
            "Dataset": "imagenet-50",
            "Method": "SCAN + ours",
            "Inferred k": "",
            "ACC(%)": "",
            "NMI(%)": "",
            "ARI(%)": "",
            "Local status": "not_run_missing_imagenet50_dataset",
        }
    )
    save_dataframe(pd.DataFrame(rows), "table3_imagenet50_summary.csv")


def run_table7(device: torch.device):
    rows = []
    ablations = [
        ("No split/merge", False, False, False),
        ("No split", False, True, False),
        ("No merge", True, False, True),
        ("No split loss (proxy)", True, True, False),
        ("Full method", True, True, True),
    ]
    for k0 in [3, 10, 20]:
        for name, enable_split, enable_merge, enable_split_bootstrap in ablations:
            result = run_dataset_experiment(
                "cifar-10",
                k0=k0,
                enable_pnp=name != "No split/merge" or enable_split or enable_merge,
                method_name="ours",
                device=device,
                epochs=6,
                batch_size=2048,
                lr=5e-3,
                entropy_weight=5.0,
                lambda_param=2.0,
                warmup_epochs=2,
                enable_split=enable_split,
                enable_merge=enable_merge,
                enable_split_bootstrap=enable_split_bootstrap,
            )
            paper_key = "No split loss" if name == "No split loss (proxy)" else name
            paper_ref = PAPER_TABLE7.get(paper_key, {}).get(k0)
            rows.append(
                {
                    "K0": k0,
                    "Ablation": name,
                    "Paper ACC(%)": paper_ref["ACC(%)"] if paper_ref else "",
                    "Paper NMI(%)": paper_ref["NMI(%)"] if paper_ref else "",
                    "Paper ARI(%)": paper_ref["ARI(%)"] if paper_ref else "",
                    "Ours (local) ACC(%)": result.acc * 100.0,
                    "Ours (local) NMI(%)": result.nmi * 100.0,
                    "Ours (local) ARI(%)": result.ari * 100.0,
                    "K*": result.inferred_k,
                }
            )
    save_dataframe(pd.DataFrame(rows), "table7_ablation_components.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", default=["table1", "table2", "table5", "table6", "table7"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SCAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    requested = set(args.tables)
    if "all" in requested or "table1" in requested or "table2" in requested:
        run_table1_and_2(device)
    if "all" in requested or "table3" in requested:
        run_table3_summary()
    if "all" in requested or "table5" in requested:
        run_table5(device)
    if "all" in requested or "table6" in requested:
        run_table6(device)
    if "all" in requested or "table7" in requested:
        run_table7(device)


if __name__ == "__main__":
    main()
