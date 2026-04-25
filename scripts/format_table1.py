from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "data" / "scan_results"
INPUT_PATH = RESULT_DIR / "table1_main_comparison.csv"
OUTPUT_PATH = RESULT_DIR / "table1_structured_for_report.csv"


DATASET_ORDER = [
    ("cifar-10", "CIFAR-10"),
    ("cifar-20", "CIFAR-100"),
    ("stl-10", "STL-10"),
    ("imagenet-10", "ImageNet-10"),
]
METRICS = ["NMI(%)", "ACC(%)", "ARI(%)"]


def _lookup_value(df: pd.DataFrame, dataset: str, source: str, method: str, k0: int | None, metric: str) -> str:
    q = (df["Dataset"] == dataset) & (df["Source"] == source) & (df["Method"] == method)
    if k0 is not None:
        q = q & (df["K0"].fillna(-1).astype(int) == int(k0))
    rows = df[q]
    if rows.empty:
        return "-"
    value = rows.iloc[0][metric]
    try:
        return f"{float(value):.1f}"
    except Exception:
        return str(value)


def build_structured_table(df: pd.DataFrame) -> pd.DataFrame:
    def choose_method(source: str, preferred: str, fallback: str, dataset: str, k0: int | None) -> str:
        q = (df["Dataset"] == dataset) & (df["Source"] == source)
        if k0 is not None:
            q = q & (df["K0"].fillna(-1).astype(int) == int(k0))
        candidate = df[q & (df["Method"] == preferred)]
        if not candidate.empty:
            return preferred
        return fallback

    # Resolve local naming variants once so the formatter works with both old/new CSVs.
    local_ours_name = choose_method("local", "Ours", "ours", "cifar-10", 3)
    paper_pnp_name_k3 = choose_method("paper", "PnP (paper, K0=3)", "Ours (paper, K0=3)", "cifar-10", 3)
    paper_pnp_name_k20 = choose_method("paper", "PnP (paper, K0=20)", "Ours (paper, K0=20)", "cifar-10", 20)
    paper_pnp_name_k30 = choose_method("paper", "PnP (paper, K0=30)", "Ours (paper, K0=30)", "cifar-20", 30)

    methods = [
        ("SCAN (paper)", ("paper", "SCAN", None)),
        ("SCAN (local)", ("local", "SCAN (local)", None)),
        ("PnP (K0=3)", ("paper", paper_pnp_name_k3, 3)),
        ("PnP (K0=20)", ("paper", paper_pnp_name_k20, 20)),
        ("PnP (K0=30)", ("paper", paper_pnp_name_k30, 30)),
        ("Ours (K0=3)", ("local", local_ours_name, 3)),
        ("Ours (K0=20)", ("local", local_ours_name, 20)),
        ("Ours (K0=30)", ("local", local_ours_name, 30)),
    ]

    records = []
    for method_name, (source, lookup_method, k0) in methods:
        row = {"Method": method_name}
        for dataset_key, dataset_label in DATASET_ORDER:
            for metric in METRICS:
                row[f"{dataset_label}|{metric.replace('(%)', '')}"] = _lookup_value(
                    df=df,
                    dataset=dataset_key,
                    source=source,
                    method=lookup_method,
                    k0=k0,
                    metric=metric,
                )
        records.append(row)
    return pd.DataFrame(records)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    structured = build_structured_table(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    structured.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
