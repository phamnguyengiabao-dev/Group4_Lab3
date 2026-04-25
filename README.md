# Group4_Lab3

Reproduction workspace for **Deep Plug-and-Play Cluster** with the following policy:

- **SCAN backbone/pipeline** is taken from upstream implementation (allowed by instructor guidance).
- **PnP split/merge module** is implemented locally by the team in this repo.
- Naming policy in tables:
  - **`PnP`** = paper's reported "Ours" (SCAN + paper PnP module).
  - **`Ours`** = local runs in this workspace (SCAN + team's local PnP module).

## Current workflow

### 1) Generate reproduction tables

Run from project root:

```bash
python scripts/generate_reproduction_tables.py --tables table1 table2 table3 table5 table6 table7
```

Generated CSV files are saved under:

- `data/scan_results/table1_main_comparison.csv`
- `data/scan_results/table1_paper_vs_local.csv`
- `data/scan_results/table2_inferred_k.csv`
- `data/scan_results/table3_imagenet50_summary.csv`
- `data/scan_results/table5_k0_stability.csv`
- `data/scan_results/table6_lambda_ablation.csv`
- `data/scan_results/table7_ablation_components.csv`

### 2) Open notebooks for report tables

- `notebooks/01_main_experiments.ipynb` for Table 1 and Table 2
- `notebooks/02_ablation_study.ipynb` for Table 5, Table 6, Table 7
- `notebooks/03_table3_imagenet50.ipynb` for Table 3 status

## Notes

- Main local runs are executed on four datasets in Table 1 scope:
  - CIFAR-10
  - CIFAR-100 (20 superclasses)
  - STL-10
  - ImageNet-10
- ImageNet-50 (Table 3) is summarized from paper values; local run is marked as pending if the dataset is not available in the current machine/workspace.
- Old notebook flows that were not aligned with SCAN+PnP direction have been removed.
