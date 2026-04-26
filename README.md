# Group4_Lab3

Reproduction workspace for **Deep Plug-and-Play Cluster** with the following policy:

- **SCAN backbone/pipeline** is taken from upstream implementation (allowed by instructor guidance).
- **PnP split/merge module** is implemented locally by the team in this repo.
- Naming policy in tables:
  - **`PnP`** = paper's reported "Ours" (SCAN + paper PnP module).
  - **`Ours`** = local runs in this workspace (SCAN + team's local PnP module).

## Current workflow

### Data processing flow

- `src/scan_datasets.py`: builds the current dataset flow and applies evaluation transforms for `cifar-10`, `cifar-20`, `stl-10`, and `imagenet-10`.
- `src/data_pipeline.py`: extracts frozen features from pretrained checkpoints and mines nearest neighbors.
- `src/pnp_training.py`: contains the local SCAN head, phase-locked PnP split/merge logic, and the asymmetric cooldown training loop.
- `src/experiment_pipeline.py`: wraps experiment execution, paper-table references, and CSV export for the report.
- `src/scan_local.py`: compatibility layer that re-exports the current local pipeline to keep existing scripts/notebooks working.
- `third_party/Unsupervised-Classification/data/cifar.py` and `third_party/Unsupervised-Classification/data/stl.py`: upstream dataset loaders reused by the current flow for CIFAR and STL.

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

- `notebooks/01_main_experiments.ipynb` for the reproduction section: Table 1, Table 2, and Table 3 status
- `notebooks/02_ablation_study.ipynb` for the ablation section: Table 5, Table 6, and Table 7
- `notebooks/03_additional_datasets.ipynb` for extension runs on datasets outside the paper scope

## Notes

- Main local runs are executed on four datasets in Table 1 scope:
  - CIFAR-10
  - CIFAR-100 (20 superclasses)
  - STL-10
  - ImageNet-10
- Local table generation now uses KMeans head initialization, `epochs=80`, `warmup_epochs=30`, and a phase-locked asymmetric cooldown for PnP because the older quick-run setup was collapsing or underfitting on local runs.
- ImageNet-50 (Table 3) is summarized from paper values; local run is marked as pending if the dataset is not available in the current machine/workspace.
- Old notebook flows that were not aligned with SCAN+PnP direction have been removed.
- Legacy cached feature files from older BloodMNIST/STL-only experiments are not part of the current flow and can be safely removed.
