# Reproduction Notes (SCAN + PnP)

Use these section titles in the report:

1. `Main comparison on CIFAR-10 / CIFAR-100-20 / STL-10 / ImageNet-10`
2. `Convergence of inferred K (Table 2)`
3. `Ablation on K0 stability (Table 5)`
4. `Ablation on lambda (Table 6)`
5. `Ablation on split/merge components (Table 7)`
6. `ImageNet-50 summary (Table 3)`

Naming convention in all tables:

- `SCAN` for baseline from paper / local SCAN branch.
- `PnP` for numbers reported by the paper (their "Ours", i.e. SCAN + paper PnP).
- `Ours` for runs executed in this workspace (SCAN + local PnP).

Mandatory comparison columns:

- `paper result`
- `our result`
- `relative deviation (%)`

Consistency statement for methodology chapter:

> SCAN backbone and training pipeline are reused from upstream implementation, while the plug-and-play split/merge module is locally implemented and integrated in this project according to the paper's formulation.
