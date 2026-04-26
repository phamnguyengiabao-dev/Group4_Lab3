from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.divergence import compute_merge_threshold, pairwise_js_matrix
from src.metrics import evaluate_clustering
from src.utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "Unsupervised-Classification"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from losses.losses import SCANLoss  # type: ignore  # noqa: E402


PnPAction = Literal["none", "merge", "split"]


class DynamicClusterHead(nn.Module):
    """A linear clustering head with split/merge support."""

    def __init__(self, input_dim: int, k: int):
        super().__init__()
        if k < 2:
            raise ValueError("k must be at least 2.")
        self.input_dim = input_dim
        self.output_layer = nn.Linear(input_dim, k)

    @property
    def k(self) -> int:
        return int(self.output_layer.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(x)

    def _rebuild(self, new_weight: torch.Tensor, new_bias: torch.Tensor):
        layer = nn.Linear(self.input_dim, new_weight.size(0), bias=True).to(new_weight.device)
        with torch.no_grad():
            layer.weight.copy_(new_weight)
            layer.bias.copy_(new_bias)
        self.output_layer = layer

    def split_cluster(self, cluster_idx: int, noise_std: float = 1e-2):
        w = self.output_layer.weight.data
        b = self.output_layer.bias.data
        duplicate = w[cluster_idx].clone()
        duplicate_b = b[cluster_idx].clone()
        noise = torch.randn_like(duplicate) * noise_std
        new_weight = torch.cat([w, (duplicate + noise).unsqueeze(0)], dim=0)
        new_bias = torch.cat([b, duplicate_b.unsqueeze(0)], dim=0)
        self._rebuild(new_weight, new_bias)

    def split_cluster_with_centroids(self, cluster_idx: int, centroid_a: torch.Tensor, centroid_b: torch.Tensor):
        w = self.output_layer.weight.data.clone()
        b = self.output_layer.bias.data.clone()
        new_weight = torch.cat([w, centroid_b.unsqueeze(0)], dim=0)
        new_bias = torch.cat([b, torch.zeros(1, device=b.device, dtype=b.dtype)], dim=0)
        new_weight[cluster_idx] = centroid_a
        self._rebuild(new_weight, new_bias)

    def merge_clusters(self, k1_idx: int, k2_idx: int):
        i, j = sorted((int(k1_idx), int(k2_idx)))
        w = self.output_layer.weight.data
        b = self.output_layer.bias.data
        merged_w = 0.5 * (w[i] + w[j])
        merged_b = 0.5 * (b[i] + b[j])
        keep_mask = torch.ones(self.k, dtype=torch.bool, device=w.device)
        keep_mask[i] = False
        keep_mask[j] = False
        remain_w = w[keep_mask]
        remain_b = b[keep_mask]
        new_weight = torch.cat([remain_w, merged_w.unsqueeze(0)], dim=0)
        new_bias = torch.cat([remain_b, merged_b.unsqueeze(0)], dim=0)
        self._rebuild(new_weight, new_bias)


def _mean_cluster_distributions(probs: torch.Tensor, assignments: torch.Tensor, k: int) -> torch.Tensor:
    rows = []
    for cluster_id in range(k):
        mask = assignments == cluster_id
        if int(mask.sum()) == 0:
            rows.append(torch.full((k,), 1.0 / k, device=probs.device, dtype=probs.dtype))
        else:
            dist = probs[mask].mean(dim=0)
            dist = dist / torch.clamp(dist.sum(), min=1e-8)
            rows.append(dist)
    return torch.stack(rows, dim=0)


def _kmeans_subcluster_centroids(features: torch.Tensor, random_state: int = 42) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if features.size(0) < 4:
        return None, None
    x = features.cpu().numpy()
    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = km.fit_predict(x)
    if len(np.unique(labels)) < 2:
        return None, None
    c0 = torch.from_numpy(x[labels == 0].mean(axis=0)).float().to(features.device)
    c1 = torch.from_numpy(x[labels == 1].mean(axis=0)).float().to(features.device)
    return c0, c1


def _kmeans_head_init(features: torch.Tensor, k: int, random_state: int = 42) -> Optional[torch.Tensor]:
    if features.size(0) < k:
        return None
    x = features.cpu().numpy()
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(x)
    return torch.from_numpy(km.cluster_centers_).float().to(features.device)


def _pnp_step(
    head: DynamicClusterHead,
    features: torch.Tensor,
    probs: torch.Tensor,
    lambda_param: float,
    enable_split: bool,
    enable_merge: bool,
    seed: int,
    k0: int,
    target_k: int,
    enable_split_bootstrap: bool,
) -> PnPAction:
    assignments = probs.argmax(dim=1)
    k = head.k
    cluster_probs = _mean_cluster_distributions(probs, assignments, k)
    js_mat = pairwise_js_matrix(cluster_probs)

    target_band_low = max(2, target_k - 1)
    target_band_high = target_k + 1
    if target_band_low <= k <= target_band_high:
        return "none"

    is_growth_mode = k0 < target_k and k < target_band_low
    is_shrink_mode = k0 > target_k and k > target_band_high

    if enable_merge and k > 2 and not is_growth_mode:
        merge_threshold = compute_merge_threshold(js_mat, lambda_param, k)
        js_mat_masked = js_mat.clone()
        js_mat_masked.fill_diagonal_(float("inf"))
        min_js = float(js_mat_masked.min().item())
        if min_js < merge_threshold:
            idx = int(torch.argmin(js_mat_masked).item())
            c1, c2 = idx // k, idx % k
            head.merge_clusters(c1, c2)
            print(f"[PnP] MERGED clusters {c1} & {c2}. New K = {head.k}")
            return "merge"

    split_cap = min(20, target_k + 2)
    if enable_split and k <= split_cap and not is_shrink_mode:
        global_variance = float(features.var(dim=0).mean().item())
        split_variance_threshold = min(0.05, max(0.008, 0.75 * global_variance))
        variances = []
        for cluster_id in range(k):
            mask = assignments == cluster_id
            if int(mask.sum()) > 5:
                variances.append(float(features[mask].var(dim=0).mean().item()))
            else:
                variances.append(0.0)

        best_split_cluster = int(np.argmax(variances))
        best_split_variance = float(variances[best_split_cluster])

        if enable_split_bootstrap and k <= 5 and best_split_variance <= 0.0:
            counts = torch.bincount(assignments, minlength=k)
            best_split_cluster = int(torch.argmax(counts).item())
            mask = assignments == best_split_cluster
            if int(mask.sum()) > 5:
                best_split_variance = float(features[mask].var(dim=0).mean().item())

        if best_split_variance > split_variance_threshold:
            mask = assignments == best_split_cluster
            c0, c1 = _kmeans_subcluster_centroids(features[mask], random_state=seed)
            if c0 is not None and c1 is not None:
                head.split_cluster_with_centroids(best_split_cluster, c0, c1)
            else:
                head.split_cluster(best_split_cluster)
            print(
                f"[PnP] SPLIT cluster {best_split_cluster} "
                f"(variance={best_split_variance:.4f}, threshold={split_variance_threshold:.4f}, cap={split_cap}). "
                f"New K = {head.k}"
            )
            return "split"

    return "none"


def train_scan_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    neighbor_indices: np.ndarray,
    k0: int,
    method_name: str,
    epochs: int = 80,
    batch_size: int = 2048,
    lr: float = 1e-2,
    entropy_weight: float = 5.0,
    lambda_param: float = 2.0,
    warmup_epochs: int = 30,
    enable_pnp: bool = False,
    enable_split: bool = True,
    enable_merge: bool = True,
    enable_split_bootstrap: bool = True,
    init_strategy: str = "kmeans",
    seed: int = 42,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.experiment_pipeline import ExperimentResult

    set_seed(seed)
    x_train = train_features.to(device).float()
    y_train = train_labels.cpu().numpy()
    x_eval = eval_features.to(device).float()
    y_eval = eval_labels.cpu().numpy()
    target_k = int(np.unique(y_train).size)
    head = DynamicClusterHead(input_dim=x_train.size(1), k=k0).to(device)
    if init_strategy == "kmeans":
        centroids = _kmeans_head_init(x_train, k0, random_state=seed)
        if centroids is not None:
            with torch.no_grad():
                head.output_layer.weight.copy_(centroids)
                head.output_layer.bias.zero_()
    elif init_strategy != "random":
        raise ValueError(f"Unsupported init_strategy {init_strategy}")

    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = SCANLoss(entropy_weight=entropy_weight).to(device)

    n = x_train.size(0)
    anchor_indices = np.arange(n)
    cooldown = 0
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(anchor_indices)
        head.train()
        for start in range(0, n, batch_size):
            batch_idx = anchor_indices[start : start + batch_size]
            if len(batch_idx) == 0:
                continue
            neighbor_cols = rng.integers(0, neighbor_indices.shape[1], size=len(batch_idx))
            neighbor_idx = neighbor_indices[batch_idx, neighbor_cols]

            anchors = x_train[torch.as_tensor(batch_idx, device=device)]
            neighbors = x_train[torch.as_tensor(neighbor_idx, device=device)]
            anchor_logits = head(anchors)
            neighbor_logits = head(neighbors)
            loss, _, _ = criterion(anchor_logits, neighbor_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if cooldown > 0:
            cooldown -= 1
        elif enable_pnp and epoch >= warmup_epochs:
            head.eval()
            with torch.no_grad():
                probs_all = F.softmax(head(x_train), dim=1)
            action = _pnp_step(
                head=head,
                features=x_train,
                probs=probs_all,
                lambda_param=lambda_param,
                enable_split=enable_split,
                enable_merge=enable_merge,
                seed=seed + epoch,
                k0=k0,
                target_k=target_k,
                enable_split_bootstrap=enable_split_bootstrap,
            )
            if action != "none":
                optimizer = torch.optim.Adam(head.parameters(), lr=lr * 0.5, weight_decay=1e-4)
                if action == "merge":
                    cooldown = 3
                elif action == "split":
                    cooldown = 10

    head.eval()
    with torch.no_grad():
        eval_logits = head(x_eval)
        eval_preds = torch.argmax(eval_logits, dim=1).cpu().numpy()
        inferred_k = int(torch.unique(torch.argmax(head(x_train), dim=1)).numel())
    acc, nmi, ari = evaluate_clustering(y_eval, eval_preds)
    return ExperimentResult(
        dataset="",
        method=method_name,
        k0=k0,
        inferred_k=inferred_k,
        acc=float(acc),
        nmi=float(nmi),
        ari=float(ari),
    )
