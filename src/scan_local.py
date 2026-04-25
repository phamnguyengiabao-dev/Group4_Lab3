from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from src.divergence import compute_js_divergence, compute_merge_threshold, compute_split_threshold, pairwise_js_matrix
from src.metrics import evaluate_clustering
from src.scan_datasets import DatasetBundle, build_dataset_bundle
from src.utils import relative_deviation, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "Unsupervised-Classification"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from losses.losses import SCANLoss  # type: ignore  # noqa: E402
from models.resnet import resnet50  # type: ignore  # noqa: E402
from models.resnet_cifar import resnet18 as cifar_resnet18  # type: ignore  # noqa: E402
from models.resnet_stl import resnet18 as stl_resnet18  # type: ignore  # noqa: E402


SCAN_PRETEXT_DIR = PROJECT_ROOT / "data" / "scan_pretext"
SCAN_CACHE_DIR = PROJECT_ROOT / "data" / "scan_cache"
SCAN_RESULTS_DIR = PROJECT_ROOT / "data" / "scan_results"
MOCO_CKPT = PROJECT_ROOT / "data" / "checkpoints" / "moco_v2_800ep_pretrain.pth.tar"


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

# Table 6 values were extracted from the PDF text and are used only as reference.
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

    def reset_parameters(self):
        self.output_layer.reset_parameters()

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


def _strip_prefix_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


def build_pretrained_encoder(dataset_name: str, device: torch.device) -> Tuple[nn.Module, int]:
    if dataset_name == "cifar-10":
        backbone = cifar_resnet18()["backbone"]
        state = torch.load(SCAN_PRETEXT_DIR / "cifar10_pretext.pth.tar", map_location="cpu")
        backbone.load_state_dict(_strip_prefix_state_dict(state, "backbone."), strict=True)
        return backbone.to(device).eval(), 512

    if dataset_name == "cifar-20":
        backbone = cifar_resnet18()["backbone"]
        state = torch.load(SCAN_PRETEXT_DIR / "cifar20_pretext.pth.tar", map_location="cpu")
        backbone.load_state_dict(_strip_prefix_state_dict(state, "backbone."), strict=True)
        return backbone.to(device).eval(), 512

    if dataset_name == "stl-10":
        backbone = stl_resnet18()["backbone"]
        state = torch.load(SCAN_PRETEXT_DIR / "stl10_pretext.pth.tar", map_location="cpu")
        backbone.load_state_dict(_strip_prefix_state_dict(state, "backbone."), strict=True)
        return backbone.to(device).eval(), 512

    if dataset_name == "imagenet-10":
        backbone = resnet50()["backbone"]
        checkpoint = torch.load(MOCO_CKPT, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        cleaned = {}
        for key, value in state_dict.items():
            if not key.startswith("module.encoder_q."):
                continue
            new_key = key[len("module.encoder_q.") :]
            if new_key.startswith("fc."):
                continue
            cleaned[new_key] = value
        backbone.load_state_dict(cleaned, strict=False)
        return backbone.to(device).eval(), 2048

    raise ValueError(f"Unsupported dataset {dataset_name}")


def _feature_cache_paths(dataset_name: str) -> Tuple[Path, Path]:
    SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_path = SCAN_CACHE_DIR / f"{dataset_name}_train_features.pt"
    eval_path = SCAN_CACHE_DIR / f"{dataset_name}_eval_features.pt"
    return train_path, eval_path


@torch.no_grad()
def _extract_features_for_dataset(dataset, encoder: nn.Module, device: torch.device, batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"]
        features = encoder(images)
        feats.append(features.cpu())
        labels.append(targets.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def load_or_extract_features(dataset_name: str, device: torch.device, force_recompute: bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    train_cache, eval_cache = _feature_cache_paths(dataset_name)

    if not force_recompute and train_cache.exists() and eval_cache.exists():
        train_blob = torch.load(train_cache)
        eval_blob = torch.load(eval_cache)
        return (train_blob["features"], train_blob["labels"]), (eval_blob["features"], eval_blob["labels"])

    bundle = build_dataset_bundle(dataset_name)
    encoder, _ = build_pretrained_encoder(dataset_name, device)
    train_features, train_labels = _extract_features_for_dataset(bundle.train_dataset, encoder, device)
    eval_features, eval_labels = _extract_features_for_dataset(bundle.eval_dataset, encoder, device)
    torch.save({"features": train_features, "labels": train_labels}, train_cache)
    torch.save({"features": eval_features, "labels": eval_labels}, eval_cache)
    return (train_features, train_labels), (eval_features, eval_labels)


def mine_neighbors(features: torch.Tensor, topk: int) -> np.ndarray:
    x = F.normalize(features.float(), dim=1).cpu().numpy()
    nn_model = NearestNeighbors(n_neighbors=topk + 1, metric="cosine")
    nn_model.fit(x)
    indices = nn_model.kneighbors(x, return_distance=False)
    return indices[:, 1:]


def _mean_cluster_distributions(probs: torch.Tensor, assignments: torch.Tensor, k: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for cluster_id in range(k):
        mask = assignments == cluster_id
        if int(mask.sum()) == 0:
            rows.append(torch.full((k,), 1.0 / k, device=probs.device, dtype=probs.dtype))
        else:
            dist = probs[mask].mean(dim=0)
            dist = dist / torch.clamp(dist.sum(), min=1e-8)
            rows.append(dist)
    return torch.stack(rows, dim=0)


def _split_candidate_js(features: torch.Tensor, probs: torch.Tensor, random_state: int = 42) -> float:
    if features.size(0) < 4:
        return 0.0
    x = features.cpu().numpy()
    km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = km.fit_predict(x)
    if len(np.unique(labels)) < 2:
        return 0.0
    p0 = probs[torch.from_numpy(labels == 0).to(probs.device)]
    p1 = probs[torch.from_numpy(labels == 1).to(probs.device)]
    if p0.size(0) == 0 or p1.size(0) == 0:
        return 0.0
    c0 = torch.from_numpy(x[labels == 0].mean(axis=0)).float().to(probs.device)
    c1 = torch.from_numpy(x[labels == 1].mean(axis=0)).float().to(probs.device)
    d0 = F.softmax(c0, dim=0)
    d1 = F.softmax(c1, dim=0)
    return float(compute_js_divergence(d0, d1).item())


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


def _pnp_step(
    head: DynamicClusterHead,
    features: torch.Tensor,
    lambda_param: float,
    seed: int,
    initial_k: int,
    enable_split: bool,
    enable_merge: bool,
    enable_split_bootstrap: bool,
) -> bool:
    head.eval()
    with torch.no_grad():
        logits = head(features)
        probs = F.softmax(logits, dim=1)
        assignments = probs.argmax(dim=1)
        cluster_dists = _mean_cluster_distributions(probs, assignments, head.k)
        js_mat = pairwise_js_matrix(cluster_dists)
        split_threshold = compute_split_threshold(js_mat, lambda_param, head.k)

        best_split_cluster = None
        best_split_score = -math.inf
        best_split_centroids: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
        for cluster_id in range(head.k):
            mask = assignments == cluster_id
            if int(mask.sum()) < 4:
                continue
            js_local = _split_candidate_js(features[mask], probs[mask], random_state=seed)
            if js_local > split_threshold and js_local > best_split_score:
                best_split_score = js_local
                best_split_cluster = cluster_id
                best_split_centroids = _kmeans_subcluster_centroids(features[mask], random_state=seed)

        split_allowed = enable_split and (head.k <= initial_k or best_split_score > 1.5 * split_threshold)
        if best_split_cluster is not None and split_allowed:
            c0, c1 = best_split_centroids
            if c0 is not None and c1 is not None:
                head.split_cluster_with_centroids(int(best_split_cluster), c0, c1)
            else:
                head.split_cluster(int(best_split_cluster))
            return True

        # Fallback: split the most dispersed cluster in feature space when the JS
        # criterion is too conservative for a frozen-feature local setup.
        if enable_split and head.k <= initial_k:
            global_var = float(features.var(dim=0).mean().item())
            best_var_cluster = None
            best_var_value = -math.inf
            best_var_centroids: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
            min_cluster_size = max(64, features.size(0) // max(2 * head.k, 1))
            for cluster_id in range(head.k):
                mask = assignments == cluster_id
                cluster_size = int(mask.sum())
                if cluster_size < min_cluster_size:
                    continue
                cluster_var = float(features[mask].var(dim=0).mean().item())
                if cluster_var > best_var_value:
                    best_var_value = cluster_var
                    best_var_cluster = cluster_id
                    best_var_centroids = _kmeans_subcluster_centroids(features[mask], random_state=seed)
            if best_var_cluster is not None and best_var_value > 0.5 * global_var:
                c0, c1 = best_var_centroids
                if c0 is not None and c1 is not None:
                    head.split_cluster_with_centroids(int(best_var_cluster), c0, c1)
                else:
                    head.split_cluster(int(best_var_cluster))
                return True

            # Bootstrap split for severely under-initialized runs.
            if enable_split_bootstrap and head.k <= 5:
                counts = torch.bincount(assignments, minlength=head.k)
                bootstrap_cluster = int(torch.argmax(counts).item())
                mask = assignments == bootstrap_cluster
                c0, c1 = _kmeans_subcluster_centroids(features[mask], random_state=seed)
                if c0 is not None and c1 is not None:
                    head.split_cluster_with_centroids(bootstrap_cluster, c0, c1)
                else:
                    head.split_cluster(bootstrap_cluster)
                return True

        if not enable_merge or head.k <= 2:
            return False

        merge_threshold = compute_merge_threshold(js_mat, lambda_param, head.k)
        masked = js_mat.clone()
        masked.fill_diagonal_(float("inf"))
        min_value = float(masked.min().item())
        if min_value < merge_threshold:
            flat_idx = int(masked.argmin().item())
            i = flat_idx // masked.size(1)
            j = flat_idx % masked.size(1)
            head.merge_clusters(i, j)
            return True

    return False


def train_scan_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    neighbor_indices: np.ndarray,
    k0: int,
    method_name: str,
    epochs: int = 15,
    batch_size: int = 1024,
    lr: float = 1e-2,
    entropy_weight: float = 5.0,
    lambda_param: float = 2.0,
    warmup_epochs: int = 5,
    enable_pnp: bool = False,
    enable_split: bool = True,
    enable_merge: bool = True,
    enable_split_bootstrap: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> ExperimentResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    x_train = train_features.to(device).float()
    y_train = train_labels.cpu().numpy()
    x_eval = eval_features.to(device).float()
    y_eval = eval_labels.cpu().numpy()
    head = DynamicClusterHead(input_dim=x_train.size(1), k=k0).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = SCANLoss(entropy_weight=entropy_weight).to(device)

    n = x_train.size(0)
    anchor_indices = np.arange(n)
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

        if enable_pnp and epoch >= warmup_epochs:
            changed = _pnp_step(
                head,
                x_train,
                lambda_param=lambda_param,
                seed=seed + epoch,
                initial_k=k0,
                enable_split=enable_split,
                enable_merge=enable_merge,
                enable_split_bootstrap=enable_split_bootstrap,
            )
            if changed:
                optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

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
