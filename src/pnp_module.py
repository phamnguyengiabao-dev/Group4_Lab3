from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from src.divergence import (
    compute_js_divergence,
    compute_merge_threshold,
    compute_split_threshold,
    pairwise_js_matrix,
)
from src.model import BaseClusteringNet, clustering_loss


def _cluster_distributions(probs: torch.Tensor, assignments: torch.Tensor, k: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for cluster_id in range(k):
        mask = assignments == cluster_id
        if mask.sum() == 0:
            rows.append(torch.full((k,), 1.0 / k, device=probs.device))
        else:
            dist = probs[mask].mean(dim=0)
            dist = dist / (dist.sum() + 1e-8)
            rows.append(dist)
    return torch.stack(rows, dim=0)


def _split_candidate_js(features: torch.Tensor) -> float:
    if features.size(0) < 4:
        return 0.0

    x = features.detach().cpu().numpy()
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    sub_labels = km.fit_predict(x)
    if (sub_labels == 0).sum() == 0 or (sub_labels == 1).sum() == 0:
        return 0.0

    c0 = torch.from_numpy(x[sub_labels == 0].mean(axis=0)).float()
    c1 = torch.from_numpy(x[sub_labels == 1].mean(axis=0)).float()
    p0 = torch.softmax(c0, dim=0)
    p1 = torch.softmax(c1, dim=0)
    return float(compute_js_divergence(p0, p1).item())


class DeepPlugAndPlayTrainer:
    def __init__(
        self,
        model: BaseClusteringNet,
        lambda_param: float = 2.0,
        gamma: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
        train_epochs_per_cycle: int = 20,
        max_cycles: int = 10,
        enable_split: bool = True,
        enable_merge: bool = True,
    ):
        self.model = model
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.train_epochs_per_cycle = train_epochs_per_cycle
        self.max_cycles = max_cycles
        self.enable_split = enable_split
        self.enable_merge = enable_merge
        self.optimizer = self.model.reset_optimizer(lr=self.lr)

    def _train_epochs(self, features: torch.Tensor):
        loader = DataLoader(TensorDataset(features), batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.train_epochs_per_cycle):
            for (x_batch,) in loader:
                x_batch = x_batch.to(next(self.model.parameters()).device)
                probs, logits = self.model(x_batch)
                loss = clustering_loss(probs, logits)
                if self.gamma > 0:
                    cluster_prior = probs.mean(dim=0)
                    prior_entropy = -(cluster_prior * torch.log(cluster_prior + 1e-8)).sum()
                    loss = loss - self.gamma * prior_entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def _infer(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        x = features.to(next(self.model.parameters()).device)
        probs, _ = self.model(x)
        assignments = probs.argmax(dim=1)
        return probs, assignments

    def fit(self, features: torch.Tensor) -> Dict[str, List[int]]:
        history: Dict[str, List[int]] = {"k": [self.model.K]}

        for _ in range(self.max_cycles):
            changed = False
            self._train_epochs(features)
            probs, assignments = self._infer(features)
            cluster_dists = _cluster_distributions(probs, assignments, self.model.K)
            js_mat = pairwise_js_matrix(cluster_dists)
            t_s = compute_split_threshold(js_mat, self.lambda_param, self.model.K)

            if self.enable_split:
                for cluster_id in range(self.model.K):
                    mask = assignments == cluster_id
                    if mask.sum() == 0:
                        continue
                    js_local = _split_candidate_js(features[mask.cpu()])
                    if js_local > t_s:
                        self.model.split_cluster(cluster_id)
                        self.optimizer = self.model.reset_optimizer(lr=self.lr)
                        changed = True
                        break

            if not changed and self.enable_merge:
                probs, assignments = self._infer(features)
                cluster_dists = _cluster_distributions(probs, assignments, self.model.K)
                js_mat = pairwise_js_matrix(cluster_dists)
                t_m = compute_merge_threshold(js_mat, self.lambda_param, self.model.K)

                if self.model.K > 2:
                    mask = torch.ones_like(js_mat, dtype=torch.bool)
                    mask.fill_diagonal_(False)
                    candidate_values = js_mat[mask]
                    if candidate_values.numel() > 0:
                        min_js = float(candidate_values.min().item())
                        if min_js < t_m:
                            tmp = js_mat.clone()
                            tmp.fill_diagonal_(float("inf"))
                            flat_idx = int(tmp.argmin().item())
                            i = flat_idx // tmp.size(1)
                            j = flat_idx % tmp.size(1)
                            self.model.merge_clusters(i, j)
                            self.optimizer = self.model.reset_optimizer(lr=self.lr)
                            changed = True

            history["k"].append(self.model.K)
            if not changed:
                break

        return history
