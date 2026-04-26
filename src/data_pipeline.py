from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from src.scan_datasets import build_dataset_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "Unsupervised-Classification"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from models.resnet import resnet50  # type: ignore  # noqa: E402
from models.resnet_cifar import resnet18 as cifar_resnet18  # type: ignore  # noqa: E402
from models.resnet_stl import resnet18 as stl_resnet18  # type: ignore  # noqa: E402


SCAN_PRETEXT_DIR = PROJECT_ROOT / "data" / "scan_pretext"
SCAN_CACHE_DIR = PROJECT_ROOT / "data" / "scan_cache"
MOCO_CKPT = PROJECT_ROOT / "data" / "checkpoints" / "moco_v2_800ep_pretrain.pth.tar"


def _strip_prefix_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
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

    if dataset_name in {"imagenet-10", "beans"}:
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
def _extract_features_for_dataset(
    dataset,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def load_or_extract_features(
    dataset_name: str,
    device: torch.device,
    force_recompute: bool = False,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
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
