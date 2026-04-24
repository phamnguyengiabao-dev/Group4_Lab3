import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from src.model import BaseClusteringNet
from src.metrics import evaluate_clustering
from src.pnp_module import DeepPlugAndPlayTrainer

def set_seed(seed=42):
    """Cố định seed để đảm bảo tính tái lập (Reproducibility)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Kiểm tra và trả về thiết bị khả dụng"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(dataloader, backbone, device):
    """Trích xuất vector Z trước để tiết kiệm thời gian train"""
    backbone.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            z = backbone(imgs)
            z = z.view(z.size(0), -1)
            features.append(z.cpu())
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

def train_clustering(
    features,
    labels,
    k_max,
    device,
    apply_sparsity=True,
    epochs=50,
    lr=0.05,
    lambda_param=2.0,
    enable_split=True,
    enable_merge=True,
):
    """
    Train Deep Plug-and-Play clustering and return (K*, ACC, NMI, ARI).

    Notes:
        - `apply_sparsity=False` is mapped to disabling split encouragement (gamma=0),
          useful for a lightweight ablation with old notebooks.
    """
    x = features.to(device)
    model = BaseClusteringNet(input_dim=x.shape[1], hidden_dim=256, k=k_max).to(device)
    trainer = DeepPlugAndPlayTrainer(
        model=model,
        lambda_param=lambda_param,
        gamma=0.1 if apply_sparsity else 0.0,
        lr=lr,
        batch_size=256,
        train_epochs_per_cycle=max(5, epochs // 5),
        max_cycles=5,
        enable_split=enable_split,
        enable_merge=enable_merge,
    )
    _ = trainer.fit(x)

    model.eval()
    with torch.no_grad():
        probs, _ = model(x)
        predicted_labels = torch.argmax(probs, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy().reshape(-1)
    active_k = len(np.unique(predicted_labels))
    acc, nmi, ari = evaluate_clustering(true_labels, predicted_labels)
    return active_k, acc, nmi, ari
