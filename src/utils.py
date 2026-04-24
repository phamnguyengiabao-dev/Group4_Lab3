import torch
import random
import numpy as np
from src.model import BaseClusteringNet
from src.metrics import evaluate_clustering
from src.pnp_trainer import DeepPlugAndPlayTrainer

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

def load_feature_file(feature_path, label_path, device=None):
    """Load pre-extracted features/labels from .npy files."""
    features = np.load(feature_path)
    labels = np.load(label_path)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long).reshape(-1)
    if device is not None:
        features = features.to(device)
        labels = labels.to(device)
    return features, labels

def train_clustering(
    features,
    labels,
    k_max,
    device,
    apply_sparsity=True,
    epochs=50,
    lr=0.05,
    lambda_param=2.0,
    gamma=0.1,
    warmup_epochs=20,
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
        gamma=gamma if apply_sparsity else 0.0,
        warmup_epochs=warmup_epochs,
        lr=lr,
        batch_size=256,
        epochs=epochs,
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
