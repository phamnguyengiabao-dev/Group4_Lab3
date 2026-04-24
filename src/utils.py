import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model import CustomPnPModule, pnp_loss
from src.metrics import evaluate_clustering

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

def train_clustering(features, labels, k_max, device, apply_sparsity=True, epochs=50, lr=0.05):
    """Huấn luyện Module PnP và trả về kết quả"""
    model = CustomPnPModule(feature_dim=features.shape[1], k_max=k_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_z, _ in loader:
            batch_z = batch_z.to(device)
            optimizer.zero_grad()
            p, dist = model(batch_z)
            loss = pnp_loss(p, dist, apply_sparsity=apply_sparsity)
            loss.backward()
            optimizer.step()
            
    # Đánh giá sau khi train xong
    model.eval()
    with torch.no_grad():
        p_final, _ = model(features.to(device))
        predicted_labels = torch.argmax(p_final, dim=1).cpu().numpy()
        true_labels = labels.numpy()
        
        active_k = len(np.unique(predicted_labels)) 
        acc, nmi, ari = evaluate_clustering(true_labels, predicted_labels)
        
    return active_k, acc, nmi, ari