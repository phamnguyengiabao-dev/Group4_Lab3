import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomPnPModule(nn.Module):
    """Module phân cụm tự code. Mô phỏng cơ chế Split-and-Merge."""
    def __init__(self, feature_dim=512, k_max=30):
        super().__init__()
        self.k_max = k_max
        # Ma trận tâm cụm (K_max, Dim) được cập nhật thông qua Backprop
        self.centers = nn.Parameter(torch.randn(k_max, feature_dim))

    def forward(self, z, temperature=1.0):
        # Tính bình phương khoảng cách Euclidean từ vector đặc trưng z đến các tâm cụm
        dist = torch.cdist(z, self.centers) ** 2
        # Soft-assignment (Xác suất điểm dữ liệu thuộc về cụm k)
        p = F.softmax(-dist / temperature, dim=1)
        return p, dist

def pnp_loss(p, dist, lambda_sparsity=1.5, apply_sparsity=True):
    """Hàm Loss: Clustering Loss + Sparsity Regularization"""
    # 1. Kéo dữ liệu về gần tâm cụm (Đảm bảo tính Compactness)
    cluster_loss = torch.mean(torch.sum(p * dist, dim=1))
    
    # 2. Sparsity Loss: Ép các cụm dư thừa biến mất (Pruning/Merge)
    if apply_sparsity:
        cluster_weights = torch.mean(p, dim=0) 
        sparsity_loss = lambda_sparsity * torch.sum(torch.sqrt(cluster_weights + 1e-8)) 
        return cluster_loss + sparsity_loss
        
    return cluster_loss