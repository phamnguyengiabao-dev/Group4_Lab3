import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClusteringNet(nn.Module):
    """
    2-layer MLP clustering network with dynamic split/merge on output neurons.
    """

    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        super().__init__()
        if k < 2:
            raise ValueError("k must be >= 2.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = k

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, k)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits = self.output_layer(h)
        probs = F.softmax(logits, dim=1)
        return probs, logits

    def _rebuild_output_layer(self, new_weight: torch.Tensor, new_bias: torch.Tensor):
        new_k = new_weight.size(0)
        new_layer = nn.Linear(self.hidden_dim, new_k, bias=True).to(new_weight.device)
        with torch.no_grad():
            new_layer.weight.copy_(new_weight)
            new_layer.bias.copy_(new_bias)
        self.output_layer = new_layer
        self.K = new_k

    def split_cluster(self, k_index: int, noise_std: float = 1e-3):
        """
        Split one output neuron into two by copying weights with small random noise.
        """
        if not (0 <= k_index < self.K):
            raise IndexError("k_index out of range.")
        w = self.output_layer.weight.data
        b = self.output_layer.bias.data

        duplicated_w = w[k_index].clone()
        duplicated_b = b[k_index].clone()
        noise = torch.randn_like(duplicated_w) * noise_std

        new_weight = torch.cat([w, (duplicated_w + noise).unsqueeze(0)], dim=0)
        new_bias = torch.cat([b, duplicated_b.unsqueeze(0)], dim=0)
        self._rebuild_output_layer(new_weight, new_bias)

    def merge_clusters(self, k1_index: int, k2_index: int):
        """
        Merge two output neurons by averaging weights and biases.
        """
        if k1_index == k2_index:
            raise ValueError("k1_index and k2_index must be different.")
        if not (0 <= k1_index < self.K and 0 <= k2_index < self.K):
            raise IndexError("cluster index out of range.")

        i, j = sorted((k1_index, k2_index))
        w = self.output_layer.weight.data
        b = self.output_layer.bias.data

        merged_w = 0.5 * (w[i] + w[j])
        merged_b = 0.5 * (b[i] + b[j])

        keep_mask = torch.ones(self.K, dtype=torch.bool, device=w.device)
        keep_mask[i] = False
        keep_mask[j] = False

        remaining_w = w[keep_mask]
        remaining_b = b[keep_mask]
        new_weight = torch.cat([remaining_w, merged_w.unsqueeze(0)], dim=0)
        new_bias = torch.cat([remaining_b, merged_b.unsqueeze(0)], dim=0)
        self._rebuild_output_layer(new_weight, new_bias)

    def reset_optimizer(self, lr: float = 1e-3, weight_decay: float = 0.0):
        """
        Create a fresh optimizer after topology changes.
        """
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


def clustering_loss(probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Entropy-minimization objective to sharpen assignments.
    """
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
    l2 = 1e-4 * (logits ** 2).mean()
    return entropy + l2
