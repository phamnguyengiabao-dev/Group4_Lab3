import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class BaseClusteringNet(nn.Module):
    """
    Base clustering network for the two-step PnP pipeline.

    The network only consumes tabular features with shape [N, D]. It learns a
    latent representation with a single hidden layer and exposes dynamic output
    topology so the PnP module can split or merge clusters during training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        super().__init__()
        if k < 2:
            raise ValueError("k must be >= 2.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = k

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, k)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        logits = self.output_layer(h)
        probs = F.softmax(logits, dim=1)
        return probs, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return latent representation before output layer.
        """
        return self.encoder(x)

    @torch.no_grad()
    def init_weights_kmeans(self, x: torch.Tensor, random_state: int = 42, n_init: int = 10):
        """
        Initialize the output layer with K-Means centroids computed in latent space.

        Args:
            x: Input feature tensor with shape [N, D].
            random_state: Reproducible seed for K-Means.
            n_init: Number of K-Means restarts.
        """
        if x.ndim != 2:
            raise ValueError("x must have shape [N, D].")
        if x.size(0) < self.K:
            raise ValueError("Need at least K samples for K-Means initialization.")

        device = next(self.parameters()).device
        latent = self.encode(x.to(device)).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.K, random_state=random_state, n_init=n_init)
        kmeans.fit(latent)

        centroids = torch.from_numpy(kmeans.cluster_centers_).to(
            device=device,
            dtype=self.output_layer.weight.dtype,
        )
        self.output_layer.weight.copy_(centroids)
        self.output_layer.bias.zero_()

    def _rebuild_output_layer(self, new_weight: torch.Tensor, new_bias: torch.Tensor):
        new_k = new_weight.size(0)
        new_layer = nn.Linear(self.hidden_dim, new_k, bias=True).to(new_weight.device)
        with torch.no_grad():
            new_layer.weight.copy_(new_weight)
            new_layer.bias.copy_(new_bias)
        self.output_layer = new_layer
        self.K = new_k

    def split_cluster(self, k_index: int, noise_std: float = 1e-2):
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
