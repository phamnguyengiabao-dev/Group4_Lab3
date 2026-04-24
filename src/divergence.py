import torch


EPS = 1e-8


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p: Tensor [..., C], probabilities (will be re-normalized for safety).
        q: Tensor [..., C], probabilities (will be re-normalized for safety).
        eps: Numerical stability constant.

    Returns:
        Tensor [...] containing JS(P || Q).
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def pairwise_js_matrix(cluster_distributions: torch.Tensor) -> torch.Tensor:
    """
    Build pairwise JS divergence matrix for cluster-level distributions.

    Args:
        cluster_distributions: Tensor [K, C], where each row is a probability distribution.

    Returns:
        Tensor [K, K], symmetric matrix of pairwise JS divergences.
    """
    k = cluster_distributions.size(0)
    js_mat = torch.zeros((k, k), dtype=cluster_distributions.dtype, device=cluster_distributions.device)

    for i in range(k):
        for j in range(i + 1, k):
            js_ij = compute_js_divergence(cluster_distributions[i], cluster_distributions[j])
            js_mat[i, j] = js_ij
            js_mat[j, i] = js_ij
    return js_mat


def compute_split_threshold(js_matrix: torch.Tensor, lambda_param: float, k: int) -> float:
    """
    Compute split threshold T_s from paper formula:
    T_s = lambda / (2K(lambda + K + 1)) * sum_{k1,k2} JS(k1 || k2)
    """
    if k <= 0:
        raise ValueError("k must be positive for split threshold.")

    total_js = float(js_matrix.sum().item())
    denom = 2.0 * k * (lambda_param + k + 1.0)
    return (lambda_param / denom) * total_js


def compute_merge_threshold(js_matrix: torch.Tensor, lambda_param: float, k: int) -> float:
    """
    Compute merge threshold T_m from paper formula:
    T_m = lambda / (2(K-1)(lambda + K)) * sum_{k1,k2} JS(k1 || k2)
    """
    if k <= 1:
        return 0.0

    total_js = float(js_matrix.sum().item())
    denom = 2.0 * (k - 1.0) * (lambda_param + k)
    return (lambda_param / denom) * total_js
