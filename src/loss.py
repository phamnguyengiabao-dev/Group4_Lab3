import torch
import torch.nn.functional as F


EPS = 1e-8


def soft_assignments_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Stable softmax-based cluster assignments Q.

    Args:
        logits: Tensor [N, K] from the clustering head.

    Returns:
        Tensor [N, K] with rows summing to 1.
    """
    q = F.softmax(logits, dim=1)
    q = torch.clamp(q, min=EPS)
    q = q / torch.clamp(q.sum(dim=1, keepdim=True), min=EPS)
    return q


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    DEC target distribution:
        p_ij = (q_ij^2 / f_j) / sum_j (q_ij^2 / f_j)

    Args:
        q: Soft assignments with shape [N, K].

    Returns:
        Sharpened target distribution with shape [N, K].
    """
    q = torch.clamp(q, min=EPS)
    freq = torch.clamp(q.sum(dim=0, keepdim=True), min=EPS)
    weight = (q ** 2) / freq
    p = weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=EPS)
    return p


def dec_loss(q: torch.Tensor) -> torch.Tensor:
    """
    DEC objective using KL(P || Q) in PyTorch's kl_div convention.

    Args:
        q: Soft assignments with shape [N, K].

    Returns:
        Scalar DEC clustering loss.
    """
    q = torch.clamp(q, min=EPS)
    p = target_distribution(q).detach()
    return F.kl_div(q.log(), p, reduction="batchmean")


def s_loss(q: torch.Tensor) -> torch.Tensor:
    """
    Entropy-based regularizer used as a simple sharpening penalty.

    Lower values indicate crisper assignments.
    """
    q = torch.clamp(q, min=EPS)
    return -(q * q.log()).sum(dim=1).mean()


def base_clustering_loss(logits: torch.Tensor = None, probs: torch.Tensor = None) -> torch.Tensor:
    """
    Convenience wrapper for callers that have either logits or probabilities.
    """
    if probs is None:
        if logits is None:
            raise ValueError("Either logits or probs must be provided.")
        probs = soft_assignments_from_logits(logits)
    return dec_loss(probs)
