import random

import numpy as np
import torch

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


def relative_deviation(observed, reference, eps=1e-8):
    """
    Compute absolute relative deviation in percent.

    Args:
        observed: Value from the current implementation.
        reference: Value reported in the paper or chosen baseline.
        eps: Numerical safeguard for zero references.
    """
    observed = float(observed)
    reference = float(reference)
    return abs(observed - reference) / max(abs(reference), eps) * 100.0
