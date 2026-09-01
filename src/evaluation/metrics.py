"""Uncertainty metrics."""
import math
from typing import List


def semantic_entropy(clusters: List[List[str]], n_total: int) -> float:
    """Discrete semantic entropy (nats). Farquhar et al. (2024), Eq. 3."""
    entropy = 0.0
    for cluster in clusters:
        p_k = len(cluster) / n_total
        if p_k > 0:
            entropy -= p_k * math.log(p_k)
    return entropy
