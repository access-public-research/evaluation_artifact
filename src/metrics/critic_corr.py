import numpy as np
from scipy.stats import spearmanr


def critic_informativeness(proxy_scores, oracle_scores):
    """Compute Spearman correlation across seeds per epoch."""
    rho, _ = spearmanr(proxy_scores, oracle_scores)
    return float(rho)
