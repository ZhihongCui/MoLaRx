"""
MMD (Maximum Mean Discrepancy) Loss with multi-scale RBF kernel.
================================================================

Used in Phase 2 for domain alignment: forces unfrozen layers' representations
to be indistinguishable between Singapore and Boston.

L_DA = MMD(h_phi_low(Singapore), h_phi_low(Boston))
"""

import torch
import torch.nn as nn


def mmd_rbf(x: torch.Tensor, y: torch.Tensor,
            bandwidths: list = None) -> torch.Tensor:
    """
    Compute MMD^2 with multi-scale RBF (Gaussian) kernel.

    Args:
        x: [N, D] samples from source distribution (Singapore)
        y: [M, D] samples from target distribution (Boston)
        bandwidths: list of bandwidth values for multi-scale kernel

    Returns:
        MMD^2 (scalar tensor, differentiable)
    """
    if bandwidths is None:
        bandwidths = [0.1, 0.5, 1.0, 5.0, 10.0]

    # Pairwise squared distances
    xx = torch.cdist(x, x, p=2).pow(2)  # [N, N]
    yy = torch.cdist(y, y, p=2).pow(2)  # [M, M]
    xy = torch.cdist(x, y, p=2).pow(2)  # [N, M]

    mmd = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    for bw in bandwidths:
        gamma = 1.0 / (2.0 * bw * bw)
        k_xx = torch.exp(-gamma * xx).mean()
        k_yy = torch.exp(-gamma * yy).mean()
        k_xy = torch.exp(-gamma * xy).mean()
        mmd = mmd + (k_xx + k_yy - 2.0 * k_xy)

    return mmd


def mmd_rbf_adaptive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    MMD with adaptive bandwidth based on median heuristic.

    More robust than fixed bandwidths — automatically adapts to
    the scale of the hidden state representations.
    """
    # Compute all pairwise distances
    all_samples = torch.cat([x, y], dim=0)
    dists = torch.cdist(all_samples, all_samples, p=2)

    # Median heuristic for bandwidth selection
    median_dist = torch.median(dists[dists > 0]).item()
    if median_dist < 1e-8:
        median_dist = 1.0

    # Use bandwidths around the median
    bandwidths = [median_dist * f for f in [0.1, 0.5, 1.0, 2.0, 5.0]]
    return mmd_rbf(x, y, bandwidths)


class MMDLoss(nn.Module):
    """MMD loss module for use in training loops."""

    def __init__(self, bandwidths=None, adaptive=False):
        super().__init__()
        self.bandwidths = bandwidths or [0.1, 0.5, 1.0, 5.0, 10.0]
        self.adaptive = adaptive

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.adaptive:
            return mmd_rbf_adaptive(x, y)
        return mmd_rbf(x, y, self.bandwidths)
