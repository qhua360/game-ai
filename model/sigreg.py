"""SIGReg: Sketch Isotropic Gaussian Regularizer.

Enforces that latent embeddings are distributed as isotropic Gaussians,
preventing representation collapse without stop-gradient or EMA.

Implements the Epps-Pulley statistical Gaussianity test via random projections.
Reference: LeWorldModel (https://github.com/lucas-maes/le-wm)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Tests whether embeddings are Gaussian-distributed by comparing their
    empirical characteristic function against the Gaussian characteristic
    function along random projection directions.

    Args:
        knots: Number of quadrature points for the Epps-Pulley test.
        num_proj: Number of random projection directions.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj

        # Quadrature points from 0 to 3 standard deviations
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # Trapezoidal quadrature weights
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt

        # Gaussian window: exp(-t²/2) — the characteristic function of N(0,1)
        phi = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Compute the SIGReg loss.

        Args:
            emb: Embeddings of shape (T, B, D) — time-first.

        Returns:
            Scalar loss value.
        """
        T, B, D = emb.shape

        # Random projection matrix, L2-normalized columns
        A = torch.randn(D, self.num_proj, device=emb.device, dtype=emb.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Project embeddings: (T, B, D) @ (D, num_proj) → (T, B, num_proj)
        proj = emb @ A

        # Evaluate characteristic function at quadrature points
        # (T, B, num_proj, 1) * (knots,) → (T, B, num_proj, knots)
        x_t = proj.unsqueeze(-1) * self.t

        # Empirical characteristic function: mean over samples (dim=0 is T*B combined)
        # Reshape to (T*B, num_proj, knots) for averaging
        x_t_flat = x_t.reshape(T * B, self.num_proj, -1)
        cos_mean = x_t_flat.cos().mean(dim=0)  # (num_proj, knots)
        sin_mean = x_t_flat.sin().mean(dim=0)  # (num_proj, knots)

        # Squared error between empirical and Gaussian characteristic functions
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted sum using trapezoidal quadrature with Gaussian window
        statistic = (err * self.weights).sum(dim=-1)  # (num_proj,)

        # Scale by number of samples and average over projections
        return (statistic * (T * B)).mean()
