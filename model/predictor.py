"""Autoregressive Transformer predictor with AdaLN action conditioning.

Predicts next-frame embeddings given context embeddings and actions.
Uses Adaptive Layer Normalization (AdaLN) to condition on actions at every layer.

Reference: LeWorldModel (https://github.com/lucas-maes/le-wm)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention with causal masking support."""

    def __init__(self, dim: int, num_heads: int = 16, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D) with causal masking."""
        B, T, _ = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.dim_head).transpose(1, 2)

        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN conditioning on actions."""

    def __init__(self, dim: int, num_heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # Zero init both weight and bias (matches LeWM)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ActionEncoder(nn.Module):
    """Encode a single discrete action into a conditioning embedding."""

    def __init__(self, num_actions: int = 12, embed_dim: int = 192) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, embed_dim)
        self.project = nn.Linear(embed_dim, embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T) integer action ID for the controlled player.

        Returns:
            (B, T, embed_dim) conditioning embedding.
        """
        return self.project(self.action_embed(actions))


class ARPredictor(nn.Module):
    """Autoregressive predictor with AdaLN action conditioning.

    Given context embeddings and a single player's actions, predicts
    next-frame embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        max_seq_len: int = 16,
        num_actions: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.action_encoder = ActionEncoder(num_actions, embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ConditionalBlock(embed_dim, num_heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ctx_emb: torch.Tensor, ctx_actions: torch.Tensor) -> torch.Tensor:
        """Predict next-frame embeddings.

        Args:
            ctx_emb: (B, T, D) — context frame embeddings.
            ctx_actions: (B, T) — controlled player's action (int64).

        Returns:
            (B, T, D) — predicted next-frame embeddings.
        """
        B, T, D = ctx_emb.shape

        act_emb = self.action_encoder(ctx_actions)  # (B, T, D)

        x = ctx_emb + self.pos_embed[:, :T]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, act_emb)

        x = self.output_norm(x)
        x = self.output_proj(x)

        return x
