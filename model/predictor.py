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

    def __init__(self, dim: int, num_heads: int = 16, dim_head: int = 12, dropout: float = 0.0) -> None:
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

        qkv = self.to_qkv(x)  # (B, T, 3 * num_heads * dim_head)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape: (B, T, num_heads, dim_head) → (B, num_heads, T, dim_head)
        q = q.reshape(B, T, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)

        # (B, num_heads, T, dim_head) → (B, T, num_heads * dim_head)
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
    """Transformer block with AdaLN conditioning on actions.

    AdaLN produces per-layer shift, scale, and gate parameters from the
    action embedding, modulating both attention and MLP sub-layers.
    """

    def __init__(self, dim: int, num_heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

        # AdaLN modulation: produces 6 parameter sets (shift/scale/gate for attn + MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # Initialize gate biases to zero (residual at init = identity)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — token embeddings
            c: (B, T, D) — action conditioning signal

        Returns:
            (B, T, D) — updated token embeddings
        """
        # Get modulation parameters from action conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Attention with AdaLN
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # MLP with AdaLN
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class ActionEncoder(nn.Module):
    """Encode discrete actions for all 6 players into a single embedding.

    Each player's action (0-11) is embedded independently, then all 6
    embeddings are concatenated and projected to the model dimension.
    """

    def __init__(self, num_actions: int = 12, num_players: int = 6, embed_dim: int = 192) -> None:
        super().__init__()
        per_player_dim = embed_dim // num_players  # 32
        self.player_embed = nn.Embedding(num_actions, per_player_dim)
        self.project = nn.Linear(num_players * per_player_dim, embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T, 6) integer action IDs per player.

        Returns:
            (B, T, embed_dim) action embeddings.
        """
        B, T, P = actions.shape
        per_player = self.player_embed(actions)  # (B, T, 6, 32)
        flat = per_player.reshape(B, T, -1)      # (B, T, 192)
        return self.project(flat)                  # (B, T, 192)


class ARPredictor(nn.Module):
    """Autoregressive predictor with AdaLN action conditioning.

    Given context embeddings and actions, predicts next-frame embeddings.

    Architecture:
        - Action encoder: per-player embeddings → projected to 192-dim
        - Positional embeddings added to input
        - 6 ConditionalBlock layers with AdaLN
        - Output projector: LayerNorm + Linear

    ~10M parameters.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 16,
        dim_head: int = 12,
        mlp_dim: int = 2048,
        max_seq_len: int = 16,
        num_actions: int = 12,
        num_players: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Action encoder
        self.action_encoder = ActionEncoder(num_actions, num_players, embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            ConditionalBlock(embed_dim, num_heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ctx_emb: torch.Tensor, ctx_actions: torch.Tensor) -> torch.Tensor:
        """Predict next-frame embeddings.

        Args:
            ctx_emb: (B, T, D) — context frame embeddings.
            ctx_actions: (B, T, 6) — context actions (int64).

        Returns:
            (B, T, D) — predicted next-frame embeddings.
        """
        B, T, D = ctx_emb.shape

        # Encode actions
        act_emb = self.action_encoder(ctx_actions)  # (B, T, D)

        # Add positional embeddings
        x = ctx_emb + self.pos_embed[:, :T]
        x = self.dropout(x)

        # Apply conditional transformer blocks
        for block in self.blocks:
            x = block(x, act_emb)

        # Output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        return x
