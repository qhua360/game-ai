"""JEPA World Model for ice hockey.

Combines ViT-Tiny encoder (via stable-pretraining), AR predictor with AdaLN,
and SIGReg regularization. Training is handled externally by spt.Module.

Reference: LeWorldModel (https://github.com/lucas-maes/le-wm)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.encoder import ViTTinyEncoder
from model.predictor import ARPredictor


class LeWM(nn.Module):
    """JEPA World Model.

    Learns to predict future latent embeddings from current observations
    and actions. The encoder maps frames to a compact latent space, and
    the predictor forecasts next-frame embeddings conditioned on actions.

    Training loss computation is handled by the forward function passed
    to spt.Module (see model/train.py), not by this class.

    Architecture:
        - Encoder: ViT-Tiny via spt (~5M params) — 84x84 RGB → 192-dim
        - Predictor: 6-layer Transformer with AdaLN (~10M params)
        - Total: ~15M parameters
    """

    def __init__(
        self,
        embed_dim: int = 192,
        history_size: int = 3,
        num_preds: int = 1,
        # Encoder params
        encoder_scale: str = "tiny",
        img_size: int = 84,
        patch_size: int = 14,
        # Predictor params
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 12,
        predictor_mlp_dim: int = 2048,
        predictor_dropout: float = 0.1,
        # Action params
        num_actions: int = 12,
        num_players: int = 6,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.history_size = history_size
        self.num_preds = num_preds

        # Encoder (via stable-pretraining vit_hf)
        self.encoder = ViTTinyEncoder(
            encoder_scale=encoder_scale,
            image_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Predictor
        self.predictor = ARPredictor(
            embed_dim=embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            num_actions=num_actions,
            num_players=num_players,
            dropout=predictor_dropout,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of observations.

        Args:
            obs: (B, T, C, H, W) float32 images [0, 1].

        Returns:
            (B, T, embed_dim) embeddings.
        """
        return self.encoder.encode_sequence(obs)

    def predict(self, ctx_emb: torch.Tensor, ctx_actions: torch.Tensor) -> torch.Tensor:
        """Predict next-frame embeddings.

        Args:
            ctx_emb: (B, T, D) context embeddings.
            ctx_actions: (B, T, 6) context actions.

        Returns:
            (B, T, D) predicted embeddings.
        """
        return self.predictor(ctx_emb, ctx_actions)

    def rollout(
        self,
        initial_emb: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out predicted embeddings for planning (CEM).

        Args:
            initial_emb: (B, history_size, D) — initial context embeddings.
            action_sequences: (B, S, H, 6) — S candidate action sequences of horizon H.

        Returns:
            (B, S, H, D) — predicted embedding trajectories.
        """
        B, S, H, A = action_sequences.shape
        D = initial_emb.shape[-1]
        hist = self.history_size

        ctx = initial_emb.unsqueeze(1).expand(B, S, hist, D).reshape(B * S, hist, D)
        predictions = []

        for t in range(H):
            act = action_sequences[:, :, t].reshape(B * S, 1, A)

            if t == 0:
                ctx_act = torch.zeros(B * S, hist, A, device=ctx.device, dtype=act.dtype)
                ctx_act[:, -1] = act.squeeze(1)
            else:
                ctx_act = act_history[:, -hist:]

            pred = self.predict(ctx[:, -hist:], ctx_act)
            next_emb = pred[:, -1:]
            predictions.append(next_emb)

            ctx = torch.cat([ctx, next_emb], dim=1)
            if t == 0:
                act_history = act
            else:
                act_history = torch.cat([act_history, act], dim=1)

        result = torch.cat(predictions, dim=1).reshape(B, S, H, D)
        return result

    def param_count(self) -> dict[str, int]:
        """Count parameters by component."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        pred_params = sum(p.numel() for p in self.predictor.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc_params, "predictor": pred_params, "total": total}
