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
    and a single controlled player's action.
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
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        predictor_dropout: float = 0.1,
        # Action params
        num_actions: int = 12,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.history_size = history_size
        self.num_preds = num_preds

        self.encoder = ViTTinyEncoder(
            encoder_scale=encoder_scale,
            image_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.predictor = ARPredictor(
            embed_dim=embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            num_actions=num_actions,
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
            ctx_actions: (B, T) controlled player's action.

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
            action_sequences: (B, S, H) — S candidate action sequences of horizon H.

        Returns:
            (B, S, H, D) — predicted embedding trajectories.
        """
        B, S, H = action_sequences.shape
        D = initial_emb.shape[-1]
        hist = self.history_size

        ctx = initial_emb.unsqueeze(1).expand(B, S, hist, D).reshape(B * S, hist, D)
        act_history = torch.zeros(B * S, hist, device=initial_emb.device, dtype=action_sequences.dtype)
        predictions = []

        for t in range(H):
            act = action_sequences[:, :, t].reshape(B * S)

            act_history = torch.cat([act_history[:, 1:], act.unsqueeze(1)], dim=1)

            pred = self.predict(ctx[:, -hist:], act_history)
            next_emb = pred[:, -1:]
            predictions.append(next_emb)

            ctx = torch.cat([ctx, next_emb], dim=1)

        result = torch.cat(predictions, dim=1).reshape(B, S, H, D)
        return result

    def get_cost(
        self,
        initial_emb: torch.Tensor,
        action_candidates: torch.Tensor,
        puck_probe: torch.nn.Module,
        target_x: float = 1.0,
        target_y: float = 0.5,
    ) -> torch.Tensor:
        """Compute cost for CEM planning using puck position probe.

        Args:
            initial_emb: (1, history_size, D) context embeddings.
            action_candidates: (1, S, H) candidate action sequences.
            puck_probe: Linear(192, 2) mapping embeddings to (puck_x, puck_y).
            target_x: Normalized x position of opponent goal.
            target_y: Normalized y position of opponent goal.

        Returns:
            (1, S) cost per candidate — lower is better.
        """
        predicted = self.rollout(initial_emb, action_candidates)
        final_emb = predicted[:, :, -1]  # (1, S, D)
        B, S, D = final_emb.shape

        pred_pos = puck_probe(final_emb.reshape(B * S, D))
        pred_pos = pred_pos.reshape(B, S, 2)

        dx = pred_pos[:, :, 0] - target_x
        dy = pred_pos[:, :, 1] - target_y
        cost = dx.pow(2) + dy.pow(2)
        return cost

    def param_count(self) -> dict[str, int]:
        """Count parameters by component."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        pred_params = sum(p.numel() for p in self.predictor.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc_params, "predictor": pred_params, "total": total}
