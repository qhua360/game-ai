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
        # Pad action history with zeros to match history size
        act_history = torch.zeros(B * S, hist, A, device=initial_emb.device, dtype=action_sequences.dtype)
        predictions = []

        for t in range(H):
            act = action_sequences[:, :, t].reshape(B * S, 1, A)

            # Shift action history left and insert new action at the end
            act_history = torch.cat([act_history[:, 1:], act], dim=1)  # (B*S, hist, A)

            pred = self.predict(ctx[:, -hist:], act_history)  # (B*S, hist, D)
            next_emb = pred[:, -1:]  # (B*S, 1, D)
            predictions.append(next_emb)

            ctx = torch.cat([ctx, next_emb], dim=1)

        result = torch.cat(predictions, dim=1).reshape(B, S, H, D)
        return result

    def criterion(
        self,
        predicted_emb: torch.Tensor,
        goal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE cost between final predicted embedding and goal.

        Args:
            predicted_emb: (B, S, H, D) predicted trajectory embeddings.
            goal_emb: (B, 1, D) goal state embedding.

        Returns:
            (B, S) cost per sample — lower is better.
        """
        final = predicted_emb[:, :, -1]  # (B, S, D)
        goal = goal_emb.expand_as(final)  # (B, S, D)
        return (final - goal).pow(2).mean(dim=-1)  # (B, S)

    def get_cost(
        self,
        initial_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cost for CEM planning.

        Args:
            initial_emb: (1, history_size, D) context embeddings.
            goal_emb: (1, 1, D) goal state embedding.
            action_candidates: (1, S, H, 6) candidate action sequences.

        Returns:
            (1, S) cost per candidate.
        """
        predicted = self.rollout(initial_emb, action_candidates)  # (1, S, H, D)
        return self.criterion(predicted, goal_emb)  # (1, S)

    def param_count(self) -> dict[str, int]:
        """Count parameters by component."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        pred_params = sum(p.numel() for p in self.predictor.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc_params, "predictor": pred_params, "total": total}
