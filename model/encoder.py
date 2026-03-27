"""ViT-Tiny encoder for JEPA world model.

Uses stable-pretraining's vit_hf() for the Vision Transformer backbone.
Encodes 84x84 RGB frames into 192-dim latent embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stable_pretraining.backbone.utils import vit_hf


class ViTTinyEncoder(nn.Module):
    """Vision Transformer encoder using stable-pretraining backbone.

    Architecture:
        - Input: (B, 3, 84, 84) RGB float32 [0, 1]
        - HuggingFace ViT-Tiny via spt.backbone.utils.vit_hf()
        - CLS token extraction → projector → (B, 192) embedding

    Args:
        encoder_scale: ViT scale ("tiny", "small", "base").
        image_size: Input image resolution.
        patch_size: Patch size for ViT.
        embed_dim: Output embedding dimension.
        pretrained: Whether to use pretrained weights.
    """

    def __init__(
        self,
        encoder_scale: str = "tiny",
        image_size: int = 84,
        patch_size: int = 14,
        embed_dim: int = 192,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        # HuggingFace ViT via stable-pretraining
        self.encoder = vit_hf(
            encoder_scale,
            patch_size=patch_size,
            image_size=image_size,
            pretrained=pretrained,
            use_mask_token=False,
        )
        hidden_dim = self.encoder.config.hidden_size
        self.embed_dim = embed_dim

        # Projector: CLS token → final embedding
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent embeddings.

        Args:
            x: (B, C, H, W) float32 images normalized to [0, 1].

        Returns:
            (B, embed_dim) latent embeddings.
        """
        output = self.encoder(x, interpolate_pos_encoding=True)
        cls_token = output.last_hidden_state[:, 0]  # (B, hidden_dim)
        emb = self.projector(cls_token)  # (B, embed_dim)
        return emb

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of frames.

        Args:
            obs: (B, T, C, H, W) sequence of frames.

        Returns:
            (B, T, embed_dim) sequence of embeddings.
        """
        B, T, C, H, W = obs.shape
        flat = obs.reshape(B * T, C, H, W)
        emb = self.forward(flat)  # (B*T, embed_dim)
        return emb.reshape(B, T, self.embed_dim)
