"""Training script for the JEPA world model using stable-pretraining.

Uses spt.Module for Lightning training orchestration and spt.Manager
for checkpointing, logging, and distributed training support.

Reference: LeWorldModel train.py (https://github.com/lucas-maes/le-wm)
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import lightning as pl
import stable_pretraining as spt
import torch
from torch.utils.data import DataLoader

from data.dataset import HockeyTrajectoryDataset
from model.sigreg import SIGReg
from model.world_model import LeWM


# ---------------------------------------------------------------------------
# Forward function (called each training/validation step by spt.Module)
# ---------------------------------------------------------------------------

def lejepa_forward(self, batch: dict, stage: str, *, cfg: dict) -> dict:
    """JEPA training forward pass.

    This function is bound to spt.Module via functools.partial.
    `self` is the spt.Module instance, which has:
        - self.model: LeWM world model
        - self.sigreg: SIGReg regularizer

    Args:
        batch: Dict with "obs" (B, T, C, H, W) and "actions" (B, T, 6).
        stage: "fit" or "validate".
        cfg: Training config dict.

    Returns:
        Dict with "loss" key (required by spt.Module).
    """
    obs = batch["obs"]
    actions = batch["actions"]

    history_size = cfg["history_size"]
    num_preds = cfg["num_preds"]
    lambda_sigreg = cfg["lambda_sigreg"]

    # Encode all frames
    emb = self.model.encode(obs)  # (B, T, 192)

    # Split: context frames 0..history_size-1, targets shifted by num_preds
    ctx_emb = emb[:, :history_size]
    ctx_actions = actions[:, :history_size]
    tgt_emb = emb[:, num_preds:]

    # Predict next embeddings
    pred_emb = self.model.predict(ctx_emb, ctx_actions)

    # Losses
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = self.sigreg(emb.transpose(0, 1))
    loss = pred_loss + lambda_sigreg * sigreg_loss

    # Log metrics
    self.log_dict(
        {
            f"{stage}/loss": loss,
            f"{stage}/pred_loss": pred_loss,
            f"{stage}/sigreg_loss": sigreg_loss,
        },
        on_step=(stage == "fit"),
        on_epoch=True,
        sync_dist=True,
    )

    return {"loss": loss, "pred_loss": pred_loss.detach(), "sigreg_loss": sigreg_loss.detach()}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 5e-5,
    weight_decay: float = 1e-3,
    gradient_clip_val: float = 1.0,
    seq_len: int = 4,
    frameskip: int = 3,
    history_size: int = 3,
    num_preds: int = 1,
    lambda_sigreg: float = 0.09,
    num_workers: int = 4,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
) -> None:
    """Train the JEPA world model using stable-pretraining framework."""

    # Config dict passed to forward function
    cfg = {
        "history_size": history_size,
        "num_preds": num_preds,
        "lambda_sigreg": lambda_sigreg,
    }

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path}...")
    dataset = HockeyTrajectoryDataset(data_path, seq_len=seq_len, frameskip=frameskip)
    print(f"Dataset: {len(dataset)} sub-trajectories")

    train_set, val_set = spt.data.random_split(
        dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(seed)
    )
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    # MPS tensors can't be pickled across DataLoader worker processes
    # https://github.com/pytorch/pytorch/issues/87688
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if use_mps and num_workers > 0:
        print(f"MPS detected: setting num_workers=0 (MPS tensors can't be shared across workers)")
        num_workers = 0

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    # ── Model ─────────────────────────────────────────────────────────────
    world_model = LeWM(
        history_size=history_size,
        num_preds=num_preds,
    )
    params = world_model.param_count()
    print(f"Model: {params['total']:,} params (encoder: {params['encoder']:,}, predictor: {params['predictor']:,})")

    # ── spt.Module ────────────────────────────────────────────────────────
    # Compute scheduler params explicitly — spt's smart defaults use
    # trainer.estimated_stepping_batches which fails on MPS/Lightning 2.6
    steps_per_epoch = len(train_loader)
    max_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(0.01 * max_steps))

    module = spt.Module(
        forward=partial(lejepa_forward, cfg=cfg),
        model=world_model,
        sigreg=SIGReg(),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": {
                    "type": "AdamW",
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                "scheduler": {
                    "type": "LinearWarmupCosineAnnealingLR",
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "warmup_start_lr": 0.0,
                    "eta_min": 0.0,
                },
                "interval": "epoch",
            },
        },
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    # Auto-detect precision: bf16 on CUDA, 32 on MPS/CPU
    accelerator = "auto"
    if torch.cuda.is_available():
        precision = "bf16-mixed"
    else:
        precision = "32-true"  # MPS doesn't support bf16/fp16 mixed precision

    # Set MPS fallback for unsupported ops
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # WandB logger
    wandb_logger = pl.pytorch.loggers.WandbLogger(
        project="jepa-game-ai",
        name=f"hockey-lewm-bs{batch_size}-lr{lr}",
        save_dir=checkpoint_dir,
        log_model=False,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        default_root_dir=checkpoint_dir,
        enable_checkpointing=True,
        log_every_n_steps=50,
        logger=wandb_logger,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=seed,
        ckpt_path=None,
    )
    manager()

    print(f"\nTraining complete. Checkpoints in {checkpoint_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JEPA world model")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 trajectory file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        seq_len=args.seq_len,
        frameskip=args.frameskip,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
