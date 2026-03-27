"""PyTorch Dataset for loading hockey trajectories."""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HockeyTrajectoryDataset(Dataset):
    """Loads sub-trajectories from HDF5 for JEPA training.

    Each sample is a sequence of (seq_len) frames sampled with frameskip
    between frames. Sample start positions are strided by `stride` to
    control overlap:

        frameskip=15, stride=3, seq_len=4:
        Sample 0: frames [0,  15, 30, 45]
        Sample 1: frames [3,  18, 33, 48]
        Sample 2: frames [6,  21, 36, 51]
        ...

    Args:
        path: Path to HDF5 trajectory file.
        seq_len: Number of frames per sample.
        frameskip: Gap between frames within a sample (in raw frames).
        stride: Gap between sample start positions (in raw frames).
    """

    def __init__(
        self,
        path: str,
        seq_len: int = 4,
        frameskip: int = 15,
        stride: int = 3,
    ) -> None:
        self.path = path
        self.seq_len = seq_len
        self.frameskip = frameskip
        self.stride = stride
        self.span = (seq_len - 1) * frameskip  # raw frames covered by one sample

        # Load metadata into memory (small arrays)
        with h5py.File(path, "r") as f:
            self.num_frames = f["observations"].shape[0]
            self.episode_ids = f["episode_ids"][:]

        # Precompute valid start indices
        self._valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self) -> np.ndarray:
        """Find valid start positions strided by self.stride within each episode."""
        valid = []
        episodes = np.unique(self.episode_ids)
        for ep in episodes:
            ep_mask = self.episode_ids == ep
            ep_indices = np.where(ep_mask)[0]
            if len(ep_indices) == 0:
                continue
            ep_start = ep_indices[0]
            ep_end = ep_indices[-1]

            # Stride through the episode
            i = ep_start
            while i + self.span <= ep_end:
                valid.append(i)
                i += self.stride

        return np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self._valid_indices[idx]
        indices = [start + i * self.frameskip for i in range(self.seq_len)]

        with h5py.File(self.path, "r") as f:
            obs = f["observations"][indices]       # (seq_len, 84, 84, 3) uint8
            actions_a = f["actions_a"][indices]     # (seq_len, 3) int32
            actions_b = f["actions_b"][indices]     # (seq_len, 3) int32

        # Normalize observations to [0, 1] float32, reorder to CHW
        obs_tensor = torch.from_numpy(obs).float() / 255.0  # (T, H, W, C)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)          # (T, C, H, W)

        # Concatenate both teams' actions into joint action vector
        actions = np.concatenate([actions_a, actions_b], axis=-1)  # (T, 6)
        actions_tensor = torch.from_numpy(actions).long()

        return {
            "obs": obs_tensor,        # (seq_len, 3, 84, 84) float32
            "actions": actions_tensor,  # (seq_len, 6) int64
        }


# Use spt.data.random_split for train/val splitting:
#   train_set, val_set = spt.data.random_split(dataset, [0.9, 0.1])
