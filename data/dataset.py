"""PyTorch Dataset for loading hockey trajectories."""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HockeyTrajectoryDataset(Dataset):
    """Loads sub-trajectories for JEPA training.

    Converts HDF5 to memory-mapped numpy files on first load for fast
    random access that scales to any dataset size.

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
        self.seq_len = seq_len
        self.frameskip = frameskip
        self.stride = stride
        self.span = (seq_len - 1) * frameskip

        # Convert HDF5 to memmap on first use, then open memmap
        mmap_dir = path + ".mmap"
        if not os.path.exists(mmap_dir):
            self._convert_hdf5_to_memmap(path, mmap_dir)

        self._obs = np.memmap(
            os.path.join(mmap_dir, "obs.npy"), dtype=np.uint8, mode="r",
        ).reshape(-1, 84, 84, 3)
        self._actions_a = np.memmap(
            os.path.join(mmap_dir, "actions_a.npy"), dtype=np.int32, mode="r",
        ).reshape(-1, 3)
        self._actions_b = np.memmap(
            os.path.join(mmap_dir, "actions_b.npy"), dtype=np.int32, mode="r",
        ).reshape(-1, 3)
        self.episode_ids = np.memmap(
            os.path.join(mmap_dir, "episode_ids.npy"), dtype=np.int32, mode="r",
        )
        self.num_frames = len(self.episode_ids)

        self._valid_indices = self._compute_valid_indices()

    @staticmethod
    def _convert_hdf5_to_memmap(hdf5_path: str, mmap_dir: str) -> None:
        """Convert HDF5 to flat memory-mapped numpy files."""
        os.makedirs(mmap_dir, exist_ok=True)
        print(f"Converting {hdf5_path} to memmap at {mmap_dir}...", flush=True)

        with h5py.File(hdf5_path, "r") as f:
            n = f["observations"].shape[0]

            mapping = {
                "obs": ("observations", np.uint8, (n, 84, 84, 3)),
                "actions_a": ("actions_a", np.int32, (n, 3)),
                "actions_b": ("actions_b", np.int32, (n, 3)),
                "episode_ids": ("episode_ids", np.int32, (n,)),
            }

            for name, (hdf5_key, dtype, shape) in mapping.items():
                mm = np.memmap(
                    os.path.join(mmap_dir, f"{name}.npy"),
                    dtype=dtype, mode="w+", shape=shape,
                )
                # Copy in chunks to avoid memory spikes
                chunk = 10_000
                for start in range(0, n, chunk):
                    end = min(start + chunk, n)
                    mm[start:end] = f[hdf5_key][start:end]
                mm.flush()
                del mm

        print(f"Done! {n:,} frames converted.", flush=True)

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

        obs = self._obs[indices]              # (seq_len, 84, 84, 3) uint8
        actions_a = self._actions_a[indices]   # (seq_len, 3) int32
        actions_b = self._actions_b[indices]   # (seq_len, 3) int32

        # Normalize observations to [0, 1] float32, reorder to CHW
        obs_tensor = torch.from_numpy(obs.copy()).float() / 255.0
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)

        # Concatenate both teams' actions into joint action vector
        actions = np.concatenate([actions_a, actions_b], axis=-1)  # (T, 6)
        actions_tensor = torch.from_numpy(actions.copy()).long()

        return {
            "obs": obs_tensor,        # (seq_len, 3, 84, 84) float32
            "actions": actions_tensor,  # (seq_len, 6) int64
        }


# Use spt.data.random_split for train/val splitting:
#   train_set, val_set = spt.data.random_split(dataset, [0.9, 0.1])
