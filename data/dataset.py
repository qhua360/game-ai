"""PyTorch Dataset for loading hockey trajectories."""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HockeyTrajectoryDataset(Dataset):
    """Loads sub-trajectories for JEPA training.

    Converts HDF5 to memory-mapped numpy files on first load for fast
    random access. Each sample contains RGB frames and the controlled
    player's discrete action.
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

        # Convert HDF5 to memmap on first use
        mmap_dir = path + ".mmap"
        if not os.path.exists(mmap_dir):
            self._convert_hdf5_to_memmap(path, mmap_dir)

        with open(os.path.join(mmap_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        n = meta["num_frames"]
        self._obs = np.memmap(
            os.path.join(mmap_dir, "obs.npy"), dtype=np.uint8, mode="r",
        ).reshape(n, 84, 84, 3)
        self._actions = np.memmap(
            os.path.join(mmap_dir, "actions.npy"), dtype=np.int32, mode="r",
        ).reshape(n)
        self.episode_ids = np.memmap(
            os.path.join(mmap_dir, "episode_ids.npy"), dtype=np.int32, mode="r",
        )
        self.num_frames = n

        self._valid_indices = self._compute_valid_indices()

    @staticmethod
    def _convert_hdf5_to_memmap(hdf5_path: str, mmap_dir: str) -> None:
        """Convert HDF5 to flat memory-mapped numpy files."""
        os.makedirs(mmap_dir, exist_ok=True)
        print(f"Converting {hdf5_path} to memmap at {mmap_dir}...", flush=True)

        with h5py.File(hdf5_path, "r") as f:
            n = f["observations"].shape[0]

            with open(os.path.join(mmap_dir, "meta.json"), "w") as mf:
                json.dump({"num_frames": n}, mf)

            mapping = {
                "obs": ("observations", np.uint8, (n, 84, 84, 3)),
                "actions": ("actions", np.int32, (n,)),
                "episode_ids": ("episode_ids", np.int32, (n,)),
            }

            for name, (hdf5_key, dtype, shape) in mapping.items():
                mm = np.memmap(
                    os.path.join(mmap_dir, f"{name}.npy"),
                    dtype=dtype, mode="w+", shape=shape,
                )
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

        obs = self._obs[indices]          # (T, 84, 84, 3) uint8
        actions = self._actions[indices]   # (T,) int32

        obs_tensor = torch.from_numpy(obs.copy()).float() / 255.0
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        actions_tensor = torch.from_numpy(actions.copy()).long()

        return {
            "obs": obs_tensor,         # (T, 3, 84, 84) float32
            "actions": actions_tensor,  # (T,) int64
        }


# Use spt.data.random_split for train/val splitting:
#   train_set, val_set = spt.data.random_split(dataset, [0.9, 0.1])
