"""Merge multiple HDF5 trajectory shards into a single file (streaming, low memory)."""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np


def merge_shards(shard_pattern: str, output_path: str) -> None:
    """Merge HDF5 shards by streaming — constant memory usage."""
    shard_paths = sorted(glob.glob(shard_pattern))
    if not shard_paths:
        print(f"No shards found matching {shard_pattern}")
        return

    print(f"Merging {len(shard_paths)} shards: {shard_paths}")

    # First pass: count total frames and collect episode offsets
    total_frames = 0
    shard_info = []
    episode_offset = 0
    for path in shard_paths:
        with h5py.File(path, "r") as f:
            n = f["observations"].shape[0]
            max_ep = int(f["episode_ids"][-1]) if n > 0 else -1
            shard_info.append({"path": path, "n": n, "ep_offset": episode_offset})
            total_frames += n
            episode_offset += max_ep + 1
            print(f"  {path}: {n:,} frames")

    print(f"Total: {total_frames:,} frames")

    # Second pass: stream copy chunk by chunk
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    chunk_size = 1024  # copy this many frames at a time

    with h5py.File(output_path, "w") as out:
        # Get shape info from first shard
        with h5py.File(shard_paths[0], "r") as ref:
            obs_shape = ref["observations"].shape[1:]  # (84, 84, 3)
            act_a_cols = ref["actions_a"].shape[1]
            act_b_cols = ref["actions_b"].shape[1]

        ds_obs = out.create_dataset(
            "observations", shape=(total_frames, *obs_shape),
            dtype=np.uint8, chunks=(128, *obs_shape), compression="gzip", compression_opts=1,
        )
        ds_act_a = out.create_dataset(
            "actions_a", shape=(total_frames, act_a_cols),
            dtype=np.int32, chunks=(1024, act_a_cols),
        )
        ds_act_b = out.create_dataset(
            "actions_b", shape=(total_frames, act_b_cols),
            dtype=np.int32, chunks=(1024, act_b_cols),
        )
        ds_ep = out.create_dataset(
            "episode_ids", shape=(total_frames,),
            dtype=np.int32, chunks=(4096,),
        )

        write_idx = 0
        for info in shard_info:
            with h5py.File(info["path"], "r") as f:
                n = info["n"]
                ep_off = info["ep_offset"]
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    ds_obs[write_idx:write_idx + (end - start)] = f["observations"][start:end]
                    ds_act_a[write_idx:write_idx + (end - start)] = f["actions_a"][start:end]
                    ds_act_b[write_idx:write_idx + (end - start)] = f["actions_b"][start:end]
                    ds_ep[write_idx:write_idx + (end - start)] = f["episode_ids"][start:end] + ep_off
                    write_idx += end - start

                if write_idx % 100_000 < chunk_size:
                    print(f"  Written {write_idx:,}/{total_frames:,} frames...")

            print(f"  Finished {info['path']}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nMerged: {total_frames:,} frames → {output_path} ({file_size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge HDF5 trajectory shards")
    parser.add_argument("--pattern", type=str, default="data/trajectories/shard_*.h5")
    parser.add_argument("--output", type=str, default="data/trajectories/bot_v_bot.h5")
    args = parser.parse_args()
    merge_shards(args.pattern, args.output)


if __name__ == "__main__":
    main()
