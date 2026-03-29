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

    # First pass: count total frames
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

    # Detect schema from first shard
    with h5py.File(shard_paths[0], "r") as ref:
        dataset_keys = list(ref.keys())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    chunk_size = 1024

    with h5py.File(output_path, "w") as out:
        # Create output datasets matching first shard's schema
        datasets = {}
        with h5py.File(shard_paths[0], "r") as ref:
            for key in dataset_keys:
                src = ref[key]
                shape = (total_frames, *src.shape[1:])
                kwargs = {"dtype": src.dtype}
                if key == "observations":
                    kwargs["chunks"] = (128, *src.shape[1:])
                    kwargs["compression"] = "gzip"
                    kwargs["compression_opts"] = 1
                elif len(src.shape) > 1:
                    kwargs["chunks"] = (1024, *src.shape[1:])
                else:
                    kwargs["chunks"] = (4096,)
                datasets[key] = out.create_dataset(key, shape=shape, **kwargs)

        # Stream copy
        write_idx = 0
        for info in shard_info:
            with h5py.File(info["path"], "r") as f:
                n = info["n"]
                ep_off = info["ep_offset"]
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    size = end - start
                    for key in dataset_keys:
                        data = f[key][start:end]
                        if key == "episode_ids":
                            data = data + ep_off
                        datasets[key][write_idx:write_idx + size] = data
                    write_idx += size

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
