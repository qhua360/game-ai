"""Collect trajectories from bot-vs-bot games for JEPA training.

Streams frames directly to HDF5 on disk — constant memory usage regardless
of episode count.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import h5py
import numpy as np

# Headless pygame setup — must happen before any pygame import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()
pygame.display.set_mode((100, 100))

from ai.scripted_bot import ScriptedBot
from games.hockey.constants import ACTION_NONE, NUM_ACTIONS, PLAYERS_PER_TEAM, TEAM_A, TEAM_B
from games.hockey.env import HockeyEnv

# Initial allocation size and growth factor for resizable HDF5 datasets
INITIAL_CAPACITY = 10_000
GROWTH_FACTOR = 2


def collect_episodes(
    num_episodes: int,
    output_path: str,
    random_fraction: float = 0.2,
    max_steps_per_episode: int = 2000,
    verbose: bool = True,
) -> None:
    """Run bot games and stream trajectories directly to HDF5 on disk."""
    env = HockeyEnv(render_mode="rgb_array")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    t0 = time.time()
    write_idx = 0
    capacity = INITIAL_CAPACITY

    with h5py.File(output_path, "w") as f:
        # Create resizable datasets (maxshape=None allows unlimited growth)
        ds_obs = f.create_dataset(
            "observations",
            shape=(capacity, 84, 84, 3),
            maxshape=(None, 84, 84, 3),
            dtype=np.uint8,
            chunks=(128, 84, 84, 3),
            compression="gzip",
            compression_opts=1,
        )
        ds_act_a = f.create_dataset(
            "actions_a",
            shape=(capacity, PLAYERS_PER_TEAM),
            maxshape=(None, PLAYERS_PER_TEAM),
            dtype=np.int32,
            chunks=(1024, PLAYERS_PER_TEAM),
        )
        ds_act_b = f.create_dataset(
            "actions_b",
            shape=(capacity, PLAYERS_PER_TEAM),
            maxshape=(None, PLAYERS_PER_TEAM),
            dtype=np.int32,
            chunks=(1024, PLAYERS_PER_TEAM),
        )
        ds_ep = f.create_dataset(
            "episode_ids",
            shape=(capacity,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(4096,),
        )

        for ep in range(num_episodes):
            diff_a = random.choice(difficulties)
            diff_b = random.choice(difficulties)
            use_random = random.random() < random_fraction

            bot_a = ScriptedBot(team=TEAM_A, difficulty=diff_a)
            bot_b = ScriptedBot(team=TEAM_B, difficulty=diff_b)

            obs, info = env.reset()

            for step in range(max_steps_per_episode):
                if use_random:
                    actions_a = [random.randint(0, NUM_ACTIONS - 1) for _ in range(PLAYERS_PER_TEAM)]
                    actions_b = [random.randint(0, NUM_ACTIONS - 1) for _ in range(PLAYERS_PER_TEAM)]
                else:
                    actions_a = bot_a.get_actions(info["state"])
                    actions_b = bot_b.get_actions(info["state"])

                # Only save frames during active play
                if info["phase"] == "PLAY":
                    if write_idx >= capacity:
                        capacity = int(capacity * GROWTH_FACTOR)
                        ds_obs.resize(capacity, axis=0)
                        ds_act_a.resize(capacity, axis=0)
                        ds_act_b.resize(capacity, axis=0)
                        ds_ep.resize(capacity, axis=0)

                    ds_obs[write_idx] = obs
                    ds_act_a[write_idx] = actions_a
                    ds_act_b[write_idx] = actions_b
                    ds_ep[write_idx] = ep
                    write_idx += 1

                obs, reward, terminated, truncated, info = env.step(
                    actions_a, opponent_actions=actions_b
                )

                if terminated:
                    break

            if verbose and (ep + 1) % 10 == 0:
                elapsed = time.time() - t0
                fps = write_idx / elapsed
                score = info["score"]
                print(
                    f"Episode {ep + 1}/{num_episodes} | "
                    f"Play frames: {write_idx} | "
                    f"{fps:.0f} frames/sec | "
                    f"Score: {score[0]}-{score[1]} | "
                    f"{'random' if use_random else f'{diff_a} vs {diff_b}'}",
                    flush=True,
                )

        # Trim datasets to actual size
        ds_obs.resize(write_idx, axis=0)
        ds_act_a.resize(write_idx, axis=0)
        ds_act_b.resize(write_idx, axis=0)
        ds_ep.resize(write_idx, axis=0)

    env.close()

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDone! Saved {write_idx} frames from {num_episodes} episodes to {output_path}")
    print(f"File size: {file_size_mb:.1f} MB | Time: {elapsed:.1f}s | {write_idx / elapsed:.0f} frames/sec")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hockey trajectories")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--output", type=str, default="data/trajectories/bot_v_bot.h5", help="Output HDF5 path")
    parser.add_argument("--random-fraction", type=float, default=0.2, help="Fraction of random-action episodes")
    parser.add_argument("--max-steps", type=int, default=25000, help="Max steps per episode (25K covers a full 3-period game)")
    args = parser.parse_args()

    collect_episodes(
        num_episodes=args.episodes,
        output_path=args.output,
        random_fraction=args.random_fraction,
        max_steps_per_episode=args.max_steps,
    )


if __name__ == "__main__":
    main()
