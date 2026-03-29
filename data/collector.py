"""Collect trajectories from bot games for JEPA training.

Streams frames directly to HDF5 on disk — constant memory usage regardless
of episode count. Supports configurable player counts.
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
from games.hockey.constants import ACTION_NONE, NUM_ACTIONS, TEAM_A, TEAM_B
from games.hockey.env import HockeyEnv

# Initial allocation size and growth factor for resizable HDF5 datasets
INITIAL_CAPACITY = 10_000
GROWTH_FACTOR = 2
STATE_DIM = 9  # player state vector size


def collect_episodes(
    num_episodes: int,
    output_path: str,
    num_players_a: int = 3,
    num_players_b: int = 3,
    randomize_positions: bool = False,
    max_steps_per_episode: int = 25000,
    action_hold_frames: int = 15,
    verbose: bool = True,
) -> None:
    """Run bot games and stream trajectories directly to HDF5 on disk."""
    env = HockeyEnv(
        render_mode="rgb_array",
        num_players_a=num_players_a,
        num_players_b=num_players_b,
        randomize_positions=randomize_positions,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    write_idx = 0
    capacity = INITIAL_CAPACITY

    with h5py.File(output_path, "w") as f:
        # Store config for later use
        f.attrs["num_players_a"] = num_players_a
        f.attrs["num_players_b"] = num_players_b
        f.attrs["action_hold_frames"] = action_hold_frames

        ds_obs = f.create_dataset(
            "observations",
            shape=(capacity, 84, 84, 3), maxshape=(None, 84, 84, 3),
            dtype=np.uint8, chunks=(128, 84, 84, 3), compression="gzip", compression_opts=1,
        )
        ds_act_a = f.create_dataset(
            "actions_a",
            shape=(capacity, num_players_a), maxshape=(None, num_players_a),
            dtype=np.int32, chunks=(1024, num_players_a),
        )
        ds_act_b = f.create_dataset(
            "actions_b",
            shape=(capacity, max(1, num_players_b)), maxshape=(None, max(1, num_players_b)),
            dtype=np.int32, chunks=(1024, max(1, num_players_b)),
        )
        ds_state = f.create_dataset(
            "state_vectors",
            shape=(capacity, STATE_DIM), maxshape=(None, STATE_DIM),
            dtype=np.float32, chunks=(1024, STATE_DIM),
        )
        ds_ep = f.create_dataset(
            "episode_ids",
            shape=(capacity,), maxshape=(None,),
            dtype=np.int32, chunks=(4096,),
        )

        def _grow():
            nonlocal capacity
            capacity = int(capacity * GROWTH_FACTOR)
            for ds in [ds_obs, ds_act_a, ds_act_b, ds_state, ds_ep]:
                ds.resize(capacity, axis=0)

        for ep in range(num_episodes):
            # Create bots with action persistence
            bot_a = ScriptedBot(team=TEAM_A, difficulty="medium", action_hold_frames=action_hold_frames)
            if num_players_b > 0:
                bot_b = ScriptedBot(team=TEAM_B, difficulty="medium", action_hold_frames=action_hold_frames)

            obs, info = env.reset()

            for step in range(max_steps_per_episode):
                # Get actions (with persistence — same action held for N frames)
                actions_a = bot_a.get_actions(info["state"])

                if num_players_b > 0:
                    actions_b = bot_b.get_actions(info["state"])
                else:
                    actions_b = []

                # Only save during PLAY phase
                if info["phase"] == "PLAY":
                    if write_idx >= capacity:
                        _grow()

                    ds_obs[write_idx] = obs
                    ds_act_a[write_idx] = actions_a
                    ds_act_b[write_idx] = actions_b if actions_b else [0]
                    ds_state[write_idx] = env.get_player_state(player_id=0)
                    ds_ep[write_idx] = ep
                    write_idx += 1

                obs, reward, terminated, truncated, info = env.step(
                    actions_a, opponent_actions=actions_b if actions_b else None
                )

                if terminated:
                    break

            if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
                elapsed = time.time() - t0
                fps = write_idx / max(0.1, elapsed)
                print(
                    f"Episode {ep + 1}/{num_episodes} | "
                    f"Frames: {write_idx} | "
                    f"{fps:.0f} frames/sec",
                    flush=True,
                )

        # Trim datasets to actual size
        for ds in [ds_obs, ds_act_a, ds_act_b, ds_state, ds_ep]:
            ds.resize(write_idx, axis=0)

    env.close()

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDone! Saved {write_idx} frames from {num_episodes} episodes to {output_path}")
    print(f"File size: {file_size_mb:.1f} MB | Time: {elapsed:.1f}s | {write_idx / elapsed:.0f} frames/sec")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hockey trajectories")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/trajectories/bot_v_bot.h5")
    parser.add_argument("--num-players-a", type=int, default=3)
    parser.add_argument("--num-players-b", type=int, default=3)
    parser.add_argument("--randomize", action="store_true", help="Randomize positions each episode")
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument("--action-hold", type=int, default=15, help="Frames to hold each action")
    args = parser.parse_args()

    collect_episodes(
        num_episodes=args.episodes,
        output_path=args.output,
        num_players_a=args.num_players_a,
        num_players_b=args.num_players_b,
        randomize_positions=args.randomize,
        max_steps_per_episode=args.max_steps,
        action_hold_frames=args.action_hold,
    )


if __name__ == "__main__":
    main()
