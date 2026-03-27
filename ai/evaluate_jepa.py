"""Evaluate agents against each other."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pygame
pygame.init()
pygame.display.set_mode((100, 100))

from ai.base_agent import Agent
from ai.jepa_agent import JEPAAgent
from ai.scripted_bot import ScriptedBot
from games.hockey.constants import TEAM_A, TEAM_B
from games.hockey.env import HockeyEnv


def evaluate(
    agent_a: Agent,
    agent_b: Agent,
    num_episodes: int = 50,
    max_steps: int = 25000,
    render: bool = False,
) -> dict:
    """Run agent_a vs agent_b and report statistics.

    Args:
        agent_a: Team A agent.
        agent_b: Team B agent.
        num_episodes: Number of full games to play.
        max_steps: Max steps per episode.
        render: Whether to render the game.

    Returns:
        Dict with win_rate, avg_goals_for, avg_goals_against, avg_plan_time.
    """
    render_mode = "human" if render else "rgb_array"
    env = HockeyEnv(render_mode=render_mode)

    wins_a = 0
    total_goals_a = 0
    total_goals_b = 0
    plan_times: list[float] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent_a.reset()
        agent_b.reset()

        for step in range(max_steps):
            t0 = time.perf_counter()
            a_actions = agent_a.get_actions(info["state"], obs=obs)
            plan_times.append(time.perf_counter() - t0)

            b_actions = agent_b.get_actions(info["state"], obs=obs)

            obs, reward, terminated, truncated, info = env.step(
                a_actions, opponent_actions=b_actions
            )

            if terminated:
                break

        score = info["score"]
        total_goals_a += score[0]
        total_goals_b += score[1]
        if score[0] > score[1]:
            wins_a += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"Episode {ep + 1}/{num_episodes} | "
                f"Score: {score[0]}-{score[1]} | "
                f"Win rate: {wins_a / (ep + 1):.1%} | "
                f"Avg plan: {sum(plan_times[-100:]) / max(1, len(plan_times[-100:])):.1f}ms",
                flush=True,
            )

    env.close()

    n = num_episodes
    results = {
        "win_rate": wins_a / n,
        "avg_goals_for": total_goals_a / n,
        "avg_goals_against": total_goals_b / n,
        "avg_plan_time_ms": sum(plan_times) / max(1, len(plan_times)) * 1000,
        "total_episodes": n,
    }

    print("\n=== Results ===")
    print(f"Win rate:         {results['win_rate']:.1%}")
    print(f"Avg goals for:    {results['avg_goals_for']:.2f}")
    print(f"Avg goals against:{results['avg_goals_against']:.2f}")
    print(f"Avg plan time:    {results['avg_plan_time_ms']:.1f}ms")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JEPA agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--opponent", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Run scripted-vs-scripted baseline too")
    args = parser.parse_args()

    print(f"JEPA Agent vs ScriptedBot({args.opponent})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    agent_a = JEPAAgent(
        checkpoint_path=args.checkpoint,
        team=TEAM_A,
        device=args.device,
    )
    agent_b = ScriptedBot(team=TEAM_B, difficulty=args.opponent)

    jepa_results = evaluate(
        agent_a, agent_b,
        num_episodes=args.episodes,
        render=args.render,
    )

    if args.baseline:
        print("\n\n=== Baseline: ScriptedBot(hard) vs ScriptedBot({}) ===".format(args.opponent))
        baseline_a = ScriptedBot(team=TEAM_A, difficulty="hard")
        baseline_b = ScriptedBot(team=TEAM_B, difficulty=args.opponent)
        baseline_results = evaluate(
            baseline_a, baseline_b,
            num_episodes=args.episodes,
        )

        print(f"\nJEPA win rate:     {jepa_results['win_rate']:.1%}")
        print(f"Baseline win rate: {baseline_results['win_rate']:.1%}")


if __name__ == "__main__":
    main()
