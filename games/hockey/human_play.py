"""Play ice hockey with keyboard controls."""

from __future__ import annotations

import sys

import pygame

from games.hockey.constants import (
    ACTION_CHECK,
    ACTION_DOWN,
    ACTION_DOWN_LEFT,
    ACTION_DOWN_RIGHT,
    ACTION_LEFT,
    ACTION_NONE,
    ACTION_PASS,
    ACTION_RIGHT,
    ACTION_SHOOT,
    ACTION_UP,
    ACTION_UP_LEFT,
    ACTION_UP_RIGHT,
    FPS,
    PLAYERS_PER_TEAM,
    TEAM_B,
)
from games.hockey.env import HockeyEnv
from games.hockey import sound as sound_module
from ai.scripted_bot import ScriptedBot


def get_keyboard_action() -> int:
    """Map currently pressed keys to a single action."""
    keys = pygame.key.get_pressed()

    # Special actions take priority
    if keys[pygame.K_SPACE]:
        return ACTION_SHOOT
    if keys[pygame.K_p]:
        return ACTION_PASS
    if keys[pygame.K_c]:
        return ACTION_CHECK

    # Movement
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    if up and left:
        return ACTION_UP_LEFT
    if up and right:
        return ACTION_UP_RIGHT
    if down and left:
        return ACTION_DOWN_LEFT
    if down and right:
        return ACTION_DOWN_RIGHT
    if up:
        return ACTION_UP
    if down:
        return ACTION_DOWN
    if left:
        return ACTION_LEFT
    if right:
        return ACTION_RIGHT

    return ACTION_NONE


def main() -> None:
    env = HockeyEnv(render_mode="human")
    bot = ScriptedBot(team=TEAM_B, difficulty="medium")

    obs, info = env.reset()
    clock = pygame.time.Clock()

    # Player controls the first player on team A (id=0)
    controlled_idx = 0  # index within team A

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    # Cycle controlled player
                    controlled_idx = (controlled_idx + 1) % PLAYERS_PER_TEAM
                elif event.key == pygame.K_m:
                    muted = sound_module.toggle_mute()
                    print(f"Sound {'muted' if muted else 'unmuted'}")

        # Get human action for controlled player
        human_action = get_keyboard_action()

        # Build team A actions: human controls one player, bot controls rest
        team_a_actions = [ACTION_NONE] * PLAYERS_PER_TEAM
        team_a_actions[controlled_idx] = human_action

        # Simple AI for the other team A players (teammates)
        teammate_bot = ScriptedBot(team=0, difficulty="easy")
        teammate_actions = teammate_bot.get_actions(info["state"])
        for i in range(PLAYERS_PER_TEAM):
            if i != controlled_idx:
                team_a_actions[i] = teammate_actions[i]

        # Opponent bot
        bot_actions = bot.get_actions(info["state"])

        # Step
        obs, reward, terminated, truncated, info = env.step(
            team_a_actions, opponent_actions=bot_actions
        )

        # Render (already done in step via renderer, but let's update controlled highlight)
        state = info["state"]
        controlled_player_id = controlled_idx  # team A player ids are 0, 1, 2
        env._renderer.render(state, controlled_player_id=controlled_player_id)

        if terminated:
            print(f"Game Over! Final score: RED {state.score[0]} - {state.score[1]} BLUE")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(FPS)

    env.close()


if __name__ == "__main__":
    main()
