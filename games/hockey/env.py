"""Gymnasium-compatible ice hockey environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from games.hockey.constants import (
    FACEOFF_A_POSITIONS,
    FACEOFF_B_POSITIONS,
    NUM_ACTIONS,
    PERIOD_LENGTH_SEC,
    PLAYERS_PER_TEAM,
    OBS_HEIGHT,
    OBS_WIDTH,
    RINK_HEIGHT,
    RINK_WIDTH,
    RINK_X,
    RINK_Y,
    TEAM_A,
    TEAM_B,
    TEAM_A_POSITIONS,
    TEAM_B_POSITIONS,
    ACTION_NONE,
)
from games.hockey.entities import GameState, Phase, Player, Puck, Rink, create_goals
from games.hockey.physics import step_physics
from games.hockey.renderer import Renderer
from games.hockey import sound as sound_module


class HockeyEnv(gym.Env):
    """2D Ice Hockey environment.

    Observation: 84x84x3 RGB frame.
    Action: One discrete action per player on the controlled team.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = "human") -> None:
        super().__init__()

        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS] * PLAYERS_PER_TEAM)

        # Internal state
        self._state: GameState | None = None
        self._renderer: Renderer | None = None
        self._sound_initialized = False

        # Init renderer
        if render_mode == "human":
            import pygame
            pygame.init()
            self._renderer = Renderer(headless=False)
            sound_module.init()
            self._sound_initialized = True
        elif render_mode == "rgb_array":
            import pygame
            pygame.init()
            self._renderer = Renderer(headless=True)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        rink = Rink()
        players = []

        # Create team A players
        for i, (fx, fy) in enumerate(TEAM_A_POSITIONS):
            players.append(Player(
                x=rink.x + fx * rink.width,
                y=rink.y + fy * rink.height,
                team=TEAM_A,
                player_id=i,
            ))

        # Create team B players
        for i, (fx, fy) in enumerate(TEAM_B_POSITIONS):
            players.append(Player(
                x=rink.x + fx * rink.width,
                y=rink.y + fy * rink.height,
                team=TEAM_B,
                player_id=PLAYERS_PER_TEAM + i,
                facing_angle=3.14159,  # face left
            ))

        puck = Puck(x=rink.center_x, y=rink.center_y)
        goals = create_goals(rink)

        self._state = GameState(
            players=players,
            puck=puck,
            goals=goals,
            rink=rink,
            score=[0, 0],
            period=1,
            time_remaining=PERIOD_LENGTH_SEC,
            phase=Phase.FACEOFF,
            phase_timer=1.5,
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray | list[int],
        opponent_actions: list[int] | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the environment.

        Args:
            action: Actions for team A (the controlled team).
            opponent_actions: Actions for team B. If None, team B does nothing.
        """
        assert self._state is not None, "Call reset() first"

        team_a_actions = list(action) if not isinstance(action, list) else action
        team_b_actions = opponent_actions or [ACTION_NONE] * PLAYERS_PER_TEAM

        # Ensure correct lengths
        while len(team_a_actions) < PLAYERS_PER_TEAM:
            team_a_actions.append(ACTION_NONE)
        while len(team_b_actions) < PLAYERS_PER_TEAM:
            team_b_actions.append(ACTION_NONE)

        prev_score = list(self._state.score)

        # Physics step
        events = step_physics(self._state, team_a_actions, team_b_actions)

        # Play sounds
        if self._sound_initialized:
            for event in events:
                sound_module.play(event.kind)

        # Calculate reward (from team A perspective)
        reward = 0.0
        for event in events:
            if event.kind == "goal":
                if event.data["scoring_team"] == TEAM_A:
                    reward += 1.0
                else:
                    reward -= 1.0
            elif event.kind == "pickup":
                if event.data["team"] == TEAM_A:
                    reward += 0.01
            elif event.kind == "shot":
                if event.data["team"] == TEAM_A:
                    reward += 0.05

        terminated = self._state.phase == Phase.GAME_OVER
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info["events"] = events

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            return None
        if self._state is None:
            return None

        if self.render_mode == "human":
            self._renderer.render(self._state)
            return None
        elif self.render_mode == "rgb_array":
            return self._renderer.get_obs(self._state)
        return None

    def close(self) -> None:
        if self._sound_initialized:
            sound_module.cleanup()
        if self._renderer:
            self._renderer.cleanup()
        import pygame
        pygame.quit()

    def _get_obs(self) -> np.ndarray:
        if self._renderer is None or self._state is None:
            return np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)
        return self._renderer.get_obs(self._state)

    def _get_info(self) -> dict[str, Any]:
        if self._state is None:
            return {}
        return {
            "state": self._state,
            "score": list(self._state.score),
            "period": self._state.period,
            "time_remaining": self._state.time_remaining,
            "phase": self._state.phase.name,
        }
