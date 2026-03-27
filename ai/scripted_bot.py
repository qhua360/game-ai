"""Simple rule-based hockey bot."""

from __future__ import annotations

import math
import random

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
    NUM_ACTIONS,
    PLAYERS_PER_TEAM,
    TEAM_A,
    TEAM_B,
)
import numpy as np

from ai.base_agent import Agent
from games.hockey.entities import GameState, Phase, Player


class ScriptedBot(Agent):
    """Rule-based bot with adjustable difficulty."""

    def __init__(self, team: int, difficulty: str = "medium") -> None:
        super().__init__(team)
        self.difficulty = difficulty
        self._reaction_frames = {"easy": 12, "medium": 6, "hard": 0}[difficulty]
        self._random_chance = {"easy": 0.30, "medium": 0.10, "hard": 0.02}[difficulty]
        self._frame_counter = 0

    def get_actions(self, state: GameState, obs: np.ndarray | None = None) -> list[int]:
        """Return one action per player on this team."""
        self._frame_counter += 1

        if state.phase != Phase.PLAY:
            return [ACTION_NONE] * PLAYERS_PER_TEAM

        my_players = state.team_players(self.team)
        actions = []

        for player in my_players:
            # Random action chance (difficulty noise)
            if random.random() < self._random_chance:
                actions.append(random.randint(0, NUM_ACTIONS - 1))
                continue

            # Reaction delay: repeat no-op for first N frames after state change
            if self._frame_counter % max(1, self._reaction_frames) != 0 and self.difficulty == "easy":
                actions.append(ACTION_NONE)
                continue

            action = self._decide(player, state, my_players)
            actions.append(action)

        return actions

    def _decide(self, player: Player, state: GameState, teammates: list[Player]) -> int:
        """Decide action for a single player."""
        puck = state.puck
        opp_goal = state.opponent_goal(self.team)
        own_goal = state.get_goal_for_team(self.team)

        # --- I have the puck ---
        if player.has_puck:
            goal_x = opp_goal.x + opp_goal.width / 2
            goal_y = opp_goal.center_y
            dist_to_goal = player.distance_to(goal_x, goal_y)

            # Close to goal -> shoot
            if dist_to_goal < 250:
                return ACTION_SHOOT

            # Teammate closer to goal -> pass
            for t in teammates:
                if t is not player and t.distance_to(goal_x, goal_y) < dist_to_goal - 50:
                    return ACTION_PASS

            # Otherwise skate toward goal
            return _direction_toward(player, goal_x, goal_y)

        # --- Teammate has puck -> support ---
        team_has_puck = any(t.has_puck for t in teammates)
        if team_has_puck:
            carrier = next(t for t in teammates if t.has_puck)
            # Move to open space ahead of carrier toward opponent goal
            target_x = (carrier.x + opp_goal.x) / 2
            target_y = player.y  # maintain lane
            # Spread vertically
            idx = teammates.index(player)
            if idx == 0:
                target_y = state.rink.center_y - 100
            elif idx == 2:
                target_y = state.rink.center_y + 100
            else:
                target_y = state.rink.center_y
            return _direction_toward(player, target_x, target_y)

        # --- Nobody has puck -> chase it ---
        # Only closest player chases, others defend
        my_dists = [(t, t.distance_to(puck.x, puck.y)) for t in teammates]
        my_dists.sort(key=lambda x: x[1])

        if player is my_dists[0][0]:
            # Closest -> chase puck
            return _direction_toward(player, puck.x, puck.y)

        # Opponent has puck near our goal -> intercept
        if puck.carrier and puck.carrier.team != self.team:
            opp_dist_to_our_goal = puck.carrier.distance_to(
                own_goal.x + own_goal.width / 2, own_goal.center_y
            )
            if opp_dist_to_our_goal < 300:
                # Skate toward puck carrier
                return _direction_toward(player, puck.carrier.x, puck.carrier.y)

        # Check if nearby opponent has puck -> body check
        if puck.carrier and puck.carrier.team != self.team:
            if player.distance_to(puck.carrier.x, puck.carrier.y) < 40:
                return ACTION_CHECK

        # Default: defend — position between puck and own goal
        defend_x = (puck.x + own_goal.x + own_goal.width / 2) / 2
        defend_y = (puck.y + own_goal.center_y) / 2
        if player.distance_to(defend_x, defend_y) > 20:
            return _direction_toward(player, defend_x, defend_y)

        return ACTION_NONE


def _direction_toward(player: Player, tx: float, ty: float) -> int:
    """Get the discrete movement action that moves player toward (tx, ty)."""
    dx = tx - player.x
    dy = ty - player.y

    if abs(dx) < 10 and abs(dy) < 10:
        return ACTION_NONE

    angle = math.atan2(dy, dx)
    # Map angle to 8 discrete directions
    # 0=right, pi/2=down, pi=left, -pi/2=up
    deg = math.degrees(angle)

    if -22.5 <= deg < 22.5:
        return ACTION_RIGHT
    elif 22.5 <= deg < 67.5:
        return ACTION_DOWN_RIGHT
    elif 67.5 <= deg < 112.5:
        return ACTION_DOWN
    elif 112.5 <= deg < 157.5:
        return ACTION_DOWN_LEFT
    elif deg >= 157.5 or deg < -157.5:
        return ACTION_LEFT
    elif -157.5 <= deg < -112.5:
        return ACTION_UP_LEFT
    elif -112.5 <= deg < -67.5:
        return ACTION_UP
    elif -67.5 <= deg < -22.5:
        return ACTION_UP_RIGHT

    return ACTION_NONE
