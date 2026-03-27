"""Abstract base class for all hockey agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from games.hockey.entities import GameState


class Agent(ABC):
    """Base class for hockey agents.

    All agents (scripted, JEPA, future self-play) inherit from this.
    Provides a uniform interface for the game loop and evaluation.
    """

    def __init__(self, team: int) -> None:
        self.team = team

    @abstractmethod
    def get_actions(self, state: GameState, obs: np.ndarray | None = None) -> list[int]:
        """Return one action per player on this team.

        Args:
            state: Current game state (positions, puck, score, etc.)
            obs: Optional 84x84x3 RGB frame. Required by learned agents
                 (JEPA) for visual encoding, ignored by scripted bots.

        Returns:
            List of PLAYERS_PER_TEAM action ints, each in [0, NUM_ACTIONS).
        """

    def reset(self) -> None:
        """Reset agent state between episodes. Override if needed."""
