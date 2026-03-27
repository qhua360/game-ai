"""Game entity dataclasses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from games.hockey.constants import (
    GOAL_DEPTH,
    GOAL_WIDTH,
    PLAYERS_PER_TEAM,
    RINK_HEIGHT,
    RINK_WIDTH,
    RINK_X,
    RINK_Y,
    TEAM_A,
    TEAM_B,
)


class Phase(Enum):
    FACEOFF = auto()
    PLAY = auto()
    GOAL_SCORED = auto()
    PERIOD_END = auto()
    GAME_OVER = auto()


@dataclass
class Player:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    team: int = TEAM_A
    player_id: int = 0
    has_puck: bool = False
    facing_angle: float = 0.0  # radians, 0 = right
    check_cooldown: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def distance_to(self, other_x: float, other_y: float) -> float:
        return math.hypot(self.x - other_x, self.y - other_y)


@dataclass
class Puck:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    carrier: Player | None = None

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def is_free(self) -> bool:
        return self.carrier is None


@dataclass
class Goal:
    """A goal net on the rink edge."""
    team: int  # which team this goal belongs to (scoring here = point for opponent)
    x: float = 0.0
    y: float = 0.0
    width: float = GOAL_DEPTH
    height: float = GOAL_WIDTH

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class Rink:
    x: float = RINK_X
    y: float = RINK_Y
    width: float = RINK_WIDTH
    height: float = RINK_HEIGHT

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class GameState:
    players: list[Player] = field(default_factory=list)
    puck: Puck = field(default_factory=lambda: Puck(0, 0))
    goals: list[Goal] = field(default_factory=list)
    rink: Rink = field(default_factory=Rink)
    score: list[int] = field(default_factory=lambda: [0, 0])  # [team_a, team_b]
    period: int = 1
    time_remaining: float = 0.0
    phase: Phase = Phase.FACEOFF
    phase_timer: float = 0.0  # countdown for faceoff delay, goal celebration, etc.

    def team_players(self, team: int) -> list[Player]:
        return [p for p in self.players if p.team == team]

    def puck_carrier(self) -> Player | None:
        return self.puck.carrier

    def get_goal_for_team(self, team: int) -> Goal:
        """Get the goal that belongs to a team (scoring here = point for opponent)."""
        for g in self.goals:
            if g.team == team:
                return g
        raise ValueError(f"No goal for team {team}")

    def opponent_goal(self, team: int) -> Goal:
        """Get the goal a team is trying to score on."""
        opp = TEAM_B if team == TEAM_A else TEAM_A
        return self.get_goal_for_team(opp)


def create_goals(rink: Rink) -> list[Goal]:
    """Create the two goals on opposite ends of the rink."""
    goal_y = rink.center_y - GOAL_WIDTH / 2
    return [
        Goal(team=TEAM_A, x=rink.left - GOAL_DEPTH, y=goal_y),
        Goal(team=TEAM_B, x=rink.right, y=goal_y),
    ]
