"""JEPA planning agent using world model + discrete CEM."""

from __future__ import annotations

import collections
import math
import os

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from ai.base_agent import Agent
from ai.scripted_bot import ScriptedBot
from games.hockey.constants import (
    ACTION_NONE,
    GOAL_DEPTH,
    GOAL_WIDTH,
    NUM_ACTIONS,
    OBS_HEIGHT,
    OBS_WIDTH,
    PLAYERS_PER_TEAM,
    RINK_HEIGHT,
    RINK_WIDTH,
    RINK_X,
    RINK_Y,
    TEAM_A,
    TEAM_B,
)
from games.hockey.entities import (
    GameState,
    Goal,
    Phase,
    Player,
    Puck,
    Rink,
    create_goals,
)
from games.hockey.renderer import Renderer
from model.world_model import LeWM


class DiscreteCEMSolver:
    """Cross-Entropy Method solver for discrete multi-player action spaces.

    Maintains categorical logits for each (timestep, player) and samples
    actions via multinomial. Elite selection updates logits toward the
    empirical distribution of top-K samples.
    """

    def __init__(
        self,
        model: LeWM,
        num_samples: int = 256,
        n_steps: int = 8,
        topk: int = 25,
        horizon: int = 5,
        num_controlled_players: int = PLAYERS_PER_TEAM,
        num_actions: int = NUM_ACTIONS,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = min(topk, num_samples)
        self.horizon = horizon
        self.num_players = num_controlled_players
        self.num_actions = num_actions
        self.temperature = temperature
        self.device = torch.device(device)

        # Warm-start logits from previous plan
        self._prev_logits: torch.Tensor | None = None

    @torch.inference_mode()
    def solve(
        self,
        initial_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        opponent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Run CEM optimization for discrete actions.

        Args:
            initial_emb: (1, history_size, D) context embeddings.
            goal_emb: (1, 1, D) goal state embedding.
            opponent_actions: (1, H, 3) predicted opponent actions (int64).

        Returns:
            (H, 3) int64 — optimized action sequence for controlled team.
        """
        H = self.horizon
        P = self.num_players
        A = self.num_actions
        S = self.num_samples

        # Initialize logits: uniform or warm-start from previous plan
        if self._prev_logits is not None:
            logits = self._warm_start(self._prev_logits)
        else:
            logits = torch.zeros(H, P, A, device=self.device)

        for step in range(self.n_steps):
            # Sample actions from categorical distribution
            probs = F.softmax(logits / self.temperature, dim=-1)  # (H, P, A)
            flat_probs = probs.reshape(H * P, A)  # (H*P, A)

            # Sample S action sequences
            # (H*P, S) indices, then reshape to (S, H, P)
            samples = torch.multinomial(
                flat_probs.unsqueeze(0).expand(S, -1, -1).reshape(S * H * P, A),
                num_samples=1,
            ).reshape(S, H, P)  # (S, H, P)

            # Build full 6-player actions: [team_a(3), team_b(3)]
            opp = opponent_actions.expand(S, -1, -1)  # (S, H, 3)
            full_actions = torch.cat([samples, opp], dim=-1)  # (S, H, 6)
            full_actions = full_actions.unsqueeze(0)  # (1, S, H, 6)

            # Evaluate cost
            costs = self.model.get_cost(initial_emb, goal_emb, full_actions)  # (1, S)
            costs = costs.squeeze(0)  # (S,)

            # Select top-K (lowest cost)
            _, elite_idx = costs.topk(self.topk, largest=False)
            elite_actions = samples[elite_idx]  # (topk, H, P)

            # Update logits from elite action frequencies
            new_logits = torch.zeros_like(logits)
            for h in range(H):
                for p in range(P):
                    counts = torch.bincount(
                        elite_actions[:, h, p], minlength=A
                    ).float()
                    # Smoothed log-probabilities
                    new_logits[h, p] = torch.log(counts + 1e-8)

            # Blend with previous logits (momentum)
            alpha = 0.5
            logits = alpha * logits + (1 - alpha) * new_logits

        # Save for warm-start
        self._prev_logits = logits.detach()

        # Return argmax actions
        return logits.argmax(dim=-1)  # (H, P)

    def _warm_start(self, prev_logits: torch.Tensor) -> torch.Tensor:
        """Shift logits left by 1 step, pad last with uniform."""
        H, P, A = prev_logits.shape
        new_logits = torch.zeros_like(prev_logits)
        new_logits[:-1] = prev_logits[1:]  # shift left
        # Last step is uniform (zeros = uniform after softmax)
        return new_logits

    def reset(self) -> None:
        """Clear warm-start state."""
        self._prev_logits = None


class JEPAAgent(Agent):
    """JEPA planning agent using world model + discrete CEM.

    Uses Model Predictive Control to select actions:
    1. Encode recent observations into latent embeddings
    2. Use CEM to find action sequences that move toward a goal state
    3. Execute the first few actions, then re-plan

    Plugs into the same interface as ScriptedBot.
    """

    def __init__(
        self,
        checkpoint_path: str,
        team: int = TEAM_A,
        device: str | torch.device = "mps",
        plan_interval: int = 6,
        horizon: int = 5,
        num_samples: int = 256,
        cem_iterations: int = 8,
        topk: int = 25,
        opponent_difficulty: str = "hard",
    ) -> None:
        super().__init__(team)
        self.device = torch.device(device)
        self.plan_interval = plan_interval
        self.horizon = horizon

        # Load world model
        self._model = LeWM()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict", checkpoint)
        # Handle spt.Module wrapping: keys may be prefixed with "model."
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v
        self._model.load_state_dict(cleaned, strict=False)
        self._model.to(self.device)
        self._model.eval()

        # CEM solver
        self._solver = DiscreteCEMSolver(
            model=self._model,
            num_samples=num_samples,
            n_steps=cem_iterations,
            topk=topk,
            horizon=horizon,
            device=self.device,
        )

        # Opponent model
        opp_team = TEAM_B if team == TEAM_A else TEAM_A
        self._opponent_bot = ScriptedBot(team=opp_team, difficulty=opponent_difficulty)

        # State
        self._obs_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=3)
        self._action_plan: collections.deque[list[int]] = collections.deque()
        self._frame_counter = 0

        # Goal embedding (generated lazily on first call)
        self._goal_emb: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset agent state between episodes."""
        self._obs_buffer.clear()
        self._action_plan.clear()
        self._frame_counter = 0
        self._solver.reset()

    def get_actions(self, state: GameState, obs: np.ndarray | None = None) -> list[int]:
        """Return one action per player on this team.

        Args:
            state: Current game state.
            obs: 84x84x3 RGB frame (required for JEPA encoding).

        Returns:
            List of 3 action ints.
        """
        # Non-play phases: do nothing
        if state.phase != Phase.PLAY:
            return [ACTION_NONE] * PLAYERS_PER_TEAM

        if obs is None:
            return [ACTION_NONE] * PLAYERS_PER_TEAM

        # Buffer observation
        self._obs_buffer.append(obs)
        self._frame_counter += 1

        # Return buffered action if available
        if self._action_plan:
            return self._action_plan.popleft()

        # Re-plan
        initial_emb = self._encode_obs_buffer()
        goal_emb = self._get_goal_embedding()
        opponent_actions = self._predict_opponent_actions(state)

        action_seq = self._solver.solve(initial_emb, goal_emb, opponent_actions)
        # action_seq: (H, 3) int64

        # Store plan (skip first action, return it immediately)
        actions_list = action_seq.cpu().tolist()
        for i in range(1, min(self.plan_interval, len(actions_list))):
            self._action_plan.append(actions_list[i])

        return actions_list[0]

    def _encode_obs_buffer(self) -> torch.Tensor:
        """Encode observation buffer into context embeddings.

        Returns:
            (1, T, D) where T = len(obs_buffer) (up to 3).
        """
        # Pad buffer if < 3 frames
        while len(self._obs_buffer) < 3:
            self._obs_buffer.appendleft(self._obs_buffer[0])

        # Stack: (3, H, W, C) → (1, 3, C, H, W) float32 [0,1]
        frames = np.stack(list(self._obs_buffer))  # (3, 84, 84, 3)
        tensor = torch.from_numpy(frames).float().to(self.device) / 255.0
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 3, 3, 84, 84)

        return self._model.encode(tensor)  # (1, 3, 192)

    def _get_goal_embedding(self) -> torch.Tensor:
        """Get or generate the goal state embedding.

        Returns:
            (1, 1, D) goal embedding.
        """
        if self._goal_emb is not None:
            return self._goal_emb

        self._goal_emb = self._generate_goal_embedding()
        return self._goal_emb

    def _generate_goal_embedding(self) -> torch.Tensor:
        """Render a synthetic goal-scored frame and encode it.

        Creates a GameState with the puck inside the opponent's goal,
        renders it headlessly, and encodes the frame.

        Returns:
            (1, 1, D) goal embedding tensor on self.device.
        """
        # Build a fake game state with puck in opponent's goal
        rink = Rink()
        goals = create_goals(rink)
        opp_goal = goals[TEAM_B] if self.team == TEAM_A else goals[TEAM_A]

        # Place puck inside the goal
        puck = Puck(
            x=opp_goal.x + opp_goal.width / 2,
            y=opp_goal.center_y,
        )

        # Place players in default positions
        players = []
        from games.hockey.constants import TEAM_A_POSITIONS, TEAM_B_POSITIONS

        for i, (fx, fy) in enumerate(TEAM_A_POSITIONS):
            players.append(Player(
                x=rink.x + fx * rink.width,
                y=rink.y + fy * rink.height,
                team=TEAM_A,
                player_id=i,
            ))
        for i, (fx, fy) in enumerate(TEAM_B_POSITIONS):
            players.append(Player(
                x=rink.x + fx * rink.width,
                y=rink.y + fy * rink.height,
                team=TEAM_B,
                player_id=PLAYERS_PER_TEAM + i,
                facing_angle=math.pi,
            ))

        state = GameState(
            players=players,
            puck=puck,
            goals=goals,
            rink=rink,
            score=[1, 0] if self.team == TEAM_A else [0, 1],
            period=1,
            time_remaining=60.0,
            phase=Phase.PLAY,
        )

        # Render headlessly
        renderer = Renderer(headless=True)
        obs = renderer.get_obs(state)  # (84, 84, 3) uint8
        renderer.cleanup()

        # Encode
        tensor = torch.from_numpy(obs).float().to(self.device) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 84, 84)
        emb = self._model.encode(tensor)  # (1, 1, D)
        return emb

    def _predict_opponent_actions(self, state: GameState) -> torch.Tensor:
        """Predict opponent actions for the planning horizon.

        Uses ScriptedBot for the current state, repeated across horizon.

        Args:
            state: Current game state.

        Returns:
            (1, H, 3) int64 opponent action tensor on self.device.
        """
        opp_actions = self._opponent_bot.get_actions(state)  # list of 3 ints
        # Repeat across horizon
        opp_tensor = torch.tensor(opp_actions, dtype=torch.long, device=self.device)
        opp_tensor = opp_tensor.unsqueeze(0).unsqueeze(0).expand(1, self.horizon, -1)
        return opp_tensor
