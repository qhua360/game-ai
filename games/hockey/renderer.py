"""Pygame renderer for ice hockey."""

from __future__ import annotations

import math

import numpy as np
import pygame

from games.hockey.constants import (
    BLUE_LINE_COLOR,
    CORNER_RADIUS,
    CREASE_COLOR,
    FACEOFF_DOT_COLOR,
    GOAL_COLOR,
    GOAL_DEPTH,
    GOAL_NET_COLOR,
    GOAL_WIDTH,
    HUD_BG_COLOR,
    HUD_TEXT_COLOR,
    ICE_COLOR,
    OBS_HEIGHT,
    OBS_WIDTH,
    PLAYER_RADIUS,
    PUCK_COLOR,
    PUCK_RADIUS,
    RED_LINE_COLOR,
    RINK_BORDER_COLOR,
    TEAM_A,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from games.hockey.entities import GameState, Phase


class Renderer:
    def __init__(self, headless: bool = False) -> None:
        self._headless = headless
        self._screen: pygame.Surface | None = None
        self._surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self._obs_surface = pygame.Surface((OBS_WIDTH, OBS_HEIGHT))
        self._font: pygame.font.Font | None = None
        self._big_font: pygame.font.Font | None = None
        self._puck_trail: list[tuple[float, float]] = []
        self._goal_flash_timer = 0.0

        if not headless:
            pygame.display.set_caption("JEPA Hockey")
            self._screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        pygame.font.init()
        self._font = pygame.font.SysFont("monospace", 18, bold=True)
        self._big_font = pygame.font.SysFont("monospace", 36, bold=True)

    def render(self, state: GameState, controlled_player_id: int | None = None) -> pygame.Surface:
        """Draw the full game scene. Returns the surface."""
        surf = self._surface
        rink = state.rink

        # Goal flash effect
        if state.phase == Phase.GOAL_SCORED:
            self._goal_flash_timer = max(0, state.phase_timer)
        else:
            self._goal_flash_timer = 0

        # Background
        surf.fill((30, 30, 40))

        # Ice
        ice_rect = pygame.Rect(rink.x, rink.y, rink.width, rink.height)
        if self._goal_flash_timer > 0 and int(self._goal_flash_timer * 6) % 2 == 0:
            pygame.draw.rect(surf, (255, 255, 220), ice_rect, border_radius=CORNER_RADIUS)
        else:
            pygame.draw.rect(surf, ICE_COLOR, ice_rect, border_radius=CORNER_RADIUS)

        # Rink markings
        self._draw_markings(surf, rink)

        # Goals
        self._draw_goals(surf, state)

        # Players
        for player in state.players:
            self._draw_player(surf, player, state, controlled_player_id)

        # Puck
        self._draw_puck(surf, state)

        # HUD
        self._draw_hud(surf, state)

        # Phase overlays
        if state.phase == Phase.FACEOFF:
            self._draw_centered_text(surf, "FACEOFF", alpha=180)
        elif state.phase == Phase.GOAL_SCORED:
            self._draw_centered_text(surf, "GOAL!", alpha=220)
        elif state.phase == Phase.PERIOD_END:
            self._draw_centered_text(surf, f"END OF PERIOD {state.period}", alpha=200)
        elif state.phase == Phase.GAME_OVER:
            winner = "RED WINS" if state.score[0] > state.score[1] else \
                     "BLUE WINS" if state.score[1] > state.score[0] else "TIE GAME"
            self._draw_centered_text(surf, f"GAME OVER — {winner}", alpha=230)

        # Blit to screen if not headless
        if self._screen is not None:
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()

        return surf

    def get_obs(self, state: GameState) -> np.ndarray:
        """Render at 84x84 for JEPA observation."""
        full = self.render(state)
        pygame.transform.smoothscale(full, (OBS_WIDTH, OBS_HEIGHT), self._obs_surface)
        arr = pygame.surfarray.array3d(self._obs_surface)
        # pygame gives (W, H, 3), transpose to (H, W, 3)
        return arr.transpose(1, 0, 2)

    def cleanup(self) -> None:
        pygame.font.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_markings(self, surf: pygame.Surface, rink) -> None:
        """Draw lines, circles, faceoff dots on the ice."""
        cx = rink.center_x
        cy = rink.center_y

        # Center red line
        pygame.draw.line(surf, RED_LINE_COLOR, (cx, rink.top + 5), (cx, rink.bottom - 5), 3)

        # Center circle
        pygame.draw.circle(surf, RED_LINE_COLOR, (int(cx), int(cy)), 50, 2)

        # Center dot
        pygame.draw.circle(surf, RED_LINE_COLOR, (int(cx), int(cy)), 5)

        # Blue lines (1/3 and 2/3 across)
        bx1 = rink.x + rink.width / 3
        bx2 = rink.x + 2 * rink.width / 3
        pygame.draw.line(surf, BLUE_LINE_COLOR, (bx1, rink.top + 5), (bx1, rink.bottom - 5), 3)
        pygame.draw.line(surf, BLUE_LINE_COLOR, (bx2, rink.top + 5), (bx2, rink.bottom - 5), 3)

        # Faceoff dots (4 in offensive/defensive zones)
        dot_positions = [
            (rink.x + rink.width * 0.2, rink.y + rink.height * 0.3),
            (rink.x + rink.width * 0.2, rink.y + rink.height * 0.7),
            (rink.x + rink.width * 0.8, rink.y + rink.height * 0.3),
            (rink.x + rink.width * 0.8, rink.y + rink.height * 0.7),
        ]
        for dx, dy in dot_positions:
            pygame.draw.circle(surf, FACEOFF_DOT_COLOR, (int(dx), int(dy)), 4)
            pygame.draw.circle(surf, FACEOFF_DOT_COLOR, (int(dx), int(dy)), 20, 1)

        # Rink border
        border_rect = pygame.Rect(rink.x, rink.y, rink.width, rink.height)
        pygame.draw.rect(surf, RINK_BORDER_COLOR, border_rect, 4, border_radius=CORNER_RADIUS)

    def _draw_goals(self, surf: pygame.Surface, state: GameState) -> None:
        """Draw goal nets on both sides."""
        for goal in state.goals:
            # Goal crease (semi-circle in front of goal)
            if goal.team == TEAM_A:
                crease_x = state.rink.left + 5
            else:
                crease_x = state.rink.right - 5
            crease_rect = pygame.Rect(crease_x - 25, goal.center_y - 30, 50, 60)
            pygame.draw.ellipse(surf, CREASE_COLOR, crease_rect)

            # Goal net rectangle
            net_rect = pygame.Rect(goal.x, goal.y, goal.width, goal.height)
            pygame.draw.rect(surf, GOAL_NET_COLOR, net_rect)
            pygame.draw.rect(surf, GOAL_COLOR, net_rect, 2)

    def _draw_player(
        self,
        surf: pygame.Surface,
        player,
        state: GameState,
        controlled_id: int | None,
    ) -> None:
        """Draw a single player."""
        color = TEAM_A_COLOR if player.team == TEAM_A else TEAM_B_COLOR
        px, py = int(player.x), int(player.y)
        r = PLAYER_RADIUS

        # Highlight controlled player
        if controlled_id is not None and player.player_id == controlled_id:
            pygame.draw.circle(surf, (255, 255, 100), (px, py), int(r + 5), 2)

        # Possession glow
        if player.has_puck:
            pygame.draw.circle(surf, (255, 255, 200), (px, py), int(r + 3), 2)

        # Player body
        pygame.draw.circle(surf, color, (px, py), int(r))
        pygame.draw.circle(surf, (255, 255, 255), (px, py), int(r), 1)

        # Direction indicator (small triangle)
        angle = player.facing_angle
        tip_x = px + math.cos(angle) * (r + 4)
        tip_y = py + math.sin(angle) * (r + 4)
        left_x = px + math.cos(angle + 2.5) * (r - 2)
        left_y = py + math.sin(angle + 2.5) * (r - 2)
        right_x = px + math.cos(angle - 2.5) * (r - 2)
        right_y = py + math.sin(angle - 2.5) * (r - 2)
        pygame.draw.polygon(surf, (255, 255, 255), [
            (int(tip_x), int(tip_y)),
            (int(left_x), int(left_y)),
            (int(right_x), int(right_y)),
        ])

        # Jersey number
        if self._font:
            try:
                num_text = self._font.render(str(player.player_id % 3 + 1), True, (255, 255, 255))
                text_rect = num_text.get_rect(center=(px, py))
                surf.blit(num_text, text_rect)
            except pygame.error:
                pass  # font not available in headless mode

    def _draw_puck(self, surf: pygame.Surface, state: GameState) -> None:
        """Draw the puck with optional speed trail."""
        puck = state.puck

        # Don't draw if carried (it's shown as glow on carrier)
        if puck.carrier is not None:
            # Draw small puck on carrier
            px, py = int(puck.x), int(puck.y)
            pygame.draw.circle(surf, PUCK_COLOR, (px, py), int(PUCK_RADIUS))
            self._puck_trail.clear()
            return

        px, py = int(puck.x), int(puck.y)

        # Trail for fast-moving puck
        self._puck_trail.append((puck.x, puck.y))
        if len(self._puck_trail) > 5:
            self._puck_trail.pop(0)

        if puck.speed > 100:
            for i, (tx, ty) in enumerate(self._puck_trail[:-1]):
                alpha = int(60 * (i + 1) / len(self._puck_trail))
                trail_r = max(2, PUCK_RADIUS - 2)
                trail_surf = pygame.Surface((trail_r * 2, trail_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(trail_surf, (*PUCK_COLOR, alpha), (trail_r, trail_r), trail_r)
                surf.blit(trail_surf, (int(tx) - trail_r, int(ty) - trail_r))

        # Main puck
        draw_r = PUCK_RADIUS + (1 if puck.speed > 200 else 0)
        pygame.draw.circle(surf, PUCK_COLOR, (px, py), int(draw_r))
        pygame.draw.circle(surf, (60, 60, 60), (px, py), int(draw_r), 1)

    def _draw_hud(self, surf: pygame.Surface, state: GameState) -> None:
        """Draw score, period, time at top."""
        if not self._font:
            return

        hud_h = 30
        hud_rect = pygame.Rect(0, 0, WINDOW_WIDTH, hud_h)
        pygame.draw.rect(surf, HUD_BG_COLOR, hud_rect)

        try:
            mins = int(state.time_remaining) // 60
            secs = int(state.time_remaining) % 60
            score_text = f"RED  {state.score[0]}  -  {state.score[1]}  BLUE"
            time_text = f"P{state.period}  {mins:02d}:{secs:02d}"

            score_surf = self._font.render(score_text, True, HUD_TEXT_COLOR)
            time_surf = self._font.render(time_text, True, HUD_TEXT_COLOR)

            surf.blit(score_surf, (WINDOW_WIDTH // 2 - score_surf.get_width() // 2, 5))
            surf.blit(time_surf, (WINDOW_WIDTH - time_surf.get_width() - 15, 5))

            hint = "Arrows:Move  Space:Shoot  P:Pass  C:Check  Tab:Switch  M:Mute"
            hint_surf = self._font.render(hint, True, (140, 140, 160))
            surf.blit(hint_surf, (10, WINDOW_HEIGHT - 22))
        except pygame.error:
            pass  # font not available in headless mode

    def _draw_centered_text(self, surf: pygame.Surface, text: str, alpha: int = 200) -> None:
        """Draw big centered text with semi-transparent background."""
        if not self._big_font:
            return
        try:
            text_surf = self._big_font.render(text, True, (255, 255, 255))
            tw, th = text_surf.get_size()
            padding = 20

            bg = pygame.Surface((tw + padding * 2, th + padding * 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, alpha))
            surf.blit(bg, (WINDOW_WIDTH // 2 - tw // 2 - padding, WINDOW_HEIGHT // 2 - th // 2 - padding))
            surf.blit(text_surf, (WINDOW_WIDTH // 2 - tw // 2, WINDOW_HEIGHT // 2 - th // 2))
        except pygame.error:
            pass  # font not available in headless mode
