"""Physics simulation for ice hockey."""

from __future__ import annotations

import math
from dataclasses import dataclass

from games.hockey.constants import (
    ACTION_CHECK,
    ACTION_DIRECTIONS,
    ACTION_NONE,
    ACTION_PASS,
    ACTION_SHOOT,
    BODY_CHECK_COOLDOWN,
    BODY_CHECK_FORCE,
    BODY_CHECK_RADIUS,
    CORNER_RADIUS,
    DT,
    ICE_FRICTION,
    PASS_SPEED,
    PLAYER_ACCEL,
    PLAYER_MAX_SPEED,
    PLAYER_RADIUS,
    PLAYER_RESTITUTION,
    PUCK_FRICTION,
    PUCK_MAX_SPEED,
    PUCK_PICKUP_RADIUS,
    PUCK_RADIUS,
    SHOT_SPEED,
    TEAM_A,
    TEAM_B,
    WALL_RESTITUTION,
)
from games.hockey.entities import GameState, Goal, Phase, Player, Puck, Rink


@dataclass
class Event:
    kind: str  # "goal", "shot", "pass", "hit", "wall_bounce", "pickup", "whistle"
    data: dict | None = None


def step_physics(state: GameState, team_a_actions: list[int], team_b_actions: list[int]) -> list[Event]:
    """Advance the game by one frame. Mutates state in place. Returns events."""
    events: list[Event] = []

    # Handle non-play phases
    if state.phase == Phase.FACEOFF:
        state.phase_timer -= DT
        if state.phase_timer <= 0:
            state.phase = Phase.PLAY
            events.append(Event("whistle"))
        return events

    if state.phase == Phase.GOAL_SCORED:
        state.phase_timer -= DT
        if state.phase_timer <= 0:
            _setup_faceoff(state)
        return events

    if state.phase == Phase.PERIOD_END:
        state.phase_timer -= DT
        if state.phase_timer <= 0:
            if state.period < 3:
                state.period += 1
                state.time_remaining = 120.0
                _setup_faceoff(state)
            else:
                state.phase = Phase.GAME_OVER
        return events

    if state.phase == Phase.GAME_OVER:
        return events

    # --- PLAY phase ---
    # Update game clock
    state.time_remaining -= DT
    if state.time_remaining <= 0:
        state.time_remaining = 0
        state.phase = Phase.PERIOD_END
        state.phase_timer = 2.0
        events.append(Event("buzzer"))
        return events

    team_a_players = state.team_players(TEAM_A)
    team_b_players = state.team_players(TEAM_B)

    # Apply actions
    for player, action in zip(team_a_players, team_a_actions):
        evts = _apply_action(player, action, state)
        events.extend(evts)
    for player, action in zip(team_b_players, team_b_actions):
        evts = _apply_action(player, action, state)
        events.extend(evts)

    # Update player positions
    for p in state.players:
        _update_player(p)

    # Update puck
    puck_events = _update_puck(state.puck, state.rink)
    events.extend(puck_events)

    # Puck pickup
    if state.puck.is_free:
        pickup_events = _check_puck_pickup(state.players, state.puck)
        events.extend(pickup_events)

    # Player-player collisions
    _resolve_player_collisions(state.players)

    # Player-wall collisions
    for p in state.players:
        _clamp_to_rink(p, state.rink)

    # Check goals
    goal_event = _check_goal(state.puck, state.goals, state.rink)
    if goal_event:
        events.append(goal_event)
        scoring_team = goal_event.data["scoring_team"]
        state.score[scoring_team] += 1
        state.phase = Phase.GOAL_SCORED
        state.phase_timer = 2.0

    # Decrease check cooldowns
    for p in state.players:
        if p.check_cooldown > 0:
            p.check_cooldown = max(0, p.check_cooldown - DT)

    return events


def _apply_action(player: Player, action: int, state: GameState) -> list[Event]:
    """Apply a single action for a player. Returns events."""
    events = []

    if action in ACTION_DIRECTIONS:
        dx, dy = ACTION_DIRECTIONS[action]
        player.vx += dx * PLAYER_ACCEL * DT
        player.vy += dy * PLAYER_ACCEL * DT
        player.facing_angle = math.atan2(dy, dx)

    elif action == ACTION_SHOOT and player.has_puck:
        _execute_shot(player, state.puck, state)
        events.append(Event("shot", {"player": player.player_id, "team": player.team}))

    elif action == ACTION_PASS and player.has_puck:
        passed = _execute_pass(player, state.puck, state)
        if passed:
            events.append(Event("pass", {"player": player.player_id, "team": player.team}))

    elif action == ACTION_CHECK:
        hit = _execute_body_check(player, state)
        if hit:
            events.append(Event("hit", {
                "checker": player.player_id,
                "target": hit.player_id,
                "checker_team": player.team,
            }))

    return events


def _update_player(player: Player) -> None:
    """Apply friction and clamp speed."""
    player.vx *= ICE_FRICTION
    player.vy *= ICE_FRICTION

    speed = player.speed
    if speed > PLAYER_MAX_SPEED:
        scale = PLAYER_MAX_SPEED / speed
        player.vx *= scale
        player.vy *= scale

    player.x += player.vx * DT
    player.y += player.vy * DT


def _update_puck(puck: Puck, rink: Rink) -> list[Event]:
    """Update puck position. If carried, follow carrier. Otherwise physics."""
    events = []
    if puck.carrier is not None:
        # Puck stays with carrier, offset in facing direction
        p = puck.carrier
        offset = PLAYER_RADIUS + PUCK_RADIUS + 2
        puck.x = p.x + math.cos(p.facing_angle) * offset
        puck.y = p.y + math.sin(p.facing_angle) * offset
        puck.vx = p.vx
        puck.vy = p.vy
        return events

    # Free puck physics
    puck.vx *= PUCK_FRICTION
    puck.vy *= PUCK_FRICTION

    speed = puck.speed
    if speed > PUCK_MAX_SPEED:
        scale = PUCK_MAX_SPEED / speed
        puck.vx *= scale
        puck.vy *= scale

    puck.x += puck.vx * DT
    puck.y += puck.vy * DT

    # Wall bounces
    bounced = _bounce_puck_off_walls(puck, rink)
    if bounced:
        events.append(Event("wall_bounce"))

    return events


def _bounce_puck_off_walls(puck: Puck, rink: Rink) -> bool:
    """Bounce puck off rink walls. Returns True if bounced."""
    bounced = False
    r = PUCK_RADIUS

    # Simple rectangular bounds first, then corner adjustment
    # Top wall
    if puck.y - r < rink.top:
        puck.y = rink.top + r
        puck.vy = abs(puck.vy) * WALL_RESTITUTION
        bounced = True
    # Bottom wall
    if puck.y + r > rink.bottom:
        puck.y = rink.bottom - r
        puck.vy = -abs(puck.vy) * WALL_RESTITUTION
        bounced = True
    # Left wall (but not in goal area)
    if puck.x - r < rink.left:
        goal_top = rink.center_y - 40  # half goal width
        goal_bot = rink.center_y + 40
        if not (goal_top < puck.y < goal_bot):
            puck.x = rink.left + r
            puck.vx = abs(puck.vx) * WALL_RESTITUTION
            bounced = True
    # Right wall (but not in goal area)
    if puck.x + r > rink.right:
        goal_top = rink.center_y - 40
        goal_bot = rink.center_y + 40
        if not (goal_top < puck.y < goal_bot):
            puck.x = rink.right - r
            puck.vx = -abs(puck.vx) * WALL_RESTITUTION
            bounced = True

    # Corner bounces (rounded corners)
    corners = [
        (rink.left + CORNER_RADIUS, rink.top + CORNER_RADIUS),
        (rink.right - CORNER_RADIUS, rink.top + CORNER_RADIUS),
        (rink.left + CORNER_RADIUS, rink.bottom - CORNER_RADIUS),
        (rink.right - CORNER_RADIUS, rink.bottom - CORNER_RADIUS),
    ]
    for cx, cy in corners:
        dx = puck.x - cx
        dy = puck.y - cy
        # Only check if puck is in the corner quadrant
        in_corner = False
        if cx < rink.center_x and cy < rink.center_y:  # top-left
            in_corner = puck.x < cx and puck.y < cy
        elif cx > rink.center_x and cy < rink.center_y:  # top-right
            in_corner = puck.x > cx and puck.y < cy
        elif cx < rink.center_x and cy > rink.center_y:  # bottom-left
            in_corner = puck.x < cx and puck.y > cy
        else:  # bottom-right
            in_corner = puck.x > cx and puck.y > cy

        if in_corner:
            dist = math.hypot(dx, dy)
            if dist > 0 and dist > CORNER_RADIUS - r:
                # Push puck out and reflect velocity
                nx, ny = dx / dist, dy / dist
                puck.x = cx + nx * (CORNER_RADIUS - r)
                puck.y = cy + ny * (CORNER_RADIUS - r)
                # Reflect velocity off normal
                dot = puck.vx * nx + puck.vy * ny
                if dot < 0:  # moving into the corner
                    puck.vx -= 2 * dot * nx
                    puck.vy -= 2 * dot * ny
                    puck.vx *= WALL_RESTITUTION
                    puck.vy *= WALL_RESTITUTION
                    bounced = True

    return bounced


def _check_puck_pickup(players: list[Player], puck: Puck) -> list[Event]:
    """Check if any player picks up a free puck."""
    events = []
    closest: Player | None = None
    closest_dist = float("inf")

    for p in players:
        dist = p.distance_to(puck.x, puck.y)
        if dist < PUCK_PICKUP_RADIUS and dist < closest_dist:
            closest = p
            closest_dist = dist

    if closest is not None:
        closest.has_puck = True
        puck.carrier = closest
        events.append(Event("pickup", {"player": closest.player_id, "team": closest.team}))

    return events


def _execute_shot(player: Player, puck: Puck, state: GameState) -> None:
    """Shoot the puck toward the opponent goal."""
    opp_goal = state.opponent_goal(player.team)
    goal_x = opp_goal.x + opp_goal.width / 2
    goal_y = opp_goal.center_y

    dx = goal_x - player.x
    dy = goal_y - player.y
    dist = math.hypot(dx, dy)
    if dist < 1:
        dist = 1

    # Release puck
    player.has_puck = False
    puck.carrier = None
    puck.x = player.x + math.cos(player.facing_angle) * (PLAYER_RADIUS + PUCK_RADIUS + 4)
    puck.y = player.y + math.sin(player.facing_angle) * (PLAYER_RADIUS + PUCK_RADIUS + 4)
    puck.vx = (dx / dist) * SHOT_SPEED
    puck.vy = (dy / dist) * SHOT_SPEED


def _execute_pass(player: Player, puck: Puck, state: GameState) -> bool:
    """Pass to the nearest teammate. Returns True if pass was made."""
    teammates = [p for p in state.players if p.team == player.team and p is not player]
    if not teammates:
        return False

    # Find nearest teammate
    target = min(teammates, key=lambda t: player.distance_to(t.x, t.y))

    dx = target.x - player.x
    dy = target.y - player.y
    dist = math.hypot(dx, dy)
    if dist < 1:
        return False

    player.has_puck = False
    puck.carrier = None
    puck.x = player.x + math.cos(player.facing_angle) * (PLAYER_RADIUS + PUCK_RADIUS + 4)
    puck.y = player.y + math.sin(player.facing_angle) * (PLAYER_RADIUS + PUCK_RADIUS + 4)
    puck.vx = (dx / dist) * PASS_SPEED
    puck.vy = (dy / dist) * PASS_SPEED
    return True


def _execute_body_check(player: Player, state: GameState) -> Player | None:
    """Body check nearest opponent in range. Returns hit player or None."""
    if player.check_cooldown > 0:
        return None

    opponents = [p for p in state.players if p.team != player.team]
    for opp in opponents:
        dist = player.distance_to(opp.x, opp.y)
        if dist < BODY_CHECK_RADIUS:
            # Apply impulse to opponent
            dx = opp.x - player.x
            dy = opp.y - player.y
            d = math.hypot(dx, dy)
            if d < 1:
                d = 1
            opp.vx += (dx / d) * BODY_CHECK_FORCE
            opp.vy += (dy / d) * BODY_CHECK_FORCE

            # Knock puck loose
            if opp.has_puck:
                opp.has_puck = False
                state.puck.carrier = None
                # Puck flies off in a random-ish direction
                state.puck.vx = opp.vx * 0.5
                state.puck.vy = opp.vy * 0.5

            player.check_cooldown = BODY_CHECK_COOLDOWN
            return opp

    return None


def _resolve_player_collisions(players: list[Player]) -> None:
    """Simple elastic-ish collisions between all players."""
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            a, b = players[i], players[j]
            dx = b.x - a.x
            dy = b.y - a.y
            dist = math.hypot(dx, dy)
            min_dist = PLAYER_RADIUS * 2

            if dist < min_dist and dist > 0:
                # Push apart
                overlap = min_dist - dist
                nx, ny = dx / dist, dy / dist
                a.x -= nx * overlap / 2
                a.y -= ny * overlap / 2
                b.x += nx * overlap / 2
                b.y += ny * overlap / 2

                # Exchange velocity component along collision normal
                rel_vx = a.vx - b.vx
                rel_vy = a.vy - b.vy
                dot = rel_vx * nx + rel_vy * ny
                if dot > 0:  # approaching
                    a.vx -= dot * nx * PLAYER_RESTITUTION
                    a.vy -= dot * ny * PLAYER_RESTITUTION
                    b.vx += dot * nx * PLAYER_RESTITUTION
                    b.vy += dot * ny * PLAYER_RESTITUTION


def _clamp_to_rink(player: Player, rink: Rink) -> None:
    """Keep player inside rink bounds."""
    r = PLAYER_RADIUS

    # Simple rectangular clamping (ignore rounded corners for players)
    if player.x - r < rink.left:
        player.x = rink.left + r
        player.vx = max(player.vx, 0)
    if player.x + r > rink.right:
        player.x = rink.right - r
        player.vx = min(player.vx, 0)
    if player.y - r < rink.top:
        player.y = rink.top + r
        player.vy = max(player.vy, 0)
    if player.y + r > rink.bottom:
        player.y = rink.bottom - r
        player.vy = min(player.vy, 0)


def _check_goal(puck: Puck, goals: list[Goal], rink: Rink) -> Event | None:
    """Check if the puck has entered either goal."""
    if puck.carrier is not None:
        return None

    for goal in goals:
        # Goal A (left side): puck goes past left edge within goal height
        if goal.team == TEAM_A:
            if puck.x - PUCK_RADIUS < rink.left and goal.top < puck.y < goal.bottom:
                return Event("goal", {"scoring_team": TEAM_B})
        # Goal B (right side): puck goes past right edge within goal height
        elif goal.team == TEAM_B:
            if puck.x + PUCK_RADIUS > rink.right and goal.top < puck.y < goal.bottom:
                return Event("goal", {"scoring_team": TEAM_A})

    return None


def _setup_faceoff(state: GameState) -> None:
    """Reset positions for a faceoff."""
    from games.hockey.constants import FACEOFF_A_POSITIONS, FACEOFF_B_POSITIONS, FACEOFF_DELAY_SEC

    rink = state.rink
    team_a = state.team_players(TEAM_A)
    team_b = state.team_players(TEAM_B)

    for i, p in enumerate(team_a):
        fx, fy = FACEOFF_A_POSITIONS[i]
        p.x = rink.x + fx * rink.width
        p.y = rink.y + fy * rink.height
        p.vx = p.vy = 0.0
        p.has_puck = False
        p.check_cooldown = 0.0

    for i, p in enumerate(team_b):
        fx, fy = FACEOFF_B_POSITIONS[i]
        p.x = rink.x + fx * rink.width
        p.y = rink.y + fy * rink.height
        p.vx = p.vy = 0.0
        p.has_puck = False
        p.check_cooldown = 0.0

    # Puck at center
    state.puck.x = rink.center_x
    state.puck.y = rink.center_y
    state.puck.vx = state.puck.vy = 0.0
    state.puck.carrier = None

    state.phase = Phase.FACEOFF
    state.phase_timer = FACEOFF_DELAY_SEC
