"""Game constants for ice hockey."""

import math

# -----------------------------------------------------------------------------
# Window / Rink
# -----------------------------------------------------------------------------
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 600
RINK_WIDTH = 900
RINK_HEIGHT = 500
RINK_X = (WINDOW_WIDTH - RINK_WIDTH) // 2   # top-left of rink area
RINK_Y = (WINDOW_HEIGHT - RINK_HEIGHT) // 2
CORNER_RADIUS = 40

# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------
FPS = 60
DT = 1.0 / FPS
ICE_FRICTION = 0.985          # per-frame velocity multiplier for players
PUCK_FRICTION = 0.993         # puck slides further
PLAYER_ACCEL = 800.0          # px/s²
PLAYER_MAX_SPEED = 300.0      # px/s
PUCK_MAX_SPEED = 600.0
SHOT_SPEED = 500.0
PASS_SPEED = 350.0
BODY_CHECK_RADIUS = 28.0
BODY_CHECK_FORCE = 450.0
BODY_CHECK_COOLDOWN = 0.8     # seconds
PLAYER_RADIUS = 12.0
PUCK_RADIUS = 6.0
PUCK_PICKUP_RADIUS = 18.0     # how close player must be to grab free puck
WALL_RESTITUTION = 0.7        # energy kept on wall bounce
PLAYER_RESTITUTION = 0.4      # energy kept on player-player collision

# Shooting direction helpers
SHOOT_TOWARD_GOAL = True      # auto-aim toward opponent goal center

# -----------------------------------------------------------------------------
# Goals
# -----------------------------------------------------------------------------
GOAL_DEPTH = 10.0             # how far the net extends behind the goal line
GOAL_WIDTH = 80.0             # vertical opening of the goal

# -----------------------------------------------------------------------------
# Teams & Players
# -----------------------------------------------------------------------------
PLAYERS_PER_TEAM = 3
TEAM_A = 0  # attacks right
TEAM_B = 1  # attacks left

# Starting positions (fractional rink coordinates, 0-1)
# Team A starts on left, Team B on right
TEAM_A_POSITIONS = [
    (0.25, 0.3),   # forward top
    (0.25, 0.7),   # forward bottom
    (0.15, 0.5),   # defender / goalie-ish
]
TEAM_B_POSITIONS = [
    (0.75, 0.3),
    (0.75, 0.7),
    (0.85, 0.5),
]

FACEOFF_A_POSITIONS = [
    (0.45, 0.5),   # center
    (0.35, 0.35),
    (0.35, 0.65),
]
FACEOFF_B_POSITIONS = [
    (0.55, 0.5),
    (0.65, 0.35),
    (0.65, 0.65),
]

# -----------------------------------------------------------------------------
# Actions (discrete)
# -----------------------------------------------------------------------------
ACTION_NONE = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_UP_LEFT = 5
ACTION_UP_RIGHT = 6
ACTION_DOWN_LEFT = 7
ACTION_DOWN_RIGHT = 8
ACTION_SHOOT = 9
ACTION_PASS = 10
ACTION_CHECK = 11
NUM_ACTIONS = 12

# Direction vectors for movement actions (normalized)
_s = math.sqrt(2) / 2
ACTION_DIRECTIONS = {
    ACTION_UP:         ( 0.0, -1.0),
    ACTION_DOWN:       ( 0.0,  1.0),
    ACTION_LEFT:       (-1.0,  0.0),
    ACTION_RIGHT:      ( 1.0,  0.0),
    ACTION_UP_LEFT:    (-_s,   -_s),
    ACTION_UP_RIGHT:   ( _s,   -_s),
    ACTION_DOWN_LEFT:  (-_s,    _s),
    ACTION_DOWN_RIGHT: ( _s,    _s),
}

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
ICE_COLOR = (220, 235, 245)
RINK_BORDER_COLOR = (40, 60, 100)
RED_LINE_COLOR = (200, 40, 40)
BLUE_LINE_COLOR = (40, 40, 200)
TEAM_A_COLOR = (220, 50, 50)
TEAM_B_COLOR = (50, 100, 220)
PUCK_COLOR = (20, 20, 20)
GOAL_COLOR = (180, 30, 30)
GOAL_NET_COLOR = (200, 200, 200)
CREASE_COLOR = (180, 210, 230)
FACEOFF_DOT_COLOR = (180, 40, 40)
HUD_BG_COLOR = (30, 30, 50)
HUD_TEXT_COLOR = (240, 240, 240)
HIGHLIGHT_COLOR = (255, 255, 100, 120)

# -----------------------------------------------------------------------------
# Observation (for JEPA)
# -----------------------------------------------------------------------------
OBS_WIDTH = 84
OBS_HEIGHT = 84

# -----------------------------------------------------------------------------
# Game Rules
# -----------------------------------------------------------------------------
PERIOD_LENGTH_SEC = 120.0     # 2-minute periods
NUM_PERIODS = 3
FACEOFF_DELAY_SEC = 1.5
GOAL_CELEBRATION_SEC = 2.0

# -----------------------------------------------------------------------------
# Sound
# -----------------------------------------------------------------------------
SAMPLE_RATE = 44100
SOUND_ENABLED = True
MASTER_VOLUME = 0.6
SFX_VOLUME = 0.7
AMBIENT_VOLUME = 0.3
