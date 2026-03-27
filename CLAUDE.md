# CLAUDE.md — JEPA Game AI Project

## Project Overview

JEPA-powered ice hockey game AI. Pygame prototype with PyTorch for AI training. The AI uses a world model (JEPA architecture) to learn game dynamics from pixels and plan actions, rather than scripted behavior trees.

## Architecture

### Implemented
- `games/hockey/` — Ice hockey game
  - `constants.py` — all tuning knobs (physics, rink dimensions, colors, sound)
  - `entities.py` — Player, Puck, Goal, Rink, GameState dataclasses
  - `physics.py` — ice friction, collisions, shots, passes, goal detection, event system
  - `sound.py` — numpy-synthesized sound effects (no asset files)
  - `renderer.py` — Pygame top-down rendering with HUD and visual effects
  - `env.py` — Gymnasium-compatible environment (84x84 RGB obs)
  - `human_play.py` — keyboard-controlled game loop
- `ai/` — AI agents
  - `scripted_bot.py` — rule-based bot with easy/medium/hard difficulty
- `model/` — JEPA world model (~12.5M params)
  - `encoder.py` — ViT-Tiny via `stable_pretraining.backbone.utils.vit_hf()` (84x84 → 192-dim)
  - `predictor.py` — 6-layer AR Transformer with AdaLN action conditioning
  - `sigreg.py` — SIGReg anti-collapse regularization (Epps-Pulley Gaussianity test)
  - `world_model.py` — LeWM combining encoder + predictor, with rollout() for planning
  - `train.py` — Training via spt.Module + spt.Manager with WandB logging
- `data/` — Data collection pipeline
  - `collector.py` — parallel bot-vs-bot trajectory collection, streams to HDF5, PLAY-phase only
  - `dataset.py` — PyTorch Dataset with configurable frameskip and stride
  - `merge_shards.py` — streaming merge of parallel collection shards
- `config/train/hockey.yaml` — Hydra training config

### Planned
- `ai/jepa_agent.py` — JEPA planning agent (MPC + discrete CEM)
- `core/` — Game-agnostic AI framework (base env, planner, trainer)
- `training/` — Self-play loop, curriculum, evaluation

## Tech Stack

- Python 3.11+ (managed with uv)
- Pygame 2.x (game)
- PyTorch 2.x (AI training)
- stable-pretraining (ViT backbone, Lightning training framework)
- stable-worldmodel (CEM solver, data utilities)
- Gymnasium (RL interface)
- NumPy, h5py (data storage)
- WandB (experiment tracking)

## Commands

```bash
# Install dependencies
uv sync
uv pip install -e ../stable-pretraining

# Play the game
uv run python -m games.hockey

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Collect training data (50 complete games, PLAY-phase only)
uv run python -m data.collector --episodes 50 --output data/trajectories/train.h5

# Train JEPA world model (20 epochs, ~3.5h on MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m model.train \
    --data data/trajectories/train.h5 --epochs 20 --batch-size 128

# Headless bot-vs-bot smoke test
uv run python -c "
import os; os.environ['SDL_VIDEODRIVER']='dummy'; os.environ['SDL_AUDIODRIVER']='dummy'
import pygame; pygame.init(); pygame.display.set_mode((100,100))
from games.hockey.env import HockeyEnv
from ai.scripted_bot import ScriptedBot
env = HockeyEnv(render_mode='rgb_array')
bot_a, bot_b = ScriptedBot(0), ScriptedBot(1)
obs, info = env.reset()
for _ in range(600):
    obs, r, d, t, info = env.step(bot_a.get_actions(info['state']), bot_b.get_actions(info['state']))
print(f'Score: {info[\"score\"]}')
env.close()
"
```

## Key Design Decisions

- **Pluggable architecture**: `core/base_env.py` will define the interface. New games implement it. JEPA model stays the same.
- **Observation space**: 84x84 RGB frames (from Pygame renderer)
- **Action space**: Discrete — 8 directions + shoot + pass + check + none (12 actions per player)
- **3v3 hockey**: Enough complexity for team play, tractable for training
- **JEPA over generative models**: Learns in latent space (192-dim), not pixel space. Faster, more stable, better generalization.
- **Synthesized sound**: All sound effects generated via numpy at startup — no external audio assets needed.
- **Event system**: `physics.step_physics()` returns Event objects that drive both sound and visual effects.

## Conventions

- Use dataclasses for game entities
- Type hints everywhere
- Gymnasium API for all environments (`reset`, `step`, `render`)
- PyTorch for all neural network code
- Tests in `tests/` directory, run with `uv run pytest`
- Use `uv` for all dependency management (not pip)
