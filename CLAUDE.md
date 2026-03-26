# CLAUDE.md — JEPA Game AI Project

## Project Overview

JEPA-powered ice hockey game AI. Pygame prototype with PyTorch for AI training. The AI uses a world model (JEPA architecture) to learn game dynamics from pixels and plan actions, rather than scripted behavior trees.

## Architecture

- `core/` — Game-agnostic AI framework (base env, world model, planner, trainer)
- `games/hockey/` — Ice hockey game (env, physics, renderer)
- `ai/` — AI agents (scripted bots, JEPA agent, policy network)
- `model/` — JEPA model components (ViT encoder, Transformer predictor, SIGReg)
- `data/` — Data collection and dataset loading
- `training/` — Self-play loop, curriculum, evaluation

## Tech Stack

- Python 3.11+
- Pygame 2.x (game)
- PyTorch 2.x (AI training)
- Gymnasium (RL interface)
- NumPy, h5py (data storage)

## Commands

```bash
# Play the game
python -m games.hockey.human_play

# Collect training data
python -m data.collector --episodes 10000

# Train JEPA world model
python -m model.train --data data/trajectories/ --epochs 100

# Evaluate AI vs scripted bots
python -m training.evaluation

# Run self-play training loop
python -m training.self_play
```

## Key Design Decisions

- **Pluggable architecture**: `core/base_env.py` defines the interface. New games implement it. JEPA model stays the same.
- **Observation space**: 84x84 RGB frames (from Pygame renderer)
- **Action space**: Discrete — 8 directions + shoot + pass + check (11 actions per player)
- **3v3 hockey**: Enough complexity for team play, tractable for training
- **JEPA over generative models**: Learns in latent space (192-dim), not pixel space. Faster, more stable, better generalization.

## Conventions

- Use dataclasses for game entities
- Type hints everywhere
- Gymnasium API for all environments (`reset`, `step`, `render`)
- PyTorch for all neural network code
- Tests in `tests/` directory, run with `pytest`
