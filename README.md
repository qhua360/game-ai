# JEPA Game AI

AI opponents that actually learn, powered by a JEPA world model. Starting with 2D ice hockey.

## What is this?

Sports game AI today is scripted — behavior trees, finite state machines, predetermined responses. Players learn the patterns and exploit them. The AI is "on rails."

This project replaces that with a **JEPA (Joint Embedding Predictive Architecture) world model** — a neural network that learns how the game world works by watching gameplay, then **imagines future outcomes** to pick the best action. The result is AI that generalizes to novel situations instead of following a script.

### How the AI works (planned)

```
Game Frame → ViT Encoder → Latent Embedding (192-dim)
                                    ↓
Current Latent + Actions → Transformer Predictor → Predicted Future
                                    ↓
MPC Planner → Imagine 300 futures → Pick the best one → Act
```

The AI "reads the play" — it encodes what it sees, imagines what could happen next for hundreds of possible actions, and picks the sequence that leads to scoring.

Based on [LeWorldModel](https://github.com/lucas-maes/le-wm) ([paper](https://arxiv.org/html/2603.19312v1)) — a 15M parameter JEPA that trains on a single GPU in hours and plans in under a second.

## The Game: Ice Hockey 3v3

Top-down 2D ice hockey with real physics:

- **Ice friction** — momentum-based skating, puck slides with damping
- **Team play** — 3v3 with passing, shooting, body checking
- **You play too** — control one player with keyboard, AI handles the rest
- **Sound effects** — synthesized audio (shots, hits, goal horn, whistle, buzzer) — no asset files needed
- **Gymnasium API** — standard RL interface for training

## Quick Start

```bash
# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Play against scripted bots
uv run python -m games.hockey

# Run tests
uv run pytest
```

### Controls

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move player |
| Space | Shoot toward opponent goal |
| P | Pass to nearest teammate |
| C | Body check |
| Tab | Switch controlled player |
| M | Mute / unmute sound |
| Esc | Quit |

## Project Structure

```
games/hockey/       # Ice hockey game
  constants.py      #   All tuning knobs (physics, rink, colors, sound)
  entities.py       #   Player, Puck, Goal, Rink, GameState dataclasses
  physics.py        #   Ice friction, collisions, shots, passes, goals
  sound.py          #   Numpy-synthesized sound effects (no asset files)
  renderer.py       #   Pygame top-down rendering with HUD
  env.py            #   Gymnasium-compatible environment
  human_play.py     #   Keyboard-controlled game loop

ai/                 # AI agents
  scripted_bot.py   #   Rule-based bot (easy/medium/hard difficulty)
```

### Planned (not yet implemented)

```
core/               # Game-agnostic AI framework
  base_env.py       #   Abstract environment interface
  world_model.py    #   JEPA world model (works with any game)
  planner.py        #   MPC + CEM action planner
  trainer.py        #   Training pipeline

model/              # JEPA components
  encoder.py        #   ViT-Tiny encoder
  predictor.py      #   Transformer predictor with AdaLN
  sigreg.py         #   Anti-collapse regularization

data/               # Data pipeline
  collector.py      #   Trajectory collection from gameplay
  dataset.py        #   PyTorch dataset for training

training/           # Training loop
  self_play.py      #   Self-play improvement
  evaluation.py     #   Win rate & generalization metrics
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_physics.py

# Headless bot-vs-bot smoke test (no window needed)
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
print(f'Score: RED {info[\"score\"][0]} - {info[\"score\"][1]} BLUE')
env.close()
"
```

## Pluggable Across Games

The AI architecture is game-agnostic. To add a new game:

1. Implement `core/base_env.py` (observations, actions, rewards)
2. Collect gameplay data
3. Retrain the JEPA model

Same encoder, same predictor, same planner — only the game changes. Designed to extend to soccer, basketball, FPS, and beyond.

## Portability

The endgame is a real, shippable game:

1. **Now** — Pygame prototype (fast AI iteration in pure Python)
2. **Later** — Distill the planner into a tiny policy network (<1M params)
3. **Ship** — Export to ONNX, port game to Godot/Unity, deploy to mobile/PC/web

## Tech Stack

- **Python 3.11+** — managed with [uv](https://docs.astral.sh/uv/)
- **Pygame 2.x** — game engine (prototype)
- **PyTorch 2.x** — AI training
- **Gymnasium** — RL environment interface
- **NumPy / HDF5** — data storage

## References

- [LeWorldModel](https://github.com/lucas-maes/le-wm) — Reference JEPA implementation
- [LeWM Paper](https://arxiv.org/html/2603.19312v1) — "LeWorldModel: A JEPA for Learning World Models"
- [V-JEPA (Meta)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) — Video JEPA
- [Gran Turismo Sophy](https://www.gran-turismo.com/us/gran-turismo-sophy/) — Sony's superhuman racing AI
- [AlphaStar](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) — DeepMind's StarCraft II AI

## License

TBD
