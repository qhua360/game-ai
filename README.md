# JEPA Game AI

AI opponents that actually learn, powered by a JEPA world model. Starting with 2D ice hockey.

## What is this?

Sports game AI today is scripted — behavior trees, finite state machines, predetermined responses. Players learn the patterns and exploit them. The AI is "on rails."

This project replaces that with a **JEPA (Joint Embedding Predictive Architecture) world model** — a neural network that learns how the game world works by watching gameplay, then **imagines future outcomes** to pick the best action. The result is AI that generalizes to novel situations instead of following a script.

### How the AI works

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
- **Gymnasium API** — standard RL interface for training

## Quick Start

```bash
# Install
pip install -e .

# Play against scripted bots
python -m games.hockey.human_play

# Collect training data
python -m data.collector --episodes 10000

# Train the JEPA world model
python -m model.train --data data/trajectories/ --epochs 100

# Evaluate JEPA agent vs scripted bots
python -m training.evaluation
```

## Project Structure

```
core/               # Game-agnostic AI framework
  base_env.py       #   Abstract environment interface
  world_model.py    #   JEPA world model (works with any game)
  planner.py        #   MPC + CEM action planner
  trainer.py        #   Training pipeline

games/hockey/       # Ice hockey game
  env.py            #   Gymnasium environment
  physics.py        #   Ice/puck/player physics
  renderer.py       #   Pygame rendering

ai/                 # AI agents
  scripted_bot.py   #   Rule-based baseline bots
  jepa_agent.py     #   JEPA world model agent
  policy_net.py     #   Distilled policy for deployment

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

- **Python 3.11+**
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
