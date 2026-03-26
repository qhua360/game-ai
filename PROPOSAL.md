# JEPA-Powered Ice Hockey Game AI

## Vision

Build a 2D ice hockey game with AI opponents powered by a JEPA (Joint Embedding Predictive Architecture) world model — AI that learns game dynamics from visual observations and generalizes to novel situations, instead of following scripted behavior trees.

The AI architecture is designed to be **pluggable across game types** (soccer, basketball, FPS) and **portable to mobile/indie** via model export and engine rewrite.

## The Problem with Current Sports Game AI

Games like NHL and FIFA use behavior trees, finite state machines, and scripted behaviors. The result:

- **Predictable**: Players quickly learn AI patterns and exploit them
- **On rails**: AI is constrained by pre-determined animation sets and decision trees
- **No generalization**: Different difficulty levels just tweak parameters (speed, accuracy), not strategy
- **No learning**: AI can't adapt to individual player styles

## Our Approach: JEPA World Model

Instead of hand-coding every decision, we train a neural network that **learns how the game world works** from raw pixel observations.

### What is JEPA?

JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning framework created by Yann LeCun at Meta. Unlike generative models that try to reconstruct every pixel, JEPA learns compact **latent representations** — abstract embeddings that capture the essential dynamics of the world while ignoring irrelevant noise.

Our reference implementation is [LeWorldModel](https://github.com/lucas-maes/le-wm) ([paper](https://arxiv.org/html/2603.19312v1)):
- 15M parameters (tiny by modern standards)
- Trains on a single GPU in hours
- Plans actions in under 1 second
- Two-term loss function (prediction + anti-collapse regularization)
- Demonstrated physical understanding through latent space probing

### How It Works

```
Game Frame (pixels) → Encoder (ViT) → Latent Embedding (192-dim)
                                              ↓
Actions + Current Latent → Predictor (Transformer) → Predicted Next Latent
                                              ↓
Planner (MPC + CEM) → Compare predicted futures → Pick best action sequence
```

1. **Encoder** sees the game frame and compresses it into a compact 192-dimensional vector
2. **Predictor** takes that vector + candidate actions and predicts what happens next (in latent space, not pixels)
3. **Planner** imagines hundreds of possible action sequences, rolls each forward through the predictor, and picks the one leading to the best outcome (scoring a goal)

The AI essentially **imagines the future** before acting — like a hockey player reading the play.

## Game Design

### Ice Hockey (3v3)

- **2D top-down** perspective
- **Ice physics**: Low-friction skating with momentum, puck sliding with damping
- **Team play**: 3 skaters per side (no dedicated goalie initially, any player can defend)
- **Actions**: 8 skate directions + shoot + pass + body check
- **Playable**: Human controls one player with keyboard, AI handles teammates and opponents

### Why Hockey?

- Fast-paced with continuous physics (momentum, ice friction, puck dynamics)
- Rich in emergent strategy (positioning, passing lanes, forechecking)
- Smaller team size than soccer = more tractable for initial AI training
- Underserved in indie games = more unique product

## Architecture

```
core/                  # Game-agnostic AI framework
  base_env.py          # Abstract environment interface
  world_model.py       # JEPA world model
  planner.py           # MPC + CEM planner
  trainer.py           # Training pipeline

games/
  hockey/              # Ice hockey (first game)
    env.py             # Gymnasium-compatible environment
    physics.py         # Ice/puck/player physics
    renderer.py        # Pygame rendering

ai/
  scripted_bot.py      # Baseline rule-based bots
  jepa_agent.py        # JEPA-powered agent
  policy_net.py        # Distilled policy for deployment

model/
  encoder.py           # ViT-Tiny encoder
  predictor.py         # Transformer predictor
  sigreg.py            # Anti-collapse regularization

data/
  collector.py         # Trajectory collection
  dataset.py           # PyTorch dataset

training/
  self_play.py         # Self-play improvement loop
  evaluation.py        # Win rate & generalization metrics
```

### Pluggable Design

To add a new game (soccer, FPS, etc.):
1. Implement the `base_env.py` interface
2. Collect gameplay data
3. Retrain the JEPA model

Same encoder architecture, same predictor, same planner. Only the input resolution and action space change.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Game Engine | Pygame 2.x (prototype) → Godot 4 / Unity (production) |
| AI Training | PyTorch 2.x |
| RL Interface | Gymnasium |
| Model Export | ONNX (for engine port / mobile) |
| Data Storage | NumPy .npz / HDF5 |

## Milestones

| # | Milestone | Description |
|---|-----------|-------------|
| M1 | Playable Game | Hockey game with scripted bots, fun to play |
| M2 | Data Pipeline | 50K+ trajectory frames collected |
| M3 | World Model | JEPA trains, latent space encodes game state |
| M4 | Planning Agent | JEPA agent beats scripted bots (>70% win rate) |
| M5 | Self-Play | AI improves through self-play iterations |
| M6 | Generalization | AI handles novel formations, rule changes, team sizes |
| M7 | Polish & Port | Fun game, challenging AI, ready for indie release |

## Portability Path

1. **Prototype** in Pygame (pure Python — fastest AI iteration)
2. **Distill** the JEPA planner into a lightweight policy network (<1M params)
3. **Export** to ONNX for cross-platform inference
4. **Port** game to Godot 4 or Unity for mobile/PC/web/console distribution
5. The distilled policy runs in a single forward pass — trivially fast on any device

## Related Work

- **[LeWorldModel](https://github.com/lucas-maes/le-wm)** — Our reference JEPA implementation
- **[Gran Turismo Sophy](https://www.gran-turismo.com/us/gran-turismo-sophy/)** — Sony's superhuman racing AI via deep RL
- **[AlphaStar](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)** — DeepMind's StarCraft II AI
- **[Google Research Football](https://github.com/google-research/football)** — RL environment for soccer
- **[V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)** — Meta's video JEPA model
