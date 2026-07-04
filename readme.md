# sim-1

The **training layer** over the differentiable humanoid physics engine
(`external/engine`). The engine is mechanism-only; this repo owns the task/scene layer
(observation composition, reward, termination), the RL algorithms, and the training
infrastructure.

## Status: P0 scaffolding (runs against a mock env)

The engine's Python binding and the adopted humanoid model are being built separately. Until the
binding lands, everything runs against a **mock VecEnv** that matches the engine's binding contract
(`sim1/envs/vecenv.py`), so the whole stack is exercised now and swaps to the real engine in P1
with no changes to the task/algorithm code.

Implemented:
- **Mock env + reach task** (`sim1/envs/mock_vecenv.py`, `sim1/tasks/reach.py`) — a learnable
  stand-in honoring the SoA `VecEnv` contract.
- **PPO** (`sim1/algos/ppo.py`) — GAE with correct time-limit truncation bootstrapping, observation
  **and reward normalization** (reward divided by the running std of the discounted return, so the
  value scale stays O(1) and learning is robust across reward magnitudes), checkpointable state.
- **Training infra** — organized/resumable run dirs, TensorBoard + JSONL metrics, tqdm progress,
  periodic + best + final checkpoints (`sim1/utils/`).
- **Config** — plain dataclasses with dotted overrides (`sim1/config.py`); the full config
  *framework* (Hydra) is intentionally deferred.
- **Tests** — run locally with `pytest -q` (mock determinism, PPO-learns, reward-norm, checkpoint
  round-trip).

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # numpy, torch, tensorboard, tqdm, pytest
```

## Train / resume / eval

```bash
python -m sim1.train --name reach -o ppo.total_steps=300000 -o env.num_envs=128
python -m sim1.train --resume runs/<run_id>        # continue a run
python -m sim1.eval  --run runs/<run_id>           # deterministic rollout
tensorboard --logdir runs                          # live curves
python -m pytest -q                                # tests
```

Runs are written under `runs/<YYYYMMDD-HHMMSS>_<name>_<gitsha>/` (frozen `config.json`, `meta.json`,
`metrics.jsonl`, `tb/`, `checkpoints/`).

## Roadmap
- **P1** — wire the real `engine_py.VecEnv`; `stand` task; benchmark env-steps/s on the Linux CPU.
- **P2** — `walk` task + locomotion reward + curriculum.
- **P3** — SHAC (analytic policy gradient via the differentiable env).
- **P4** — tracking, sword/shield, two-humanoid combat.

Plan of record: `../research/notes/investigations/2026-07-04-sim1-training-infra-plan.md`.

## Layout
```
sim1/
  config.py            dataclass configs (+ dotted overrides)
  envs/                VecEnv contract, mock backend, obs norm, TaskEnv adapter
  tasks/               Task interface + reach (P0); stand/walk/combat later
  models/              Gaussian actor-critic MLP
  algos/               ppo (P0); shac later
  utils/               seeding, run_dir, logging, checkpoint
  train.py / eval.py   entrypoints
tests/                 mock determinism, PPO-learns, checkpoint round-trip
external/engine/       the C++ engine (submodule); csrc/ binding lands in P1
```
