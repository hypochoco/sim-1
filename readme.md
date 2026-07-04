# sim-1

The **training layer** over the differentiable humanoid physics engine
(`external/engine`). The engine is mechanism-only; this repo owns the task/scene layer
(observation composition, reward, termination), the RL algorithms, and the training
infrastructure.

> **Taking over the training?** Read **[`HANDOFF.md`](HANDOFF.md)** first — it has the full
> operational context: current state + verified results, run commands, the config surface, the
> gotchas (PD-target vs torque, substeps, divergence guard, CPU-bound sim), the file map, the
> prioritized next steps, and the exact git state.

## Status: P1 — training on the real engine

The engine's Python binding (`csrc/ → sim1.engine_py`, nanobind) is wired and builds **headless**
(the graphics-free `engine::training` aggregate; no Metal/Vulkan/GLFW). PPO trains both a **mock**
env and the **physically-simulated humanoid** through the same `VecEnv` contract, so nothing in the
task/algorithm code changes between them.

Implemented:
- **Engine binding** (`csrc/engine_py.cpp`, `sim1/envs/engine_vecenv.py`) — the batched `VecEnv`,
  `SimConfig`, and rig factories (21-DOF `make_humanoid` / 28-DOF `make_amp_humanoid`), exposed as
  zero-copy NumPy views and adapted to the `VecEnv` contract (`sim1/envs/vecenv.py`).
- **Mock env + reach task** — a learnable contract-honoring stand-in for fast, engine-free dev/CI.
- **Humanoid `stand` task** (`sim1/tasks/stand.py`) — uprightness + alive + height reward, fall
  termination, PD-target actuation; obs composed from root pose/twist + joint q/qd + contacts.
- **PPO** (`sim1/algos/ppo.py`) — GAE with time-limit truncation bootstrapping, observation **and
  reward normalization**, checkpointable state. A `TaskEnv` divergence guard resets any env whose
  physics blows up so one bad world can't NaN-poison the batch.
- **Training infra** — organized/resumable run dirs, TensorBoard + JSONL metrics, tqdm progress,
  periodic + best + final checkpoints (`sim1/utils/`).
- **Config** — plain dataclasses with dotted overrides (`sim1/config.py`); the config *framework*
  (Hydra) is intentionally deferred.
- **Tests** — `pytest -q` (mock determinism, PPO-learns, reward-norm, checkpoint round-trip).

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # runtime deps: numpy, torch, tensorboard, tqdm, pytest
pip install -e .                     # builds the engine_py binding (headless engine::training)
```

## Train / resume / eval

```bash
# humanoid stand (real engine, AMP 28-DOF rig, PD-target control)
python -m sim1.train --name stand_amp \
  -o env.kind=engine -o env.model=amp -o env.backend=reduced \
  -o env.num_envs=64 -o env.episode_len=300 -o env.substeps=48 \
  -o env.action_mode=pd_target -o task.name=stand -o ppo.total_steps=2000000

# fast engine-free mock (no binding needed)
python -m sim1.train --name reach -o ppo.total_steps=300000 -o env.num_envs=128

python -m sim1.train --resume runs/<run_id>        # continue a run
python -m sim1.eval  --run runs/<run_id>           # deterministic rollout
tensorboard --logdir runs                          # live curves
python -m pytest -q                                # tests
```

Runs are written under `runs/<YYYYMMDD-HHMMSS>_<name>_<gitsha>/` (frozen `config.json`, `meta.json`,
`metrics.jsonl`, `tb/`, `checkpoints/`). Key metrics: `charts/ep_return_mean`, `charts/ep_len_mean`
(rises as the humanoid falls less), `losses/*`.

## Roadmap
- **P1 (in progress)** — engine binding ✓, `stand` task ✓; next: benchmark env-steps/s on the Linux
  CPU, tune the stand reward/curriculum.
- **P2** — `walk` task + locomotion reward + curriculum.
- **P3** — SHAC (analytic policy gradient via the differentiable env / `DiffEnvironment`).
- **P4** — tracking, sword/shield, two-humanoid combat.

Plan of record: `../research/notes/investigations/2026-07-04-sim1-training-infra-plan.md`.

## Layout
```
sim1/
  config.py            dataclass configs (+ dotted overrides)
  envs/                VecEnv contract, mock + engine backends, obs norm, TaskEnv adapter
  tasks/               Task interface + reach (mock) + stand (humanoid); walk/combat later
  models/              Gaussian actor-critic MLP
  algos/               ppo; shac later
  utils/               seeding, run_dir, logging, checkpoint
  train.py / eval.py   entrypoints
csrc/                  nanobind binding → sim1.engine_py (engine::training)
tests/                 mock determinism, PPO-learns, reward-norm, checkpoint round-trip
external/engine/       the C++ engine (submodule)
```
