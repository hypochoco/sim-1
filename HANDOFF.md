# sim-1 Training Handoff

Operational handoff for continuing the humanoid training. Read this end-to-end before running or
changing anything. Companion docs: `readme.md` (quickstart), `csrc/README.md` (the binding), and
the research notes under `../research/notes/` (method rationale + the sim-1 plan of record
`../research/notes/investigations/2026-07-04-sim1-training-infra-plan.md`).

---

## 1. What this is, in one paragraph
`sim-1` is the **Python training layer** over a **C++ differentiable humanoid physics engine**
(`external/engine`, a git submodule). The engine is *mechanism only* — it simulates the humanoid and
exposes a batched `VecEnv` (reset/step + raw state); it has **no reward/observation/task/RL logic**.
Everything task- and learning-related lives here in `sim1/`: observation composition, reward,
termination, the RL algorithm (PPO today), and the training infrastructure. The engine is reached
through a **nanobind binding** (`csrc/ → sim1.engine_py`); a **mock env** with the identical contract
lets the whole stack run/test without the engine.

## 2. Current state (works today, verified)
- **Binding builds headless** (`pip install -e .`) via the graphics-free `engine::training` aggregate
  (`ENGINE_TRAINING_ONLY=ON`) — no Metal/Vulkan/GLFW. Imports as `from sim1 import engine_py`.
- **PPO trains the real AMP humanoid on the `stand` task** and *learns*: in a 400k-step smoke run
  (64 envs, reduced backend, PD-target, substeps=48, Mac M3 CPU, ~9k steps/s) episode length rose
  **50 → 66** steps, return **134 → 182**, value loss **3.29 → 0.056**, no NaNs.
- **Mock `reach` task** trains too (fast, engine-free) — used by the test suite.
- `pytest -q` → **5 passed** (mock determinism, PPO-learns, reward-norm, checkpoint round-trip).

## 3. Setup & run
```bash
cd sim-1
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, torch, tensorboard, tqdm, pytest
pip install -e .                          # builds sim1.engine_py (headless engine::training)

# HUMANOID stand (the real training target)
python -m sim1.train --name stand_amp \
  -o env.kind=engine -o env.model=amp -o env.backend=reduced \
  -o env.num_envs=64 -o env.episode_len=300 -o env.substeps=48 \
  -o env.action_mode=pd_target -o task.name=stand -o ppo.total_steps=2000000

# HUMANOID getup (stand + recover; warm-started from a stand policy, position-keeping ON to curb drift)
python -m sim1.train --name getup_amp \
  --init-from runs/<stand_run>/checkpoints/best.pt \
  -o env.kind=engine -o env.model=amp -o env.backend=reduced \
  -o env.num_envs=64 -o env.episode_len=1000 -o env.substeps=48 \
  -o env.action_mode=pd_target -o task.name=getup -o task.rotation=quat \
  -o task.height_weight=2.0 -o task.position_weight=0.5 -o ppo.total_steps=8000000

python -m sim1.train --resume runs/<run_id>     # resume (fully restores model/opt/rms/rng/step)
python -m sim1.eval  --run runs/<run_id>        # deterministic (mean-action) rollout
tensorboard --logdir runs                       # live curves
python -m pytest -q                             # tests (mock only; no binding needed)

# MOCK reach (engine-free; sanity/dev/CI)
python -m sim1.train --name reach -o ppo.total_steps=300000 -o env.num_envs=128
```
Runs land in `runs/<YYYYMMDD-HHMMSS>_<name>_<gitsha>/` with `config.json` (frozen), `meta.json`
(seed/sha/host), `metrics.jsonl`, `tb/`, `checkpoints/{best,final,step_*}.pt`.

### Linux build notes (verified on the training box)
Verified end-to-end on Linux: binding builds, `obs (4, 84)`, `pytest 5 passed`, and a 150k-step stand
run trained at ~4,400 steps/s (64 CPU envs) with clean metrics. Two specifics vs the Mac:

- **glm is a REQUIRED dependency** (not optional-graphics). The binding's CMake does
  `find_package(glm REQUIRED)` *outside* the `ENGINE_TRAINING_ONLY` guard: glm is the engine's math
  library, used everywhere; only GLFW/Metal/Vulkan are graphics-only and guarded out. If glm isn't
  system-discoverable, point CMake at a prefix (headers + `glmConfig.cmake`):
  `export CMAKE_PREFIX_PATH=/path/to/glm-prefix`.
- **Position-independent code** — the engine's static libs must be `-fPIC` to link into the `.so`
  (Mach-O is PIC by default, so this never surfaced on the Mac). **Handled** by
  `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` in sim-1's top-level `CMakeLists.txt` — no per-build flag
  needed.

## 4. Reading the metrics
- **`charts/ep_len_mean`** — the clearest `stand` progress signal: it rises as the humanoid falls
  less (episodes hit the time limit instead of terminating on a fall). Max = `env.episode_len`.
- **`charts/ep_return_mean`** — total episode reward (alive + upright + height − ctrl). Rises with learning.
- **`losses/value_loss`** — should fall (value fits returns; reward normalization keeps it O(1)).
- **`charts/reward_mean`** — mean per-step raw reward.
- Iteration 0 logs `ep_return_mean`/`ep_len_mean` as **NaN** (no episode finished yet) — expected, benign.

## 4b. Verifying the policy actually stands (not just high reward)
Run the trained checkpoint and get **physical** diagnostics + a thresholded verdict (headless, no
rendering needed):
```bash
python -m sim1.eval --run runs/<run_id> --episodes 40
# eval [stand, amp]: 40 episodes
#   mean return         ...
#   mean episode length XXX / 300
#   survival rate       YY.Y%   (reached the time limit without falling)
#   mean uprightness    0.9xx   (1.0 = vertical)
#   mean root height    x.xxx m (standing ≈ 1.022 m)
#   VERDICT: STANDING ✓ / NOT STANDING ✗   (survival ≥ 80% and uprightness ≥ 0.8)
```
`eval.evaluate(...)` returns these as a dict for programmatic checks. This is the way to confirm
"is it standing": **survival rate** (did it last the whole episode without falling), **uprightness**
(torso vertical), and **root height** near the standing height — a high `mean_return` alone is not
proof (reward can be gamed). Tune the `_SURVIVAL_OK`/`_UPRIGHT_OK` thresholds in `sim1/eval.py`.

**Visual (3D) playback is NOT available in the headless training build** (no renderer — that's the
whole point of `ENGINE_TRAINING_ONLY`). To *watch* a trained policy: `python -m sim1.export_policy
--run runs/<id>` writes a portable `policy.txt` (deterministic policy + obs-normalizer + sim knobs,
verified faithful to 6.4e-06). A **policy-driven visual runner** is built in the engine submodule:
`external/engine/tst/physics/visual/amp_policy.cpp` (+ header `policy_net.h`), which loads `policy.txt`,
rebuilds the training `Environment` (reduced backend), drives it with the policy each control step,
and renders via the ECS (`world->pose` + `physics_ecs::syncSystem`). Controls: **arrows steer the
walk command** for goal-conditioned policies (or tilt gravity to perturb balance for stand/getup),
Space shoves, R resets, P pauses. `export_policy` emits **SIM1_POLICY_V2** with `command_type`/
`command_dim` so the runner feeds the right goal channels (root-local target); V1 files still load
(command_dim 0). Build/run (macOS):
```bash
git -C external/engine submodule update --init external/{glfw,glm,metal-cpp,stb,tinyobjloader}
cmake -S external/engine -B external/engine/build && cmake --build external/engine/build --target visuals
ENGINE_POLICY=runs/<id>/policy.txt ./external/engine/build/tst/visuals amp_policy
```
Status: written against the confirmed engine APIs but **not yet compiled here** (graphics submodules
weren't checked out). Contract + design: `../research/notes/investigations/2026-07-04-policy-visualization-loop.md`.
Until it's built, verify **numerically** with `sim1.eval` and read `metrics.jsonl`/TensorBoard.

## 5. Config surface (dotted overrides: `-o section.field=value`)
Defaults are dataclasses in `sim1/config.py` (the Hydra-style framework is deliberately deferred).
- **`env`** (the sim-tuning knobs): `kind` (`mock`|`engine`), `model` (`humanoid` 21-DOF | `amp`
  28-DOF), `backend` (`reduced`|`realtime`), `num_envs`, `episode_len`, `substeps`, `control_dt`,
  `action_mode` (`torque`|`pd_target`), `kp`, `kd`, `max_torque`, `ground_friction`, `threads`
  (0 = all cores). Mock-only: `ndof`, `dt`, `damping`, `action_scale`, `target_scale`.
- **`task`**: `name` (`reach`|`stand`|`getup`|`walk`). stand/getup/walk share the canonical
  proprioception obs + terms: `upright_weight`, `height_weight`, `alive_bonus`, `action_weight`,
  `fall_height_frac`, `upright_fall`, `pd_action_scale`. `getup` = stand reward with **no fall
  termination** (recover after falling; use long `env.episode_len` ~1000). `walk` = goal-conditioned
  locomotion (`command_weight`, `target_speed_min/max`) — obs = proprio + 2 command channels (target
  planar velocity); the trained policy is user-steerable via `task.set_goal(...)`. Warm-start any of
  these from a prior policy with `--init-from <ckpt>` (shape-tolerant: shared proprio trunk transfers;
  mismatched I/O layers re-init).
- **`task.position_weight`** / **`task.position_scale`** (stand/getup): reward for staying near the
  start `(x,z)` — `exp(-scale·‖xz−xz₀‖²)`. Default `0` (off, so it doesn't change the banked stand's
  behavior); set `position_weight≈0.5` to curb the horizontal drift seen in the first stand. Not for
  `walk` (it's meant to move). **getup caveat:** without an engine RSI/knockdown hook, `getup` only
  learns to recover from *self-induced* falls (long episode + no fall-termination); deliberate
  knockdowns / reset-from-fallen need a small engine addition (expose `Environment::setResetHook` /
  an initial-state randomizer / an impulse call) — future work.
- **`task.rotation`** (`quat` | `sixd`): root-orientation obs encoding. `sixd` is the continuous 6D
  representation (Zhou et al. — first two rotation-matrix columns; nets learn rotations better than
  from quats). It **changes obs_dim** (+2), so a `sixd` policy is a *fresh lineage* — it does NOT
  transfer from `quat`-trained weights (and vice versa). Reward/termination are unaffected (they read
  the raw quat). Caveats: (a) only the *root* orientation is 6D today — per-body 6D
  (SuperTrack-style) is the tracking-layer follow-up. The visual runner **does** handle `sixd`: the
  policy file (V3) carries a `rotation` field and `amp_policy.cpp` mirrors the quat→6D conversion
  (`tst::quatTo6D`, matching `proprio.quat_to_6d`) when composing the obs.
- **`ppo`**: `total_steps`, `rollout_len`, `lr`, `anneal_lr`, `gamma`, `gae_lambda`, `clip_coef`,
  `update_epochs`, `num_minibatches`, `ent_coef`, `vf_coef`, `max_grad_norm`, `clip_vloss`,
  `norm_adv`, `norm_obs`, `norm_reward`, `reward_clip`, `hidden_sizes`.
- **`run`**: `name`, `seed`, `device` (`cpu`|`cuda`|`mps` — auto-falls back), `runs_root`,
  `checkpoint_interval`, `keep_last`.

## 6. Gotchas & hard-won lessons (READ THIS)
- **Use PD-target control for the humanoid, not torque.** Full-scale random torques diverge the sim
  → NaN cascade. `action_mode=pd_target` (action = desired joint pose, servo-tracked) is stable and
  is the DeepMimic/ASE standard. `StandTask.action_scale` is mode-aware: `max_torque` for torque,
  `task.pd_action_scale` (≈radians) for PD.
- **`substeps=48` for contact stability.** The engine's differentiable/reduced contact needs enough
  substeps; fewer can blow up (see engine notes `2026-07-04-diff-semiimplicit-testing.md`,
  `2026-07-04-differentiable-contact-geometry.md`). Aggressive actuation into contact can still
  diverge over long horizons.
- **A divergence guard exists** (`TaskEnv.step`): any env with non-finite obs is treated as a
  fall (reset + zeroed reward) so one blown-up world can't NaN-poison the batch. If you see lots of
  guard-triggered resets, the sim is unstable — lower `pd_action_scale`, raise `substeps`, or lower `kp`.
- **Reward normalization is on by default** (`ppo.norm_reward`) — divides reward by the running std
  of the discounted return so the value scale stays O(1) across reward magnitudes. Keep it on.
- **Explained variance ~0 on the mock `reach`** is expected (action-noise-dominated returns), NOT a
  bug; on the humanoid with shaped rewards `value_loss` fits fine.
- **The sim is CPU-only** (no CUDA in the engine). The GPU only accelerates the tiny policy net.
  Throughput scales with **CPU cores / `env.threads`**, not the GPU. Benchmark env-steps/s per box.
- **Import is package-private:** `from sim1 import engine_py` (the extension installs into `sim1/`).
- **Obs layout** from the binding: `[pos3 | quat_wxyz4 | linvel3 | angvel3 | q[ndof] | qd[ndof] |
  contacts[nbody]]`; `sim1/envs/engine_vecenv.py` slices the named contract fields from it.

## 6b. Bugs found porting to Linux (FIXED)
Two latent bugs surfaced during a long Linux training run; both are now **fixed in code** (with
regression tests in `tests/test_run_infra_fixes.py`). Documented here because a run *already in
flight on another machine* pre-dates the fix — the recovery recipe below applies there.

### Bug 1 — checkpoint files sorted lexicographically (data-loss risk on resume)
- **Where:** `sim1/utils/checkpoint.py::prune_checkpoints` and `sim1/train.py::_newest_checkpoint`.
- **Symptom:** once `global_step` crosses a digit-count boundary (≥ 1,000,000, then ≥ 10,000,000),
  `prune_checkpoints` **deleted the newest** `step_*.pt` and kept stale smaller-step ones, and
  `--resume` reloaded a **stale early checkpoint**.
- **Root cause:** `sorted(glob("step_*.pt"))` is a lexicographic **string** sort, so
  `step_1024000.pt` < `step_901120.pt` (`'1' < '9'`). All 7-digit names sort before 6-digit names;
  `keep_last` then retained the wrong three and `[-1]` ("newest") was also wrong.
- **Observed impact:** after an auto-stop at ~6.7M steps, only `step_901120/942080/983040.pt`
  survived; stock `--resume` would have restarted from **983k** (losing ~5.7M steps). We recovered
  because **`best.pt` uses a fixed filename** and is immune — it held `global_step=6,303,744`.
- **Fix:** sort numerically by the embedded step in both places (`checkpoint.step_number()`).
- **Recovery recipe (for the pre-fix in-flight run):** if a naive resume looks like it restarted too
  early, resume from the best snapshot: `cp runs/<id>/checkpoints/best.pt runs/<id>/checkpoints/final.pt`
  then `python -m sim1.train --resume runs/<id>` (`best.pt` carries full model/opt/rms/rng/step).

### Bug 2 — `runs/latest` silently stops updating after a transfer
- **Where:** `sim1/utils/run_dir.py::_update_latest`.
- **Symptom:** `runs/latest` froze on an old run, so `--resume runs/latest` / `--run runs/latest` /
  TensorBoard silently targeted the wrong run.
- **Root cause:** a zip/copy that **dereferences the symlink** turns `latest` into a real directory;
  `link.unlink()` then raised `IsADirectoryError` (an `OSError`) which the `except OSError: pass`
  swallowed — so it was never recreated (and wasted disk as a copy).
- **Fix:** detect a real directory and `shutil.rmtree` it before re-creating the symlink.
- **Workaround (pre-fix checkouts):** `rm -rf runs/latest && ln -s <newest_run_dir> runs/latest`.

## 7. Architecture / where to change things
```
engine (C++, submodule)          sim-1 (this repo, Python)
  physics + physics_env            csrc/engine_py.cpp      nanobind → sim1.engine_py (VecEnv/SimConfig/rigs)
  VecEnv: reset/step + raw state   sim1/envs/vecenv.py     the VecEnv Protocol (contract)
  makeHumanoid / makeAMPHumanoid   sim1/envs/engine_vecenv.py  EngineVecEnv + make_vecenv(cfg) factory
  SimConfig (tuning knobs)         sim1/envs/mock_vecenv.py    contract-compatible mock
  DiffEnvironment (NOT yet bound)  sim1/envs/task_env.py   reset/step adapter + truncation + NaN guard
                                   sim1/tasks/{base,reach,stand}.py   obs/reward/termination
                                   sim1/models/mlp.py      Gaussian actor-critic
                                   sim1/algos/ppo.py       PPO (GAE + truncation bootstrap, resumable)
                                   sim1/utils/             seeding, run_dir, logging, checkpoint
                                   sim1/train.py           build_env → build_task → run loop
                                   sim1/eval.py            deterministic rollout
```
- **Add a task:** prefer the composable system — build a `CompositeTask` (`sim1/tasks/composite.py`)
  from a `Command` (`sim1/tasks/command.py`, the goal/user-control seam), `RewardTerm`s
  (`sim1/tasks/rewards.py`), and a termination fn, reusing the canonical `proprio_obs`
  (`sim1/tasks/proprio.py`) so weights stay transferable; `walk.py` is the worked example. Register it
  in `sim1/train.py:build_task`. (The older `reach`/`stand`/`getup` are hand-rolled but proprio-
  identical.) Mocap plugs in as a reference `ObsComponent` + an imitation `RewardTerm` — see
  `sim1/motion/` and the architecture note.
- **Tune the sim:** change `env.*` overrides (substeps, kp/kd, action_mode, friction) — no code edits.

## 8. Next steps (prioritized)
1. **Benchmark env-steps/s on the target Linux CPU** at `env.num_envs` ∈ {64, 1024, 4096} and set
   `env.threads`. This sets wall-clock (sim is CPU-bound), more than the GPU does.
2. **Improve `stand`**: longer `episode_len` (300 → ~1000), tune reward weights toward long holds;
   add **reference-state-init** (reset from perturbed/fallen poses) via the engine reset hook to learn
   recovery/get-up. Watch `ep_len_mean` approaching `episode_len`.
3. **P2 `walk`**: new task — target speed/heading reward + drift termination + curriculum.
4. **P3 SHAC** (the differentiable-sim payoff): **requires extending the binding** — the engine's
   `DiffEnvironment` (per-step Jacobian / rollout gradient, Phase F complete engine-side) is **not yet
   exposed in `csrc/engine_py.cpp`**. Bind it, wrap in a `torch.autograd.Function`, add `sim1/algos/shac.py`.
5. **P4**: tracking (mocap + retarget), sword/shield, two-humanoid combat (needs engine-side
   multi-articulation + a multi-agent env; keep the Task/VecEnv API agent-count-agnostic).
6. Optional infra: run/reward normalization is done; a config framework (Hydra) and W&B are deferred.

## 9. Repo / git state at handoff
- Engine submodule pinned at **`0f69de0`** ("small fixes for training builds": headless build guards +
  AMP rig). Committed in sim-1 as **`8281c5e`** (submodule bump) — **1 commit ahead of origin/main**.
- **Uncommitted** in sim-1 (working, verified, awaiting the owner's review/commit):
  `csrc/`, `sim1/envs/engine_vecenv.py`, `sim1/tasks/stand.py`, and edits to
  `sim1/train.py`, `sim1/envs/task_env.py`, `sim1/config.py`, `sim1/tasks/{base,reach}.py`,
  `CMakeLists.txt`, `pyproject.toml`, `readme.md`. (`runs/` is git-ignored.)

## 10. Verification checklist (run these first to confirm a working setup)
```bash
. .venv/bin/activate
pip install -e .                                   # binding builds headless
python -c "from sim1 import engine_py; e=engine_py.VecEnv(4, \
  engine_py.EnvConfig(engine_py.make_amp_humanoid(), engine_py.SimConfig())); \
  e.reset(0); a=e.actions(); a[:]=0; e.step(); print('obs', e.observations().shape)"  # → obs (4, 84)
python -m pytest -q                                # → 5 passed
# short humanoid run; expect finite metrics and ep_len_mean trending up over ~a few hundred iters
python -m sim1.train --name check -o env.kind=engine -o env.model=amp -o env.backend=reduced \
  -o env.num_envs=64 -o env.episode_len=300 -o env.substeps=48 -o env.action_mode=pd_target \
  -o task.name=stand -o ppo.total_steps=400000
python -m sim1.eval --run runs/<check_run_id> --episodes 40   # standing diagnostics + verdict
```
