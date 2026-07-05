# csrc — the `engine_py` binding

Nanobind module exposing the engine's **ECS-free training surface** (`core + physics + physics_env`)
to Python: the batched `VecEnv`, the centralized `SimConfig`, and the rig factories
(`make_humanoid` / `make_amp_humanoid`). Batch buffers are handed over as **zero-copy NumPy views**.
The Python side (`sim1/envs/engine_vecenv.py`) adapts this to the `VecEnv` contract and maps the
Python `EnvConfig` onto the engine `SimConfig`.

It links the engine's graphics-free **`engine::training`** aggregate and builds with
`ENGINE_TRAINING_ONLY=ON` (forced by the top-level `CMakeLists.txt`), so **no graphics / GLFW /
Metal / Vulkan** is configured — it links clean on a headless Linux box and needs none of the
engine's **graphics** submodules initialized. The one external it *does* require is **glm** (the
engine's header/math library, used outside the graphics guard via `find_package(glm REQUIRED)`).

## Build

```bash
. .venv/bin/activate
pip install -e .            # scikit-build-core → CMake → builds engine_py into the sim1 package
python -c "from sim1 import engine_py; \
  e=engine_py.VecEnv(4, engine_py.EnvConfig(engine_py.make_amp_humanoid(), engine_py.SimConfig())); \
  print('act_dim', e.act_dim, 'obs_dim', e.obs_dim, 'ndof', e.ndof, 'nbody', e.nbody); \
  a=e.actions(); a[:]=0.0; e.step(); print('obs', e.observations().shape)"
```

The extension installs **package-private** as `sim1.engine_py` (import it as
`from sim1 import engine_py`, which is what `engine_vecenv.py` does). Requires the engine submodule at
a commit with the config system + `makeAMPHumanoid` (bumped to `0f69de0`), plus CMake ≥3.15 and a
C++23 compiler; `scikit-build-core` + `nanobind` are pulled automatically by the build.

**On Linux** (verified on the training box) the binding needs a discoverable **glm**
(`CMAKE_PREFIX_PATH` → a prefix with `glmConfig.cmake`, if not system-installed). Position-independent
code (`-fPIC`, needed to link the static engine libs into the `.so`; Mac is PIC by default) is already
handled by `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` in the top-level `CMakeLists.txt`. See
`HANDOFF.md` §3 "Linux build notes".

Flip the trainer to the real env with `-o env.kind=engine -o env.model=amp`.

## API (matches `sim1/envs/vecenv.py`)
- `VecEnv(num_envs, EnvConfig, threads=0)` → `reset(seed)` / `reset_masked(mask, seed)` / `step()`;
  `actions()` (writable `(N, act_dim)`), `observations()` (read-only `(N, obs_dim)`); `num_envs`,
  `act_dim`, `obs_dim`, `ndof`, `nbody`. (`threads=0` → hardware concurrency.)
- `SimConfig` — `substeps`, `control_dt`, `max_torque`, `kp`, `kd`, `ground_friction`, `backend`,
  `action_mode`, `gravity`, `linear/angular_damping`, `velocity_iterations`, `ground_plane`;
  `dump()` / `config_hash()` for run history.
- `make_humanoid()` (21 DOF) / `make_amp_humanoid()` (28 DOF); `EnvConfig(articulation, sim)`.

Obs layout (the Python adapter slices this): `[pos3 | quat_wxyz4 | linvel3 | angvel3 | q[ndof] |
qd[ndof] | contacts[nbody]]`.
