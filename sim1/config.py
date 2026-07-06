"""Config schema — plain dataclasses with in-code defaults.

The full config *framework* (Hydra/YAML compose + sweeps) is intentionally deferred. For now a
run is described by these dataclasses; they are serialized into the run dir (`config.json`) for
reproducibility, and a light dotted-override mechanism (`ppo.lr=1e-3`) supports quick iteration.
Fields for the real engine (substeps/kp/kd/...) already live in `EnvConfig` so configs stay
forward-compatible when the binding lands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    kind: str = "mock"              # "mock" (P0) | "engine" (P1, when the binding lands)
    num_envs: int = 64
    episode_len: int = 64           # control steps per episode (time-limit truncation)

    # --- mock-only dynamics params ---
    ndof: int = 2
    dt: float = 0.05
    damping: float = 0.1
    action_scale: float = 1.0
    target_scale: float = 1.0

    # --- engine params (unused in P0; kept for forward-compat / sim tuning in P1) ---
    model: str = "humanoid"         # "humanoid" (21 DOF) | "amp" (28 DOF) — the engine rig
    backend: str = "reduced"        # "reduced" (Featherstone) | "realtime" (maximal)
    threads: int = 0                # VecEnv worker threads (0 = hardware concurrency)
    substeps: int = 8
    control_dt: float = 1.0 / 60.0
    kp: float = 150.0
    kd: float = 15.0
    max_torque: float = 150.0
    action_mode: str = "torque"     # "torque" | "pd_target"
    ground_friction: float = 0.9


@dataclass
class TaskConfig:
    name: str = "reach"             # "reach" (mock) | "stand" | "getup" | "walk" (humanoid)
    # reach (mock)
    pos_weight: float = 1.0
    vel_weight: float = 0.0
    action_weight: float = 0.01
    # stand / getup / walk (humanoid) — shared proprio + terms
    upright_weight: float = 1.0     # reward for the torso staying vertical
    height_weight: float = 1.0      # reward for the root staying near its standing height
    alive_bonus: float = 1.0        # per-step reward for not having fallen
    fall_height_frac: float = 0.5   # terminate when root height < frac * standing height
    upright_fall: float = 0.3       # terminate when uprightness (world up · torso up) < this
    pd_action_scale: float = 1.0    # PD-target mode: policy output → target (radians / rotvec)
    position_weight: float = 0.0    # reward for staying near the start (x,z) — stand/getup; 0 = off
    position_scale: float = 1.0     # sharpness of the position-keeping Gaussian
    rotation: str = "quat"          # root-orientation obs encoding: "quat" (4) | "sixd" (continuous 6D)
    frame: str = "world"            # proprio frame: "world" (raw) | "local" (heading-relative, yaw-invariant)
    body_obs: bool = False          # append the SuperTrack per-body 6D block (for the reusable tracking base)
    # walk (goal-conditioned locomotion)
    command_weight: float = 1.0     # reward weight on tracking the commanded velocity
    target_speed_min: float = 0.5   # sampled target-speed range (m/s) for the walk command
    target_speed_max: float = 1.5
    # track (DeepMimic-style motion imitation)
    motion_clip: str = "amp_humanoid_walk"   # ASE clip name (retargeted to our rig)
    track_reward: str = "v3"        # reward variant from sim1/tasks/track_rewards.py ("v1" | "v2" | "v3" | …)
    rsi: bool = True                # reference-state-init: start episodes from random reference frames
    track_pose_weight: float = 0.5  # imitation reward weights (per-body orientation / vel / ee / root)
    track_vel_weight: float = 0.05
    track_ee_weight: float = 0.2
    track_root_weight: float = 0.15
    track_alive_bonus: float = 0.1   # survival bonus (kept small so tracking accuracy drives learning)
    track_action_weight: float = 0.002  # small control penalty (discourage jitter/high torque)
    track_term_pos_err: float = 0.5  # RMS root-relative body position error (m) that ends an episode


@dataclass
class PPOConfig:
    total_steps: int = 300_000
    rollout_len: int = 32
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    num_minibatches: int = 4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_vloss: bool = True
    norm_adv: bool = True
    norm_obs: bool = True
    norm_reward: bool = True        # normalize rewards by the running std of the discounted return
    reward_clip: float = 10.0       # clip normalized rewards to +/- this (0 disables)
    # --- exploration (keep exploration alive / periodically revive it to escape local minima) ---
    # (1a) cyclical entropy: a cosine "restart" that periodically bumps the entropy bonus back up.
    ent_cycle_iters: int = 0        # >0: length (in PPO updates) of one entropy cycle; 0 = constant ent_coef
    ent_cycle_max: float = 0.0      # peak entropy coefficient at the start of each cycle (decays to ent_coef)
    # (1b) trust region: early-stop the update epochs once the policy has moved too far.
    target_kl: float = 0.0          # >0: stop the epoch loop when approx_kl exceeds this
    # (1c) don't let the annealed lr decay all the way to zero (preserves late-training exploration).
    lr_final_frac: float = 0.0      # anneal lr down to this fraction of lr (0.0 = to zero; old behavior)
    # (3) adaptive parameter-space noise (Plappert et al.): structured, weight-space exploration.
    param_noise: bool = False       # perturb the actor's weights during rollout collection
    param_noise_init: float = 0.01  # initial per-weight noise std
    param_noise_target: float = 0.1 # target action-space RMS distance between clean & perturbed policy
    hidden_sizes: tuple[int, ...] = (128, 128)


@dataclass
class RunConfig:
    name: str = "ppo_mock"
    seed: int = 0
    device: str = "cpu"             # "cpu" | "cuda" | "mps"
    runs_root: str = "runs"
    checkpoint_interval: int = 20   # iterations between checkpoints
    keep_last: int = 3              # rolling checkpoints to retain (plus best + final)


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TrainConfig":
        return TrainConfig(
            env=EnvConfig(**d.get("env", {})),
            task=TaskConfig(**d.get("task", {})),
            ppo=PPOConfig(**d.get("ppo", {})),
            run=RunConfig(**d.get("run", {})),
        )

    def apply_overrides(self, overrides: list[str]) -> None:
        """Apply dotted `section.field=value` overrides, coercing to the field's type."""
        for ov in overrides:
            key, sep, val = ov.partition("=")
            if not sep:
                raise ValueError(f"bad override (expected section.field=value): {ov!r}")
            section_name, _, field_name = key.partition(".")
            if not field_name:
                raise ValueError(f"bad override key (expected section.field): {key!r}")
            section = getattr(self, section_name, None)
            if section is None or not hasattr(section, field_name):
                raise ValueError(f"unknown config field: {key!r}")
            cur = getattr(section, field_name)
            setattr(section, field_name, _coerce(cur, val))


def _coerce(cur: Any, val: str) -> Any:
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    return val
