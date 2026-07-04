"""sim1 — the training layer over the differentiable humanoid physics engine.

The C++ engine is mechanism-only; this package owns the task/scene layer (observation
composition, reward, termination), the RL algorithms (PPO now, SHAC later), and the training
infrastructure (config, run dirs, logging, checkpoints). P0 runs against a mock VecEnv that
matches the engine's binding contract, so everything is testable before the binding lands.
"""

__version__ = "0.0.1"
