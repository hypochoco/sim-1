"""PPO trainer (CleanRL-style), specialized for our vectorized `TaskEnv`.

Correctness notes:
- Observation normalization (RunningMeanStd) is updated online during rollout collection and the
  normalized obs are what the policy/value see (consistent within and across the update epochs).
- GAE handles **time-limit truncation** properly: at an episode boundary the trace is cut
  (`1 - done`) and the bootstrap value is `V(terminal_obs)` for truncation / 0 for failure — carried
  in `next_values`, so a truncated episode still bootstraps instead of treating the reset obs as a
  continuation.
- All training state (model, optimizer, obs-rms, global_step, iteration, RNG) is captured by
  `state_dict()` for resumable checkpoints.
"""

from __future__ import annotations

import copy
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from sim1.config import PPOConfig
from sim1.envs.running_norm import RunningMeanStd
from sim1.envs.task_env import TaskEnv
from sim1.models.mlp import ActorCritic


class PPOTrainer:
    def __init__(self, env: TaskEnv, cfg: PPOConfig, device: str = "cpu", seed: int = 0):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.model = ActorCritic(self.obs_dim, self.act_dim, tuple(cfg.hidden_sizes)).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd((self.obs_dim,), self.device) if cfg.norm_obs else None
        self.ret_rms = RunningMeanStd((), self.device) if cfg.norm_reward else None

        self.batch_size = self.num_envs * cfg.rollout_len
        self.minibatch_size = max(1, self.batch_size // cfg.num_minibatches)
        self.num_iterations = max(1, cfg.total_steps // self.batch_size)

        self.global_step = 0
        self.iteration = 0

        # --- exploration state ---
        # (3) adaptive parameter-space noise: a frozen copy of the actor whose weights we perturb
        # each rollout; PPO trains/exports the CLEAN model (self.model), the perturbed copy only acts.
        self._pnoise_std = float(cfg.param_noise_init) if cfg.param_noise else 0.0
        if cfg.param_noise:
            self._pnoise_actor = copy.deepcopy(self.model.actor_mean).to(self.device)
            for p in self._pnoise_actor.parameters():
                p.requires_grad_(False)
        else:
            self._pnoise_actor = None

        self._recent_returns: deque[float] = deque(maxlen=100)
        self._recent_lens: deque[int] = deque(maxlen=100)
        self._ep_ret = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.num_envs, dtype=np.int64)
        self._ret_acc = np.zeros(self.num_envs, dtype=np.float64)  # discounted-return accumulator

        self._obs = self._process_obs(env.reset(), update=True)

    # --- obs normalization ---
    def _process_obs(self, obs_np, update: bool) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=self.device)
        if self.obs_rms is not None:
            if update:
                self.obs_rms.update(t)
            t = self.obs_rms.normalize(t)
        return t

    # --- reward normalization (divide by running std of the discounted return) ---
    def _norm_reward(self, raw_reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        if self.ret_rms is None:
            return raw_reward.astype(np.float32)
        self._ret_acc = self._ret_acc * self.cfg.gamma + raw_reward
        self.ret_rms.update(torch.as_tensor(self._ret_acc, dtype=torch.float32, device=self.device))
        std = float(torch.sqrt(self.ret_rms.var + 1e-8).item())
        r = raw_reward / std
        if self.cfg.reward_clip > 0:
            r = np.clip(r, -self.cfg.reward_clip, self.cfg.reward_clip)
        self._ret_acc[np.asarray(done, dtype=bool)] = 0.0  # reset accumulator at episode end
        return r.astype(np.float32)

    # --- exploration helpers ---
    def _current_ent_coef(self) -> float:
        """Cyclical entropy: cosine from `ent_cycle_max` at each cycle start down to `ent_coef`,
        then restart — periodically reviving exploration. Constant `ent_coef` when disabled."""
        c = self.cfg
        if c.ent_cycle_iters > 0 and c.ent_cycle_max > c.ent_coef:
            phase = (self.iteration % c.ent_cycle_iters) / c.ent_cycle_iters   # 0 → 1 over a cycle
            cos = 0.5 * (1.0 + math.cos(math.pi * phase))                       # 1 at start → 0 at end
            return c.ent_coef + (c.ent_cycle_max - c.ent_coef) * cos
        return c.ent_coef

    def _resample_param_noise(self) -> None:
        """Refill the perturbed actor with clean weights + fresh Gaussian noise (once per rollout)."""
        with torch.no_grad():
            for pc, pp in zip(self.model.actor_mean.parameters(), self._pnoise_actor.parameters()):
                pp.copy_(pc + torch.randn_like(pc) * self._pnoise_std)

    def _adapt_param_noise(self, obs_flat: torch.Tensor) -> float:
        """Adapt the noise std toward a target action-space distance (Plappert's rule)."""
        with torch.no_grad():
            idx = torch.randperm(obs_flat.shape[0], device=self.device)[:4096]
            o = obs_flat[idx]
            d = torch.sqrt(torch.mean((self.model.actor_mean(o) - self._pnoise_actor(o)) ** 2)).item()
        if d < self.cfg.param_noise_target:
            self._pnoise_std *= 1.01
        else:
            self._pnoise_std = max(1e-6, self._pnoise_std / 1.01)
        return d

    def _act_behavior(self, obs: torch.Tensor):
        """Sample the behavior action and return (action, behavior_logp, clean_logp, value). When
        param-noise is on, the perturbed actor is the behavior policy (its logp drives PPO's importance
        ratio), while `clean_logp` is the *unperturbed* policy's logp of the same action — used for the
        trust-region KL so the check reflects true policy drift, not the exploration perturbation."""
        if self._pnoise_actor is None:
            a, lp, _, v = self.model.get_action_and_value(obs)
            return a, lp, lp, v
        std = torch.exp(self.model.actor_logstd)
        mean = self._pnoise_actor(obs)
        dist = Normal(mean, std.expand_as(mean))
        a = dist.sample()
        lp = dist.log_prob(a).sum(-1)
        cmean = self.model.actor_mean(obs)
        clp = Normal(cmean, std.expand_as(cmean)).log_prob(a).sum(-1)
        return a, lp, clp, self.model.get_value(obs)

    # --- rollout collection + GAE ---
    def collect(self):
        cfg, T, N, dev = self.cfg, self.cfg.rollout_len, self.num_envs, self.device
        obs_b = torch.zeros((T, N, self.obs_dim), device=dev)
        act_b = torch.zeros((T, N, self.act_dim), device=dev)
        logp_b = torch.zeros((T, N), device=dev)
        logp_clean_b = torch.zeros((T, N), device=dev)
        val_b = torch.zeros((T, N), device=dev)
        rew_b = torch.zeros((T, N), device=dev)
        raw_rew_b = torch.zeros((T, N), device=dev)
        done_b = torch.zeros((T, N), device=dev)
        nextval_b = torch.zeros((T, N), device=dev)

        if self._pnoise_actor is not None:
            self._resample_param_noise()

        for t in range(T):
            obs_t = self._obs
            with torch.no_grad():
                action, logp, clean_logp, value = self._act_behavior(obs_t)
            obs_b[t], act_b[t], logp_b[t], val_b[t] = obs_t, action, logp, value
            logp_clean_b[t] = clean_logp

            act_np = action.detach().cpu().numpy().astype(np.float32)
            next_obs_np, reward, done, info = self.env.step(act_np)

            with torch.no_grad():
                next_obs_t = self._process_obs(next_obs_np, update=True)
                v_next = self.model.get_value(next_obs_t)
                v_term = self.model.get_value(self._process_obs(info["terminal_obs"], update=False))

            done_bool = torch.as_tensor(done, dtype=torch.bool, device=dev)
            trunc_bool = torch.as_tensor(info["truncated"], dtype=torch.bool, device=dev)
            boot = torch.where(trunc_bool, v_term, torch.zeros_like(v_term))
            nextval_b[t] = torch.where(done_bool, boot, v_next)
            rew_b[t] = torch.as_tensor(self._norm_reward(reward, done), dtype=torch.float32, device=dev)
            raw_rew_b[t] = torch.as_tensor(reward, dtype=torch.float32, device=dev)
            done_b[t] = done_bool.float()

            self._obs = next_obs_t
            self.global_step += N

            self._ep_ret += reward  # raw reward for reported episode returns
            self._ep_len += 1
            for i in np.nonzero(done)[0]:
                self._recent_returns.append(float(self._ep_ret[i]))
                self._recent_lens.append(int(self._ep_len[i]))
                self._ep_ret[i] = 0.0
                self._ep_len[i] = 0

        # GAE
        adv_b = torch.zeros((T, N), device=dev)
        lastgae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nonterminal = 1.0 - done_b[t]
            delta = rew_b[t] + cfg.gamma * nextval_b[t] - val_b[t]
            lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * lastgae
            adv_b[t] = lastgae
        ret_b = adv_b + val_b

        batch = {
            "obs": obs_b.reshape(-1, self.obs_dim),
            "actions": act_b.reshape(-1, self.act_dim),
            "logprobs": logp_b.reshape(-1),
            "logprobs_clean": logp_clean_b.reshape(-1),
            "advantages": adv_b.reshape(-1),
            "returns": ret_b.reshape(-1),
            "values": val_b.reshape(-1),
        }
        roll_metrics = {
            "charts/reward_mean": float(raw_rew_b.mean().item()),
            "charts/ep_return_mean": self.recent_return_mean(),
            "charts/ep_len_mean": (float(np.mean(self._recent_lens)) if self._recent_lens else float("nan")),
        }
        if self._pnoise_actor is not None:
            roll_metrics["charts/param_noise_dist"] = self._adapt_param_noise(batch["obs"])
            roll_metrics["charts/param_noise_std"] = self._pnoise_std
        return batch, roll_metrics

    # --- PPO update ---
    def update(self, batch) -> dict:
        cfg = self.cfg
        b_obs, b_act = batch["obs"], batch["actions"]
        b_logp, b_adv = batch["logprobs"], batch["advantages"]
        b_logp_clean = batch["logprobs_clean"]
        b_ret, b_val = batch["returns"], batch["values"]
        bs = b_obs.shape[0]
        inds = np.arange(bs)
        clipfracs: list[float] = []
        pg_loss = v_loss = entropy_loss = approx_kl = torch.tensor(0.0)
        grad_norm = 0.0
        ent_now = self._current_ent_coef()
        epochs_done = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, bs, self.minibatch_size):
                mb = inds[start : start + self.minibatch_size]
                _, newlogp, entropy, newval = self.model.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]              # vs behavior policy → PPO importance ratio
                ratio = logratio.exp()
                with torch.no_grad():
                    # trust-region drift measured vs the CLEAN pre-update policy (== behavior when
                    # param-noise is off), so param-noise perturbation doesn't inflate the KL / early-stop.
                    lr_clean = newlogp - b_logp_clean[mb]
                    approx_kl = ((lr_clean.exp() - 1) - lr_clean).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                newval = newval.view(-1)
                if cfg.clip_vloss:
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(newval - b_val[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_now * entropy_loss + cfg.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm))
                self.opt.step()

            epochs_done += 1
            if cfg.target_kl > 0 and approx_kl.item() > cfg.target_kl:
                break  # trust region: policy has moved far enough this update

        y_pred, y_true = b_val.detach().cpu().numpy(), b_ret.detach().cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        self.iteration += 1
        return {
            "losses/policy_loss": float(pg_loss.item()),
            "losses/value_loss": float(v_loss.item()),
            "losses/entropy": float(entropy_loss.item()),
            "losses/entropy_coef": ent_now,
            "losses/epochs_done": epochs_done,
            "losses/approx_kl": float(approx_kl.item()),
            "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "losses/grad_norm": grad_norm,
            "losses/explained_variance": explained_var,
        }

    def recent_return_mean(self) -> float:
        return float(np.mean(self._recent_returns)) if self._recent_returns else float("nan")

    # --- checkpointing ---
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms is not None else None,
            "ret_rms": self.ret_rms.state_dict() if self.ret_rms is not None else None,
            "global_step": self.global_step,
            "iteration": self.iteration,
            "param_noise_std": self._pnoise_std,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }

    def load_state_dict(self, s: dict) -> None:
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["optimizer"])
        if self.obs_rms is not None and s.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(s["obs_rms"])
        if self.ret_rms is not None and s.get("ret_rms") is not None:
            self.ret_rms.load_state_dict(s["ret_rms"])
        self.global_step = int(s["global_step"])
        self.iteration = int(s["iteration"])
        if s.get("param_noise_std") is not None:
            self._pnoise_std = float(s["param_noise_std"])
        if s.get("torch_rng") is not None:
            torch.set_rng_state(s["torch_rng"])
        if s.get("numpy_rng") is not None:
            np.random.set_state(s["numpy_rng"])
