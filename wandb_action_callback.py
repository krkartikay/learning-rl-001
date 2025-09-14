import numpy as np
from typing import Any, Dict, List, Optional

import wandb
from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggingCallback(BaseCallback):
    """
    SB3 callback to record and log actions to Weights & Biases.

    - Buffers actions per environment and logs them at episode end.
    - Also logs acceptance rate (assuming Discrete(2): 1=accept, 0=reject).

    Notes:
    - Works for both vectorized and non-vectorized envs (SB3 wraps non-vec
      envs in a DummyVecEnv).
    - Logging every step can overwhelm wandb; this callback aggregates actions
      over an episode and logs once per episode.
    """

    def __init__(self, log_table: bool = True, log_hist: bool = True, name_prefix: str = "actions"):
        super().__init__()
        self.log_table = log_table
        self.log_hist = log_hist
        self.name_prefix = name_prefix
        self._actions_per_env: List[List[int]] = []
        self._episode_idx: List[int] = []

    def _init_callback(self) -> None:
        # Initialize buffers after model and env are set
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._actions_per_env = [[] for _ in range(n_envs)]
        self._episode_idx = [0 for _ in range(n_envs)]

    def _on_step(self) -> bool:
        # Extract actions and dones from learner locals
        actions = self.locals.get("action", None)
        dones = self.locals.get("dones", None)

        if actions is None:
            # Nothing to record this step
            return True

        # Normalize actions to 1D array of ints (one per env)
        if isinstance(actions, (list, tuple)):
            acts = np.array(actions)
        else:
            acts = np.asarray(actions)
        acts = np.squeeze(acts)
        if acts.ndim == 0:
            acts = np.expand_dims(acts, 0)

        # Ensure buffers sized for current number of envs
        n_envs = acts.shape[0]
        if len(self._actions_per_env) != n_envs:
            # Resize buffers defensively if vectorization has changed
            self._actions_per_env = [[] for _ in range(n_envs)]
            self._episode_idx = [0 for _ in range(n_envs)]

        # Record actions for each env
        for i in range(n_envs):
            try:
                a = int(acts[i])
            except Exception:
                # Fallback: if action is array-like, take first element
                a = int(np.array(acts[i]).reshape(-1)[0])
            self._actions_per_env[i].append(a)

        # On episode end, flush logs
        if dones is not None:
            dones_arr = np.squeeze(np.asarray(dones)).astype(bool)
            if dones_arr.ndim == 0:
                dones_arr = np.expand_dims(dones_arr, 0)

            for i, done in enumerate(dones_arr):
                if done:
                    seq = self._actions_per_env[i]
                    if len(seq) == 0:
                        # Nothing to log
                        self._episode_idx[i] += 1
                        continue

                    # Compute simple stats assuming Discrete(2) actions
                    accept_rate = float(np.mean(seq))
                    length = len(seq)
                    log_payload: Dict[str, Any] = {
                        f"{self.name_prefix}/accept_rate": accept_rate,
                        f"{self.name_prefix}/length": length,
                    }

                    if self.log_hist:
                        # Histogram of actions (orderless distribution)
                        try:
                            log_payload[f"{self.name_prefix}/histogram"] = wandb.Histogram(seq)
                        except Exception:
                            # If Histogram construction fails (e.g., tiny seq), skip
                            pass

                    if self.log_table:
                        # Ordered sequence as a table
                        try:
                            table = wandb.Table(
                                columns=["t", "action"],
                                data=[[t, int(a)] for t, a in enumerate(seq)],
                            )
                            log_payload[f"{self.name_prefix}/sequence_table"] = table
                        except Exception:
                            pass

                    # Use SB3's timestep as x-axis step for wandb
                    wandb.log(log_payload, step=int(self.model.num_timesteps))

                    # Reset buffer for this env and bump episode counter
                    self._actions_per_env[i] = []
                    self._episode_idx[i] += 1

        return True

