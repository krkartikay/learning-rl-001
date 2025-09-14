import sys
import numpy as np
from typing import Any, Dict, List, Optional

import wandb
from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggingCallback(BaseCallback):
    """
    SB3 callback to record and log actions to Weights & Biases.
    """

    def __init__(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
        name_prefix: str = "actions",
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.name_prefix = name_prefix
        self.accept_actions = 0
        self.total_actions = 0

    def _init_callback(self) -> None:
        # Initialize buffers after model and env are set
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._actions_per_env = [[] for _ in range(n_envs)]
        self._episode_idx = [0 for _ in range(n_envs)]

    def _on_step(self) -> bool:
        # Extract actions and dones from learner locals
        actions = self.locals.get("actions")
        if actions is not None:
            accept_actions = sum(a for a in actions)
            total_actions = len(actions)
            self.accept_actions += accept_actions
            self.total_actions += total_actions
        return True

    def on_rollout_end(self):
        super().on_rollout_end()

        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        for i, done in enumerate(dones):
            if done:
                success = infos[i]["success"]
                self.wandb_run.log(
                    {
                        f"{self.name_prefix}/episode_success": int(success),
                    }
                )

        if self.total_actions > 0:
            accept_ratio = self.accept_actions / self.total_actions
        else:
            accept_ratio = 0
        self.wandb_run.log({f"{self.name_prefix}/accepts": accept_ratio})
        return True
