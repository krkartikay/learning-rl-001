import os
import wandb
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from berghain_env import BerghainEnv  # your env code
from gymnasium.wrappers import FlattenObservation
from wandb_action_callback import ActionLoggingCallback


def make_env(rank: int, seed: int | None = None):
    """Factory for SubprocVecEnv workers."""

    def _init():
        env = BerghainEnv()
        if seed is not None:
            # ensure distinct seeds per worker
            env.reset(seed=seed + rank)
        env = FlattenObservation(env)
        env = Monitor(env)
        return env

    return _init


def main():
    # Create and check a single env instance (sanity check only)
    _check_env = BerghainEnv()
    check_env(_check_env, warn=True)
    del _check_env

    wandb.init(
        project="berghain-sb3-dqn",
        config={
            "policy_type": "MlpPolicy",
            "total_timesteps": 10_000_000,
            "net_arch": [],
            "n_envs": 16,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # Build SubprocVecEnv
    n_envs = int(wandb.config.get("n_envs"))
    base_seed = 42
    vec_env = SubprocVecEnv([make_env(i, base_seed) for i in range(n_envs)])
    # vec_env = DummyVecEnv([make_env(0, base_seed)])

    # Define RL model
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=2,
        tensorboard_log=f"runs/berghain_sb3_{wandb.run.id}",
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,
        buffer_size=500_000,
        policy_kwargs=dict(net_arch=wandb.config.get("net_arch")),
    )

    # Train
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", wandb.config.total_timesteps))
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(
            [
                # Standard wandb SB3 callback (gradients, model checkpoints, etc.)
                WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{wandb.run.id}",
                    model_save_freq=100,
                    verbose=2,
                ),
                # Custom callback to record agent actions
                ActionLoggingCallback(wandb_run=wandb.run, name_prefix="actions"),
            ]
        ),
    )

    # Evaluate on a separate single environment
    eval_env = FlattenObservation(BerghainEnv())
    obs, info = eval_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        print(f"Step {obs=} {action=} {reward=}, success={info.get('success', False)}")


if __name__ == "__main__":
    main()
