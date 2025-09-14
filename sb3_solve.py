import wandb
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from berghain_env import BerghainEnv  # your env code
from gymnasium.wrappers import FlattenObservation

# Create env
env = BerghainEnv()
env = FlattenObservation(env)
check_env(env, warn=True)  # sanity check

wandb.init(
    project="berghain-sb3",
    config={
        "policy_type": "MlpPolicy",
        "total_timesteps": 1_000_000,
    },
    sync_tensorboard=True,
    save_code=True,
)

# Wrap env for SB3
vec_env = gym.wrappers.RecordEpisodeStatistics(env)

# Define RL model
model = DQN(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=f"runs/berghain_sb3_{wandb.run.id}",
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[16]),
)

# Train
model.learn(
    total_timesteps=wandb.config.total_timesteps,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb.run.id}",
        model_save_freq=100,
        verbose=2,
    ),
)

# Evaluate on new episodes
obs, info = vec_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vec_env.step(action)
    done = terminated or truncated
    print(f"Step reward={reward}, success={info.get('success', False)}")
