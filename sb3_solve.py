import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from berghain_env import BerghainEnv  # your env code

# Create env
env = BerghainEnv()
env = FlattenObservation(env)  # flatten tuple observation space
check_env(env, warn=True)  # sanity check

# Wrap env for SB3
vec_env = gym.wrappers.RecordEpisodeStatistics(env)

# Define PPO model
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    device="cpu",
)

# Train
model.learn(total_timesteps=200_000)

# Evaluate on new episodes
obs, info = vec_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vec_env.step(action)
    done = terminated or truncated
    print(f"Step reward={reward}, success={info.get('success', False)}")
