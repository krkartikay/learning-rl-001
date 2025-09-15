from berghain_env import BerghainEnv
import numpy as np

env = BerghainEnv()

WEIGHTS = [0.5, 0.3, 0.2, 0.1, 0.1]


def choose_action(state):
    pass
    
def run_eps():
    state, info = env.reset()
    step = 0
    total_reward = 0

    while True:
        print("State:", state)
        action = choose_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        print(f"{step} State: {state}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}.")
        if terminated or truncated:
            print(f"{terminated=} {truncated=}")
            return total_reward

def main():
    print("Starting manual run...")
    run_eps()

main()