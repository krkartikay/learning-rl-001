import random
import numpy as np
from tqdm import tqdm
from berghain_env import BerghainEnv

env = BerghainEnv()

WEIGHTS = [[0, 0, 0, 0, 0],   # accept
           [0, 0, 0, 0, 0]]   # reject
BIAS = [0, 5]

def softmax(logits):
    # subtract max for numerical stability
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

def choose_action(state):
    # one neuron that predicts 0 or 1
    # logits = relu(weights * state + bias)
    logits = np.dot(WEIGHTS, state) + BIAS
    logits = np.maximum(0, logits)  # relu
    probs = softmax(logits)
    choice = random.choices([0, 1], weights=probs)
    return choice[0]
    
def run_eps():
    state, info = env.reset()
    step = 0
    total_reward = 0

    while True:
        action = choose_action(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        # print(f"{step} State: {state}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}.")
        state = new_state
        step += 1
        total_reward += reward
        if terminated or truncated:
            # print(f"{terminated=} {truncated=}")
            return total_reward

def main():
    print("Starting manual run...")
    total_reward = 0
    for i in tqdm(range(1000)):
        ep_rw = run_eps()
        total_reward += ep_rw
    print(f"Average Reward: {total_reward / 1000}")

main()