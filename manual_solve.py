import random
import numpy as np
from tqdm import tqdm
from berghain_env import BerghainEnv
from matplotlib import pyplot as plt

env = BerghainEnv()

WEIGHTS = [[-.01, .01, .01, 0, 0],   # reject
              [0, 0, 0, 1, 1]]   # accept
BIAS = [0, 0]

NUM_EPISODES = 100

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
    num_accepts = 0

    while True:
        action = choose_action(state)
        new_state, reward, terminated, truncated, info = env.step(action)

        if action == 0:  # assuming 0 = accept
            num_accepts += 1

        state = new_state
        step += 1
        total_reward += reward

        if terminated or truncated:
            return total_reward, num_accepts, step, info


def main():
    print("Starting manual run...")
    rewards, accepts, steps = [], [], []
    successes = 0
    num_eps = NUM_EPISODES

    for _ in tqdm(range(num_eps)):
        ep_rw, ep_accepts, ep_steps, info = run_eps()
        rewards.append(ep_rw)
        accepts.append(ep_accepts)
        steps.append(ep_steps)
        if "success" in info and info["success"]:
            successes += 1

    # Convert to numpy arrays for convenience
    rewards = np.array(rewards)
    accepts = np.array(accepts)
    steps = np.array(steps)

    # Summary stats
    print(f"Average Reward: {rewards.mean():.3f}")
    print(f"Average Accepts per Episode: {accepts.mean():.2f}")
    print(f"Average Steps per Episode: {steps.mean():.2f}")
    print(f"% Accepts per Episode (avg): {(accepts.mean() / steps.mean()) * 100:.2f}%")
    print(f"% Successes: {(successes / num_eps) * 100:.2f}%")

    # Histograms
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].hist(rewards, bins=30, color="steelblue", edgecolor="black")
    axs[0, 0].set_title("Reward Distribution")
    axs[0, 0].set_xlabel("Total Reward")
    axs[0, 0].set_ylabel("Count")

    axs[0, 1].hist(accepts, bins=30, color="seagreen", edgecolor="black")
    axs[0, 1].set_title("Accepts per Episode")
    axs[0, 1].set_xlabel("# Accepts")
    axs[0, 1].set_ylabel("Count")

    axs[1, 0].hist(steps, bins=30, color="mediumpurple", edgecolor="black")
    axs[1, 0].set_title("Steps per Episode")
    axs[1, 0].set_xlabel("# Steps")
    axs[1, 0].set_ylabel("Count")

    axs[1, 1].bar(["Success", "Failure"],
                  [successes, num_eps - successes],
                  color=["darkorange", "gray"], edgecolor="black")
    axs[1, 1].set_title("Episode Success vs Failure")

    plt.tight_layout()
    plt.savefig("episode_stats.png")
    plt.close()
    print("Saved histograms to episode_stats.png")


main()