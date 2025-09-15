from berghain_env import BerghainEnv
from berghain_api_env import BerghainAPIEnv
import numpy as np
import random 

# env = BerghainEnv()
env = BerghainAPIEnv(base_url="https://berghain.challenges.listenlabs.ai/", scenario=1, player_id="ec4ae250-7815-41e1-8bf6-6cfc446dc81a")


def choose_action(state):
    remaining_accepts = state[0]
    n = (len(state) - 1) // 2
    rq = state[1:n+1]
    nexts = state[n+1:2*n+1]

    rq_total = sum(rq)
    rq_overlap = rq_total - remaining_accepts

    # If all nexts are True, accept
    if all(nexts):
        return 1

    # If remaining_accepts is equal to any rq and that next is False, reject
    for i in range(n):
        if rq[i-1] == remaining_accepts and not nexts[i-1]:
            return 0
    
    # If any next is True and its rq > 0, accept
    for i in range(n):
        if nexts[i-1]:
            if rq[i-1] > 0:
                return 1
            # If another rq > 0 and remaining_accepts <= that rq, reject
            for j in range(n):
                if j != i and rq[j-1] > 0 and remaining_accepts <= rq[j-1]:
                    return 0
            return 1

    # If no nexts are True
    if remaining_accepts > rq_total:
        return 1
    else:
        return 0

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