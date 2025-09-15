from berghain_env import BerghainEnv
from berghain_api_env import BerghainAPIEnv
import numpy as np
import random 

# env = BerghainEnv()
env = BerghainAPIEnv(base_url="https://berghain.challenges.listenlabs.ai/", scenario=1, player_id="ec4ae250-7815-41e1-8bf6-6cfc446dc81a")


def choose_action(state):
    print(state)
    remaining_accepts,rq_A,rq_B,next_A,next_B = state
    rq_overlap = rq_A + rq_B - remaining_accepts
    if(next_A and next_B):
        return 1
    elif(next_A):
        if(rq_A > 0):
            return 1
        elif(rq_B > 0 and remaining_accepts <= rq_B):
            return 0
        else:
            return 1
    elif(next_B):
        if(rq_B > 0):
            return 1
        elif(rq_A > 0 and remaining_accepts <= rq_A):
            return 0
        else:
            return 1
    else:
        if(remaining_accepts > rq_A + rq_B):
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