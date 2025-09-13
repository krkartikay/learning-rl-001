from berghain_env import BerghainEnv

# filepath: /home/krkartikay/code/learning/berghain_env_test.py


def main():
    env = BerghainEnv()
    (state, info) = env.reset()
    print("RESET")
    print("state:", state)
    print("info:", info)
    print()

    done = False
    step_num = 0
    total_reward = 0

    while not done:
        step_num += 1
        # use env.next_person for clarity
        person = env.next_person
        need_A = env.remaining_rq_counts["A"] > 0
        need_B = env.remaining_rq_counts["B"] > 0
        needed_A = max(0, env.remaining_rq_counts["A"])
        needed_B = max(0, env.remaining_rq_counts["B"])

        if not need_A and not need_B:
            # If no more constraints needed, accept everyone
            action = 1
        else:
            r = env.remaining_accepts
            action = 1  # accept by default
            if needed_A >= needed_B:
                e = 0.2 * r
                if (
                    (not person["A"])
                    and (e < needed_A)
                    and (need_B and (not person["B"]))
                ):
                    action = 0  # reject
            else:
                e = 0.2 * r
                if (
                    (not person["B"])
                    and (e < needed_B)
                    and (need_A and (not person["A"]))
                ):
                    action = 0  # reject

        state, reward, done, terminated, info = env.step(action)
        total_reward += reward

        print(f"STEP {step_num}")
        print(" action:", "ACCEPT" if action == 1 else "REJECT")
        print(" person:", person)
        print(" state:", state)
        print(" remaining_accepts:", env.remaining_accepts)
        print(" remaining_rejects:", env.remaining_rejects)
        print(" remaining_rq_counts:", env.remaining_rq_counts)
        print(" reward:", reward)
        print(" done:", done, " terminated:", terminated)
        print(" info:", info)
        print(" total_reward so far:", total_reward)
        print("-" * 60)

    print("EPISODE FINISHED")
    print("final remaining_rq_counts:", env.remaining_rq_counts)
    print("final remaining_accepts:", env.remaining_accepts)
    print("final remaining_rejects:", env.remaining_rejects)
    print("final total_reward:", total_reward)


if __name__ == "__main__":
    main()
