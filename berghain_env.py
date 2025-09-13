"""
Implementation of RL environment for Berghain nightclub.

At each step a "person" with binary attributes comes in.
Agent has a choice to accept or reject the person.

Finally after 1000 persons are admitted the episode ends and the reward is
determined by whether or not some minimum quota of attributes is met.

State will be (
    remaining_accepts,
    remaining_rejects,
    remaining_rq_counts[A],
    remaining_rq_counts[B],
    next_person[A],
    next_person[B]
)

Actions: 0 (reject), 1 (accept)

"""

import random


class BerghainEnv:
    def __init__(self):
        self.attributes = ["A", "B"]
        self.prob_dist = {
            "ab": 0.45,
            "Ab": 0.20,
            "aB": 0.20,
            "AB": 0.15,
        }

    def reset(self, seed=None) -> tuple:  # (state, info)
        self.remaining_accepts = 1000
        self.remaining_rejects = 20000
        self.remaining_rq_counts = {"A": 600, "B": 600}

        # compute marginal frequencies from the joint distribution
        rel_fqs = {
            a: sum(p for k, p in self.prob_dist.items() if a in k)
            for a in self.attributes
        }

        # compute Pearson (phi) correlations for binary attributes
        correls = {a: {} for a in self.attributes}
        for a in self.attributes:
            for b in self.attributes:
                if a == b:
                    correls[a][b] = 1.0
                    continue
                p_a = rel_fqs[a]
                p_b = rel_fqs[b]
                p_ab = sum(p for k, p in self.prob_dist.items() if (a in k and b in k))
                denom = (p_a * (1 - p_a) * p_b * (1 - p_b)) ** 0.5
                correls[a][b] = 0.0 if denom == 0 else (p_ab - p_a * p_b) / denom

        self.next_person = self.random_person()
        info = {"rel_fqs": rel_fqs, "correls": correls}
        state = (
            self.remaining_accepts,
            self.remaining_rejects,
            self.remaining_rq_counts["A"],
            self.remaining_rq_counts["B"],
            self.next_person["A"],
            self.next_person["B"],
        )
        return (state, info)

    def random_person(self) -> dict:
        choice = random.choices(
            list(self.prob_dist.keys()), weights=list(self.prob_dist.values())
        )[0]
        return {a: (a in choice) for a in self.attributes}

    def step(self, action) -> tuple:
        if action == 1:  # accept
            self.remaining_accepts -= 1
            for a in self.attributes:
                if self.next_person[a]:
                    self.remaining_rq_counts[a] -= 1
        else:  # reject
            self.remaining_rejects -= 1

        # Check if episode is done
        done = self.remaining_accepts == 0 or self.remaining_rejects == 0

        # Get next person
        self.next_person = self.random_person()

        # Return state and reward
        state = (
            self.remaining_accepts,
            self.remaining_rejects,
            self.remaining_rq_counts["A"],
            self.remaining_rq_counts["B"],
            self.next_person["A"],
            self.next_person["B"],
        )
        all_constraints_met = all(v <= 0 for v in self.remaining_rq_counts.values())
        info = {}

        if not done:
            # reward = -1 if (action == 0) else 0  # Reject = -1, Accept = 0
            reward = 0  # For now let's just try to see if it can learn to meet all constraints or not
        else:
            # reward is negative sum of all unmet constraints times 1000
            info = {"success": all_constraints_met}
            reward = -sum(max(0, self.remaining_rq_counts[a]) for a in self.attributes)
            # reward is + 1000 if all constraints are met
            if all_constraints_met:
                reward += 1000

        terminated = False
        return state, reward, done, terminated, info
