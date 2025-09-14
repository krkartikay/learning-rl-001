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

import numpy as np
import gymnasium as gym
from gymnasium import spaces

REWARD_SCALE = 0.001
INF = 20_000


class BerghainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.attributes = ["A", "B"]
        self.prob_dist = {
            "ab": 0.45,
            "Ab": 0.20,
            "aB": 0.20,
            "AB": 0.15,
        }

        # Gymnasium spaces (keep state layout unchanged: 6 scalars)
        # remaining_accepts: 0..1000, remaining_rejects: 0..20000
        # remaining_rq_counts[A], [B]: 0..600
        # next_person[A], [B]: 0/1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([1000, 1000, 1000, 1, 1], dtype=np.int32),
            dtype=np.int32,
        )

        self.np_random = np.random.default_rng()

    def seed(self, seed: int | None = None):
        # Optional legacy seeding helper
        self.np_random = np.random.default_rng(seed)

    def _make_state(self):
        # Return state as a tuple of integers
        # (accepts, rejects, rq_A, rq_B, next_A, next_B)
        return np.array(
            [
                int(self.remaining_accepts),
                int(self.remaining_rq_counts["A"]),
                int(self.remaining_rq_counts["B"]),
                int(self.next_person["A"]),
                int(self.next_person["B"]),
            ],
            dtype=np.int32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

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
        state = self._make_state()
        # Gymnasium reset: (observation, info)
        return state, info

    def random_person(self) -> dict:
        keys = list(self.prob_dist.keys())
        probs = np.fromiter(self.prob_dist.values(), dtype=float)
        # Use env RNG for reproducibility with seeding
        choice = self.np_random.choice(keys, p=probs)
        return {a: (a in choice) for a in self.attributes}

    def step(self, action):
        # Validate action via space (raises if invalid in debug/tools)
        assert self.action_space.contains(action), "Invalid action"

        prev_state = self._make_state()

        if action == 1:  # accept
            self.remaining_accepts -= 1
            for a in self.attributes:
                if self.next_person[a]:
                    self.remaining_rq_counts[a] = max(
                        0, self.remaining_rq_counts[a] - 1
                    )
        else:  # reject
            self.remaining_rejects -= 1

        # Gymnasium semantics: terminated for natural task completion,
        # truncated for hitting an external limit (reject cap).
        terminated = self.remaining_accepts == 0
        truncated = self.remaining_rejects == 0
        if not self.feasible(self._make_state()):
            truncated = True  # no point continuing if infeasible
        done = terminated or truncated

        # Prepare next person (even if done; some agents inspect next obs)
        self.next_person = self.random_person()

        state = self._make_state()
        all_constraints_met = all(v <= 0 for v in self.remaining_rq_counts.values())
        info = {}

        # Keep your reward function unchanged
        if not done:
            reward = -1 if (action == 0) else 0  # Reject = -1, Accept = 0
            # reward = 0  # For now let's just try to see if it can learn to meet all constraints or not
        else:
            info = {"success": all_constraints_met}
            # reward = 0 if all_constraints_met else -INF
            reward = 0

        # update reward with potential-based shaping (if any)
        reward += self.reward_potential(state) - self.reward_potential(prev_state)
        reward *= REWARD_SCALE  # scale down reward

        # Gymnasium step: (obs, reward, terminated, truncated, info)
        return state, reward, terminated, truncated, info

    def feasible(self, state) -> bool:
        # Check if we can even reach success from this state
        accepts, rq_A, rq_B, _, _ = state
        if max(rq_A, rq_B) > accepts:
            return False
        return True

    def reward_potential(self, state):
        # Custom reward potential function for reward shaping (optional)
        # state is a tuple as defined in _make_state
        _, rq_A, rq_B, _, _ = state
        remaining_counts = rq_A + rq_B
        return -remaining_counts

    # Optional (no-op) render/close to satisfy wrappers if needed
    def render(self):
        pass

    def close(self):
        pass
