import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests


class BerghainAPIEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_url: str, scenario: int, player_id: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.scenario = scenario
        self.player_id = player_id

        self.action_space = spaces.Discrete(2)
        # We don’t know number of attributes yet → set in reset()
        self.observation_space = None  

        self.game_id = None
        self.attributes = []
        self.remaining_accepts = 1000
        self.remaining_rejects = 20000
        self.remaining_rq_counts = {}
        self.next_person = {}

    def _format_obs(self):
        # Build state = [remaining_accepts, rq_counts..., next_attrs...]
        rq_values = [self.remaining_rq_counts.get(a, 0) for a in self.attributes]
        next_values = [int(self.next_person.get(a, 0)) for a in self.attributes]
        return np.array(
            [self.remaining_accepts, *rq_values, *next_values],
            dtype=np.int32,
        )

    def reset(self, *, seed=None, options=None):
        r = requests.get(
            f"{self.base_url}/new-game",
            params={"scenario": self.scenario, "playerId": self.player_id},
        )
        r.raise_for_status()
        data = r.json()
        self.game_id = data["gameId"]

        # set attributes dynamically from constraints
        self.attributes = [c["attribute"] for c in data["constraints"]]
        self.remaining_rq_counts = {
            c["attribute"]: c["minCount"] for c in data["constraints"]
        }
        self.remaining_accepts = 1000
        self.remaining_rejects = 20000

        # also fix observation space now that we know number of attributes
        n = 1 + len(self.attributes) + len(self.attributes)  # accepts + rq + next
        self.observation_space = spaces.Box(
            low=0, high=20000, shape=(n,), dtype=np.int32
        )

        # fetch first person
        r = requests.get(
            f"{self.base_url}/decide-and-next",
            params={"gameId": self.game_id, "personIndex": 0},
        )
        r.raise_for_status()
        resp = r.json()
        self.next_person = resp["nextPerson"]["attributes"]

        state = self._format_obs()
        info = {"constraints": data["constraints"]}
        return state, info

    def step(self, action):
        accept =  "true" if action == 1 else "false"
        idx = (1000 - self.remaining_accepts) + (20000 - self.remaining_rejects)

        r = requests.get(
            f"{self.base_url}/decide-and-next",
            params={"gameId": self.game_id, "personIndex": idx, "accept": accept},
        )
        r.raise_for_status()
        resp = r.json()

        # update counters
        if accept:
            self.remaining_accepts -= 1
            for attr in self.attributes:
                if self.next_person.get(attr, False):
                    self.remaining_rq_counts[attr] = max(
                        0, self.remaining_rq_counts[attr] - 1
                    )
        else:
            self.remaining_rejects -= 1

        if resp.get("nextPerson"):
            self.next_person = resp["nextPerson"]["attributes"]

        terminated = resp.get("status") == "completed"
        truncated = resp.get("status") == "failed"

        state = self._format_obs()
        reward = 0.0
        info = {}
        if terminated:
            info["final_rejected"] = resp.get("rejectedCount", 0)
        if truncated:
            info["reason"] = resp.get("reason", "")

        return state, reward, terminated, truncated, info
