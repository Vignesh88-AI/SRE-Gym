import requests
from typing import Any, Dict, Optional

class SREGymEnv:
    """ Python client for the SRE-Gym environment. """

    def __init__(self, base_url="https://argonite3-sre-gym.hf.space"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "task1_easy") -> Dict[str, Any]:
        """ Reset the environment with the given task. """
        url = f"{self.base_url}/reset"
        resp = requests.post(url, json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """ Execute one action in the environment. """
        url = f"{self.base_url}/step"
        resp = requests.post(url, json=action, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """ Get the full internal state of the environment. """
        url = f"{self.base_url}/state"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """ Check server health. """
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
