"""
Test with torch import
"""

import torch
import numpy as np
from env_wrapper import SplendorEnv

print("Imported torch successfully")

class RandomAgent:
    def choose_action(self, obs, mask, state=None):
        legal_actions = np.where(mask)[0]
        return np.random.choice(legal_actions)

print("Creating environment...")
env = SplendorEnv(num_players=2)

print("Creating agents...")
agents = [RandomAgent(), RandomAgent()]

print("\nResetting environment...")
obs, mask, info = env.reset(seed=10000)
print(f"Reset successful! Legal actions: {mask.sum()}")

print("\nAll tests passed!")
