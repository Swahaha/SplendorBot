"""
Test the exact scenario from evaluate.py
"""

import numpy as np
from env_wrapper import SplendorEnv
from heuristic_agent import HeuristicAgent

class RandomAgent:
    def choose_action(self, obs, mask, state=None):
        legal_actions = np.where(mask)[0]
        return np.random.choice(legal_actions)

print("Step 1: Create environment (like line 222)")
env = SplendorEnv(num_players=2)
print("  Created env")

print("\nStep 2: Create agents (like lines 230-236)")
agent0 = HeuristicAgent(env)
agent1 = RandomAgent()
print("  Created agents")

print("\nStep 3: First reset in evaluate_agents (like line 83)")
obs, mask, info = env.reset(seed=10000)
print(f"  First reset successful. Legal actions: {mask.sum()}")

print("\nStep 4: Choose an action and step")
action = agent0.choose_action(obs, mask, env.game.state_summary())
obs, reward, done, mask, info = env.step(action)
print(f"  Step successful")

print("\nStep 5: Second reset for game 2 (like line 81 iteration 2)")
obs2, mask2, info2 = env.reset(seed=10001)
print(f"  Second reset successful. Legal actions: {mask2.sum()}")

print("\nAll tests passed!")
