"""
Test heuristic agent specifically
"""

import numpy as np
from env_wrapper import SplendorEnv
from heuristic_agent import HeuristicAgent

print("Test 1: Create environment")
env = SplendorEnv(num_players=2)
print("  Created env successfully")

print("\nTest 2: Create heuristic agent")
agent = HeuristicAgent(env)
print("  Created agent successfully")

print("\nTest 3: Reset environment")
obs, mask, info = env.reset(seed=10000)
print(f"  Reset successful. Legal actions: {mask.sum()}")

print("\nTest 4: Get game state")
state = env.game.state_summary()
print(f"  Got state. Current player: {state['current_player']}")

print("\nTest 5: Agent choose action")
action = agent.choose_action(obs, mask, state)
print(f"  Agent chose action: {action}")

print("\nAll tests passed!")
