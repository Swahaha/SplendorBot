"""
Minimal evaluate.py without tqdm
"""

import numpy as np
from env_wrapper import SplendorEnv

class RandomAgent:
    def choose_action(self, obs, mask, state=None):
        legal_actions = np.where(mask)[0]
        return np.random.choice(legal_actions)

# Create env
print("Creating environment...")
env = SplendorEnv(num_players=2)

# Create agents
print("Creating agents...")
agents = [RandomAgent(), RandomAgent()]

# Run 2 games
for game_idx in range(2):
    print(f"\nGame {game_idx + 1}")
    obs, mask, info = env.reset(seed=10000 + game_idx)
    print(f"  Reset successful")
    
    done = False
    turn_count = 0
    
    while not done and turn_count < 100:
        current_player = info['current_player']
        state = env.game.state_summary()
        
        if not np.any(mask):
            env.game.end_turn()
            obs = env._get_observation()
            mask = env._get_action_mask()
            info = {'current_player': env.game.current_player()}
            turn_count += 1
            continue
        
        action = agents[current_player].choose_action(obs, mask, state)
        obs, reward, done, mask, info = env.step(action)
        turn_count += 1
    
    print(f"  Game finished in {turn_count} turns")

print("\nAll tests passed!")
