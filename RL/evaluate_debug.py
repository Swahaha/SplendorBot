"""
Debug version of evaluate.py with detailed logging
"""

import torch
import numpy as np
import argparse
from typing import List, Dict, Any

from env_wrapper import SplendorEnv
from heuristic_agent import HeuristicAgent
from policy_network import MaskedPolicyNetwork


class PolicyAgent:
    """Wrapper for neural network policy agent."""

    def __init__(self, policy: MaskedPolicyNetwork, device: str = 'cpu', deterministic: bool = True):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.deterministic = deterministic

    def choose_action(self, obs: np.ndarray, mask: np.ndarray, state: Dict = None) -> int:
        """Choose action using policy network."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            mask_tensor = torch.from_numpy(mask).bool().to(self.device)

            action, _, _ = self.policy.get_action(obs_tensor, mask_tensor,
                                                  deterministic=self.deterministic)

            return action.item()


class RandomAgent:
    """Random agent that chooses uniformly from legal actions."""

    def __init__(self):
        pass

    def choose_action(self, obs: np.ndarray, mask: np.ndarray, state: Dict = None) -> int:
        """Choose random legal action."""
        legal_actions = np.where(mask)[0]
        if len(legal_actions) == 0:
            raise ValueError("No legal actions available! This should never happen.")
        return np.random.choice(legal_actions)


def evaluate_agents_debug(agents: List[Any], num_games: int, num_players: int = 2,
                          base_seed: int = 0) -> Dict[str, Any]:
    """
    Evaluate agents with detailed debug logging.
    """
    print(f"\n[DEBUG] Starting evaluation with {num_games} games, {num_players} players")
    print(f"[DEBUG] Creating environment...")

    env = SplendorEnv(num_players=num_players)
    print(f"[DEBUG] Environment created successfully")
    print(f"[DEBUG] OBS_DIM: {env.OBS_DIM}, ACTION_DIM: {env.ACTION_DIM}")

    # Statistics
    wins = [0] * num_players
    total_prestige = [[] for _ in range(num_players)]
    episode_lengths = []

    for game_idx in range(num_games):
        print(f"\n[DEBUG] ===== Game {game_idx + 1}/{num_games} =====")
        seed = base_seed + game_idx
        print(f"[DEBUG] Resetting with seed {seed}...")

        obs, mask, info = env.reset(seed=seed)
        print(f"[DEBUG] Reset complete. Current player: {info['current_player']}")
        print(f"[DEBUG] Obs shape: {obs.shape}, Mask shape: {mask.shape}, Legal actions: {np.sum(mask)}")

        done = False
        turn_count = 0

        while not done:
            current_player = info['current_player']
            print(f"\n[DEBUG] Turn {turn_count}, Player {current_player}")

            print(f"[DEBUG] Getting state summary...")
            state = env.game.state_summary()
            print(f"[DEBUG] State summary retrieved")

            # Check if there are legal moves
            legal_count = np.sum(mask)
            print(f"[DEBUG] Legal actions: {legal_count}")

            if not np.any(mask):
                print(f"[DEBUG] No legal moves - skipping turn")
                env.game.end_turn()
                obs = env._get_observation()
                mask = env._get_action_mask()
                info = {'current_player': env.game.current_player()}
                turn_count += 1

                if turn_count > 500:
                    print(f"[DEBUG] Turn limit reached (500), ending game")
                    done = True
                    break
                continue

            # Agent chooses action
            print(f"[DEBUG] Agent {current_player} choosing action...")
            print(f"[DEBUG] Agent type: {type(agents[current_player]).__name__}")

            action = agents[current_player].choose_action(obs, mask, state)
            print(f"[DEBUG] Agent chose action: {action}")

            # Handle case where agent returns None (no legal moves)
            if action is None:
                print(f"[DEBUG] Agent returned None - skipping turn")
                env.game.end_turn()
                obs = env._get_observation()
                mask = env._get_action_mask()
                info = {'current_player': env.game.current_player()}
                turn_count += 1
                continue

            # Execute action
            print(f"[DEBUG] Executing action {action}...")
            obs, reward, done, mask, info = env.step(action)
            print(f"[DEBUG] Action executed. Reward: {reward}, Done: {done}")
            turn_count += 1

            if turn_count % 10 == 0:
                print(f"[DEBUG] Turn {turn_count} complete")

        # Game finished
        print(f"\n[DEBUG] Game {game_idx + 1} finished after {turn_count} turns")
        final_state = env.game.state_summary()
        prestige_scores = [p['prestige_points'] for p in final_state['players']]
        print(f"[DEBUG] Prestige scores: {prestige_scores}")

        winner = np.argmax(prestige_scores)
        wins[winner] += 1
        print(f"[DEBUG] Winner: Player {winner}")

        for i in range(num_players):
            total_prestige[i].append(prestige_scores[i])

        episode_lengths.append(turn_count)

    # Compile results
    results = {
        'num_games': num_games,
        'num_players': num_players,
        'wins': wins,
        'win_rates': [w / num_games for w in wins],
        'avg_prestige': [np.mean(p) for p in total_prestige],
        'avg_episode_length': np.mean(episode_lengths),
    }

    print("\n=== Debug Evaluation Results ===")
    print(f"Games played: {num_games}")
    print(f"Average episode length: {results['avg_episode_length']:.1f}")
    for i in range(num_players):
        print(f"Player {i}: Wins: {wins[i]} ({results['win_rates'][i]:.1%}), Avg prestige: {results['avg_prestige'][i]:.2f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug evaluate Splendor agents')
    parser.add_argument('--num_games', type=int, default=2,
                       help='Number of games to play (default: 2)')
    parser.add_argument('--agent0', type=str, default='heuristic',
                       choices=['heuristic', 'random'],
                       help='Agent 0 type (default: heuristic)')
    parser.add_argument('--agent1', type=str, default='random',
                       choices=['heuristic', 'random'],
                       help='Agent 1 type (default: random)')

    args = parser.parse_args()

    # Build agent list
    print("[DEBUG] Creating environment to get dimensions...")
    env = SplendorEnv(num_players=2)

    agent_types = [args.agent0, args.agent1]
    agents = []

    print("\n[DEBUG] Creating agents...")
    for i, agent_type in enumerate(agent_types):
        if agent_type == 'heuristic':
            print(f"[DEBUG] Creating heuristic agent for player {i}...")
            agent = HeuristicAgent(env)
            print(f"  Agent {i}: Heuristic")
        elif agent_type == 'random':
            print(f"[DEBUG] Creating random agent for player {i}...")
            agent = RandomAgent()
            print(f"  Agent {i}: Random")
        agents.append(agent)

    print(f"\n[DEBUG] Agents created: {len(agents)}")

    # Run evaluation
    print("[DEBUG] Starting evaluation...")
    results = evaluate_agents_debug(
        agents,
        num_games=args.num_games,
        num_players=2,
        base_seed=10000
    )
