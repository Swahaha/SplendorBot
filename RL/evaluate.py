"""
Evaluation Script for Splendor Agents
Test and compare different agents.
"""

# IMPORTANT: Import game modules BEFORE torch to avoid library conflicts
from env_wrapper import SplendorEnv
from heuristic_agent import HeuristicAgent

# Import other modules (numpy, argparse, etc. don't conflict)
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

# Delay torch imports until after game module is loaded
# policy_network imports torch, so we'll import it lazily when needed


class PolicyAgent:
    """Wrapper for neural network policy agent."""

    def __init__(self, policy, device: str = 'cpu', deterministic: bool = True):
        import torch  # Import here to avoid conflicts
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.deterministic = deterministic

    def choose_action(self, obs: np.ndarray, mask: np.ndarray, state: Dict = None) -> int:
        """Choose action using policy network."""
        import torch  # Import here to avoid conflicts
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


def evaluate_agents(agents: List[Any], env: SplendorEnv, num_games: int,
                   base_seed: int = 0, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate agents in head-to-head games.

    Args:
        agents: List of agent objects (must have choose_action method)
        env: SplendorEnv instance to use for evaluation
        num_games: Number of games to play
        base_seed: Base random seed
        verbose: Print progress

    Returns:
        results: Dictionary with evaluation statistics
    """
    num_players = env.num_players
    assert len(agents) == num_players, f"Need {num_players} agents, got {len(agents)}"

    # Statistics
    wins = [0] * num_players
    total_prestige = [[] for _ in range(num_players)]
    episode_lengths = []
    actions_taken = [[] for _ in range(num_players)]

    iterator = tqdm(range(num_games), desc="Evaluating") if verbose else range(num_games)

    for game_idx in iterator:
        seed = base_seed + game_idx
        obs, mask, info = env.reset(seed=seed)

        done = False
        turn_count = 0
        player_actions = [0] * num_players

        while not done:
            current_player = info['current_player']
            state = env.game.state_summary()

            # Check if there are legal moves
            if not np.any(mask):
                # No legal moves for this player - skip their turn
                env.game.end_turn()
                obs = env._get_observation()
                mask = env._get_action_mask()
                info = {'current_player': env.game.current_player()}
                turn_count += 1

                # Check if all players are stuck (game deadlock)
                if turn_count > 500:
                    done = True
                    break
                continue

            # Agent chooses action
            action = agents[current_player].choose_action(obs, mask, state)

            # Handle case where agent returns None (no legal moves)
            if action is None:
                # Skip turn
                env.game.end_turn()
                obs = env._get_observation()
                mask = env._get_action_mask()
                info = {'current_player': env.game.current_player()}
                turn_count += 1
                continue

            player_actions[current_player] += 1

            # Execute action
            obs, reward, done, mask, info = env.step(action)
            turn_count += 1

        # Game finished - determine winner
        final_state = env.game.state_summary()
        prestige_scores = [p['prestige_points'] for p in final_state['players']]

        winner = np.argmax(prestige_scores)
        wins[winner] += 1

        for i in range(num_players):
            total_prestige[i].append(prestige_scores[i])
            actions_taken[i].append(player_actions[i])

        episode_lengths.append(turn_count)

        # Update progress
        if verbose and (game_idx + 1) % 10 == 0:
            win_rates = [w / (game_idx + 1) for w in wins]
            iterator.set_postfix({
                f'P{i}_WR': f'{wr:.2f}' for i, wr in enumerate(win_rates)
            })

    # Compile results
    results = {
        'num_games': num_games,
        'num_players': num_players,
        'wins': wins,
        'win_rates': [w / num_games for w in wins],
        'avg_prestige': [np.mean(p) for p in total_prestige],
        'std_prestige': [np.std(p) for p in total_prestige],
        'avg_actions': [np.mean(a) for a in actions_taken],
        'avg_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
    }

    if verbose:
        print("\n=== Evaluation Results ===")
        print(f"Games played: {num_games}")
        print(f"Average episode length: {results['avg_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
        print()

        for i in range(num_players):
            print(f"Player {i}:")
            print(f"  Wins: {wins[i]} ({results['win_rates'][i]:.1%})")
            print(f"  Avg prestige: {results['avg_prestige'][i]:.2f} ± {results['std_prestige'][i]:.2f}")
            print(f"  Avg actions: {results['avg_actions'][i]:.1f}")

    return results


def load_policy_agent(checkpoint_path: str, obs_dim: int, action_dim: int,
                     hidden_dim: int = 512, device: str = None,
                     deterministic: bool = True) -> PolicyAgent:
    """Load trained policy agent from checkpoint."""
    import torch  # Import here to avoid conflicts
    from policy_network import MaskedPolicyNetwork

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    policy = MaskedPolicyNetwork(obs_dim, action_dim, hidden_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    return PolicyAgent(policy, device, deterministic)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Splendor agents')
    parser.add_argument('--num_games', type=int, default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--num_players', type=int, default=2,
                       help='Number of players (default: 2)')
    parser.add_argument('--base_seed', type=int, default=10000,
                       help='Base random seed (default: 10000)')

    # Agent configuration
    parser.add_argument('--agent0', type=str, default='heuristic',
                       choices=['heuristic', 'random', 'policy'],
                       help='Agent 0 type (default: heuristic)')
    parser.add_argument('--agent1', type=str, default='random',
                       choices=['heuristic', 'random', 'policy'],
                       help='Agent 1 type (default: random)')
    parser.add_argument('--agent2', type=str, default=None,
                       choices=['heuristic', 'random', 'policy'],
                       help='Agent 2 type (for 3+ players)')
    parser.add_argument('--agent3', type=str, default=None,
                       choices=['heuristic', 'random', 'policy'],
                       help='Agent 3 type (for 4 players)')

    # Policy agent configuration
    parser.add_argument('--policy_checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to policy checkpoint (default: checkpoints/best_model.pt)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for policy (default: 512)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default: deterministic)')

    args = parser.parse_args()

    # Build agent list
    env = SplendorEnv(num_players=args.num_players)
    obs_dim = env.OBS_DIM
    action_dim = env.ACTION_DIM

    agent_types = [args.agent0, args.agent1, args.agent2, args.agent3][:args.num_players]
    agents = []

    print("Creating agents...")
    for i, agent_type in enumerate(agent_types):
        if agent_type == 'heuristic':
            agent = HeuristicAgent(env)
            print(f"  Agent {i}: Heuristic")
        elif agent_type == 'random':
            agent = RandomAgent()
            print(f"  Agent {i}: Random")
        elif agent_type == 'policy':
            agent = load_policy_agent(
                args.policy_checkpoint,
                obs_dim,
                action_dim,
                args.hidden_dim,
                deterministic=not args.stochastic
            )
            mode = 'stochastic' if args.stochastic else 'deterministic'
            print(f"  Agent {i}: Policy ({mode})")
        agents.append(agent)

    # Run evaluation
    # TEMP: Use simplified evaluation to avoid segfault
    print("\nRunning evaluation (simplified version)...")

    wins = [0] * args.num_players
    for game_idx in range(args.num_games):
        print(f"\nGame {game_idx + 1}/{args.num_games}")
        obs, mask, info = env.reset(seed=args.base_seed + game_idx)
        print(f"  Reset successful")

        done = False
        turn_count = 0

        while not done and turn_count < 500:
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
            if action is None:
                env.game.end_turn()
                obs = env._get_observation()
                mask = env._get_action_mask()
                info = {'current_player': env.game.current_player()}
                turn_count += 1
                continue

            obs, reward, done, mask, info = env.step(action)
            turn_count += 1

        final_state = env.game.state_summary()
        prestige_scores = [p['prestige_points'] for p in final_state['players']]
        winner = np.argmax(prestige_scores)
        wins[winner] += 1
        print(f"  Game finished in {turn_count} turns. Winner: Player {winner} with {prestige_scores[winner]} points")

    print("\n=== Results ===")
    for i in range(args.num_players):
        print(f"Player {i}: {wins[i]} wins ({100*wins[i]/args.num_games:.1f}%)")

    return {'wins': wins}

    # results = evaluate_agents(
    #     agents,
    #     env,
    #     num_games=args.num_games,
    #     base_seed=args.base_seed,
    #     verbose=True
    # )
    # return results


if __name__ == '__main__':
    main()
