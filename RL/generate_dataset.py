"""
Dataset Generation for Behavior Cloning
Runs heuristic agent to generate supervised learning data.
"""

import numpy as np
import argparse
from tqdm import tqdm
from env_wrapper import SplendorEnv
from heuristic_agent import HeuristicAgent


def generate_dataset(num_games: int, num_players: int = 2, base_seed: int = 0,
                    output_path: str = 'bc_data.npz', verbose: bool = True):
    """
    Generate behavior cloning dataset by running heuristic agent.

    Args:
        num_games: Number of games to play
        num_players: Number of players per game
        base_seed: Base random seed
        output_path: Path to save dataset
        verbose: Print progress

    Returns:
        Dataset statistics dict
    """
    env = SplendorEnv(num_players=num_players)
    agent = HeuristicAgent(env)

    # Storage for dataset
    observations = []
    masks = []
    actions = []
    episode_lengths = []
    episode_rewards = []

    # Track statistics
    total_transitions = 0
    total_turns = 0

    # Generate episodes
    iterator = tqdm(range(num_games), desc="Generating dataset") if verbose else range(num_games)

    for episode_idx in iterator:
        seed = base_seed + episode_idx
        obs, mask, info = env.reset(seed=seed)

        episode_obs = []
        episode_masks = []
        episode_actions = []
        episode_reward = 0
        done = False
        turn_count = 0

        while not done:
            # Check if there are legal moves
            if not np.any(mask):
                # No legal moves - skip turn by calling end_turn()
                env.game.end_turn()
                obs = env._get_observation()
                mask = env._get_action_mask()
                turn_count += 1

                # Check if game should end (all players stuck)
                if turn_count > 500 or not np.any(mask):
                    done = True
                    break
                continue

            # Get game state for heuristic
            state = env.game.state_summary()

            # Heuristic chooses action
            action = agent.choose_action(obs, mask, state)

            # Handle case where heuristic returns None (no legal moves)
            if action is None:
                # Skip turn
                env.game.end_turn()
                obs = env._get_observation()
                mask = env._get_action_mask()
                turn_count += 1
                continue

            # Store transition
            episode_obs.append(obs.copy())
            episode_masks.append(mask.copy())
            episode_actions.append(action)

            # Execute action
            next_obs, reward, done, next_mask, info = env.step(action)

            episode_reward += reward
            turn_count += 1

            # Safety: limit episode length
            if turn_count > 500:
                done = True
                break

            # Update for next iteration
            obs = next_obs
            mask = next_mask

        # Store episode data
        observations.extend(episode_obs)
        masks.extend(episode_masks)
        actions.extend(episode_actions)
        episode_lengths.append(turn_count)
        episode_rewards.append(episode_reward)

        total_transitions += len(episode_actions)
        total_turns += turn_count

        # Update progress bar
        if verbose and (episode_idx + 1) % 10 == 0:
            iterator.set_postfix({
                'transitions': total_transitions,
                'avg_length': np.mean(episode_lengths),
                'avg_reward': np.mean(episode_rewards)
            })

    # Convert to numpy arrays
    observations_array = np.array(observations, dtype=np.float32)
    masks_array = np.array(masks, dtype=bool)
    actions_array = np.array(actions, dtype=np.int64)
    episode_lengths_array = np.array(episode_lengths, dtype=np.int32)
    episode_rewards_array = np.array(episode_rewards, dtype=np.float32)

    # Save dataset
    np.savez_compressed(
        output_path,
        observations=observations_array,
        masks=masks_array,
        actions=actions_array,
        episode_lengths=episode_lengths_array,
        episode_rewards=episode_rewards_array,
        num_games=num_games,
        num_players=num_players,
        base_seed=base_seed
    )

    # Compute statistics
    stats = {
        'num_games': num_games,
        'num_transitions': total_transitions,
        'avg_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'avg_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'min_episode_length': np.min(episode_lengths),
        'max_episode_length': np.max(episode_lengths),
        'obs_shape': observations_array.shape,
        'mask_shape': masks_array.shape,
        'action_shape': actions_array.shape,
    }

    if verbose:
        print("\n=== Dataset Generation Complete ===")
        print(f"Total games: {stats['num_games']}")
        print(f"Total transitions: {stats['num_transitions']}")
        print(f"Avg episode length: {stats['avg_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        print(f"Avg episode reward: {stats['avg_episode_reward']:.2f} ± {stats['std_episode_reward']:.2f}")
        print(f"Episode length range: [{stats['min_episode_length']}, {stats['max_episode_length']}]")
        print(f"Observation shape: {stats['obs_shape']}")
        print(f"Action shape: {stats['action_shape']}")
        print(f"Saved to: {output_path}")

    return stats


def load_dataset(path: str):
    """
    Load dataset from file.

    Returns:
        Dictionary with observations, masks, actions, and metadata
    """
    data = np.load(path)
    return {
        'observations': data['observations'],
        'masks': data['masks'],
        'actions': data['actions'],
        'episode_lengths': data['episode_lengths'],
        'episode_rewards': data['episode_rewards'],
        'num_games': int(data['num_games']),
        'num_players': int(data['num_players']),
        'base_seed': int(data['base_seed']),
    }


def verify_dataset(path: str):
    """Verify dataset integrity and print statistics."""
    data = load_dataset(path)

    print("\n=== Dataset Verification ===")
    print(f"Number of games: {data['num_games']}")
    print(f"Number of players: {data['num_players']}")
    print(f"Base seed: {data['base_seed']}")
    print(f"\nData shapes:")
    print(f"  Observations: {data['observations'].shape}")
    print(f"  Masks: {data['masks'].shape}")
    print(f"  Actions: {data['actions'].shape}")
    print(f"\nEpisode statistics:")
    print(f"  Lengths: {data['episode_lengths'].shape}")
    print(f"  Avg length: {np.mean(data['episode_lengths']):.1f}")
    print(f"  Rewards: {data['episode_rewards'].shape}")
    print(f"  Avg reward: {np.mean(data['episode_rewards']):.2f}")

    # Verify action validity
    num_invalid = np.sum(~data['masks'][np.arange(len(data['actions'])), data['actions']])
    print(f"\nAction validity:")
    print(f"  Total actions: {len(data['actions'])}")
    print(f"  Invalid actions: {num_invalid} ({100 * num_invalid / len(data['actions']):.2f}%)")

    # Check action distribution
    print(f"\nAction distribution:")
    unique, counts = np.unique(data['actions'], return_counts=True)
    print(f"  Unique actions used: {len(unique)} / {data['masks'].shape[1]}")
    print(f"  Most common actions: {unique[np.argsort(-counts)[:5]]}")
    print(f"  Least common actions: {unique[np.argsort(counts)[:5]]}")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate behavior cloning dataset')
    parser.add_argument('--num_games', type=int, default=1000,
                       help='Number of games to generate (default: 1000)')
    parser.add_argument('--num_players', type=int, default=2,
                       help='Number of players per game (default: 2)')
    parser.add_argument('--base_seed', type=int, default=0,
                       help='Base random seed (default: 0)')
    parser.add_argument('--output', type=str, default='bc_data.npz',
                       help='Output file path (default: bc_data.npz)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset after generation')
    parser.add_argument('--verify_only', type=str, default=None,
                       help='Only verify existing dataset at path')

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(args.verify_only)
    else:
        # Generate dataset
        stats = generate_dataset(
            num_games=args.num_games,
            num_players=args.num_players,
            base_seed=args.base_seed,
            output_path=args.output,
            verbose=True
        )

        # Verify if requested
        if args.verify:
            verify_dataset(args.output)
