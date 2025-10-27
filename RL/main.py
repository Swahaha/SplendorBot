"""
Main entry point for Splendor RL training pipeline.
Provides a simple interface to run the full training workflow.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for splendor_game import
sys.path.append(str(Path(__file__).parent.parent / 'build'))


def run_full_pipeline(num_games: int = 1000, num_epochs: int = 50,
                     num_eval_games: int = 100, num_players: int = 2,
                     use_ppo: bool = False, ppo_iterations: int = 500):
    """
    Run the complete training and evaluation pipeline.

    Steps:
    1. Generate dataset using heuristic agent
    2. Train BC policy
    3. Evaluate BC vs heuristic and random
    4. (Optional) Train PPO
    5. (Optional) Evaluate PPO vs others
    """
    from generate_dataset import generate_dataset, verify_dataset
    from train_bc import train_bc
    from evaluate import evaluate_agents
    from heuristic_agent import HeuristicAgent
    from env_wrapper import SplendorEnv
    from evaluate import PolicyAgent, RandomAgent, load_policy_agent

    print("=" * 80)
    print("SPLENDOR RL TRAINING PIPELINE")
    print("=" * 80)

    # Step 1: Generate dataset
    print("\n[1/5] Generating dataset...")
    dataset_path = 'bc_data.npz'
    stats = generate_dataset(
        num_games=num_games,
        num_players=num_players,
        base_seed=0,
        output_path=dataset_path,
        verbose=True
    )

    # Verify dataset
    print("\n[2/5] Verifying dataset...")
    verify_dataset(dataset_path)

    # Step 2: Train BC policy
    print("\n[3/5] Training BC policy...")
    trainer, history = train_bc(
        dataset_path=dataset_path,
        num_epochs=num_epochs,
        batch_size=256,
        learning_rate=3e-4,
        val_split=0.1,
        hidden_dim=512,
        device=None,  # Auto-detect
        checkpoint_dir='checkpoints',
        plot=True
    )

    print(f"\nBC Training complete!")
    print(f"Best validation accuracy: {history['best_val_acc']:.3f}")

    # Step 3: Evaluate BC policy
    print("\n[4/5] Evaluating BC policy...")

    env = SplendorEnv(num_players=num_players)

    # BC vs Heuristic
    print("\n--- BC Policy vs Heuristic ---")
    bc_agent = load_policy_agent('checkpoints/best_model.pt', env.OBS_DIM, env.ACTION_DIM)
    heuristic_agent = HeuristicAgent(env)
    agents = [bc_agent, heuristic_agent]

    results_bc_heur = evaluate_agents(
        agents, num_games=num_eval_games, num_players=2,
        base_seed=10000, verbose=True
    )

    # BC vs Random
    print("\n--- BC Policy vs Random ---")
    random_agent = RandomAgent()
    agents = [bc_agent, random_agent]

    results_bc_random = evaluate_agents(
        agents, num_games=num_eval_games, num_players=2,
        base_seed=20000, verbose=True
    )

    # Heuristic vs Random (baseline)
    print("\n--- Heuristic vs Random (Baseline) ---")
    agents = [heuristic_agent, random_agent]

    results_heur_random = evaluate_agents(
        agents, num_games=num_eval_games, num_players=2,
        base_seed=30000, verbose=True
    )

    # Step 4 & 5: PPO (optional)
    if use_ppo:
        print("\n[5/5] Training PPO policy...")
        from train_ppo import train_ppo

        ppo_trainer, ppo_history = train_ppo(
            num_iterations=ppo_iterations,
            episodes_per_iter=10,
            num_players=num_players,
            hidden_dim=512,
            learning_rate=3e-4,
            gamma=0.99,
            pretrained_path='checkpoints/best_model.pt',
            device=None,
            checkpoint_dir='ppo_checkpoints',
            plot=True
        )

        print("\n--- PPO Policy vs Heuristic ---")
        ppo_agent = load_policy_agent('ppo_checkpoints/final_model.pt', env.OBS_DIM, env.ACTION_DIM)
        agents = [ppo_agent, heuristic_agent]

        results_ppo_heur = evaluate_agents(
            agents, num_games=num_eval_games, num_players=2,
            base_seed=40000, verbose=True
        )

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    print("\nSummary of Results:")
    print(f"\n1. BC vs Heuristic:")
    print(f"   BC win rate: {results_bc_heur['win_rates'][0]:.1%}")
    print(f"   BC avg prestige: {results_bc_heur['avg_prestige'][0]:.1f}")

    print(f"\n2. BC vs Random:")
    print(f"   BC win rate: {results_bc_random['win_rates'][0]:.1%}")
    print(f"   BC avg prestige: {results_bc_random['avg_prestige'][0]:.1f}")

    print(f"\n3. Heuristic vs Random (Baseline):")
    print(f"   Heuristic win rate: {results_heur_random['win_rates'][0]:.1%}")

    if use_ppo:
        print(f"\n4. PPO vs Heuristic:")
        print(f"   PPO win rate: {results_ppo_heur['win_rates'][0]:.1%}")
        print(f"   PPO avg prestige: {results_ppo_heur['avg_prestige'][0]:.1f}")

    print("\nModels saved:")
    print("  - checkpoints/best_model.pt (BC policy)")
    if use_ppo:
        print("  - ppo_checkpoints/final_model.pt (PPO policy)")

    print("\n" + "=" * 80)


def quick_test():
    """Quick test to verify all components work."""
    from env_wrapper import SplendorEnv
    from heuristic_agent import HeuristicAgent
    from evaluate import RandomAgent, evaluate_agents

    print("Running quick test...")

    env = SplendorEnv(num_players=2)

    # Test environment
    print("\n1. Testing environment...")
    obs, mask, info = env.reset(seed=42)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action mask shape: {mask.shape}")
    print(f"   Legal actions: {mask.sum()}")

    # Test heuristic agent
    print("\n2. Testing heuristic agent...")
    heuristic = HeuristicAgent(env)
    state = env.game.state_summary()
    action = heuristic.choose_action(obs, mask, state)
    print(f"   Chose action: {action}")

    # Test step
    print("\n3. Testing environment step...")
    obs, reward, done, mask, info = env.step(action)
    print(f"   Reward: {reward}")
    print(f"   Done: {done}")

    # Test quick evaluation
    print("\n4. Testing evaluation (10 games)...")
    random = RandomAgent()
    agents = [heuristic, random]
    results = evaluate_agents(agents, num_games=10, num_players=2, verbose=False)
    print(f"   Heuristic win rate: {results['win_rates'][0]:.1%}")
    print(f"   Random win rate: {results['win_rates'][1]:.1%}")

    print("\nQuick test passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splendor RL training pipeline')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'test', 'quick'],
                       help='Mode: full pipeline, test, or quick demo')

    # Pipeline arguments
    parser.add_argument('--num_games', type=int, default=1000,
                       help='Number of games for dataset (default: 1000)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of BC training epochs (default: 50)')
    parser.add_argument('--num_eval_games', type=int, default=100,
                       help='Number of evaluation games (default: 100)')
    parser.add_argument('--num_players', type=int, default=2,
                       help='Number of players (default: 2)')
    parser.add_argument('--use_ppo', action='store_true',
                       help='Also train PPO (takes longer)')
    parser.add_argument('--ppo_iterations', type=int, default=500,
                       help='Number of PPO iterations (default: 500)')

    args = parser.parse_args()

    if args.mode == 'test' or args.mode == 'quick':
        quick_test()
    elif args.mode == 'full':
        run_full_pipeline(
            num_games=args.num_games,
            num_epochs=args.num_epochs,
            num_eval_games=args.num_eval_games,
            num_players=args.num_players,
            use_ppo=args.use_ppo,
            ppo_iterations=args.ppo_iterations
        )
