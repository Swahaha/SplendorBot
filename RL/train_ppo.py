"""
PPO Training for Splendor
Reinforcement learning with Proximal Policy Optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from collections import deque
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from env_wrapper import SplendorEnv
from policy_network import MaskedPolicyNetwork


class PPOTrainer:
    """PPO trainer with action masking."""

    def __init__(self, policy: MaskedPolicyNetwork, learning_rate: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize PPO trainer.

        Args:
            policy: Policy network
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            device: Device to train on
        """
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                   dones: np.ndarray, next_value: float) -> tuple:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Array of rewards [T]
            values: Array of value estimates [T]
            dones: Array of done flags [T]
            next_value: Value estimate for next state

        Returns:
            advantages: GAE advantages [T]
            returns: Discounted returns [T]
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        # Append next value
        values_extended = np.append(values, next_value)

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = values_extended[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        return advantages, returns

    def update(self, obs: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor,
              old_log_probs: torch.Tensor, advantages: torch.Tensor,
              returns: torch.Tensor, num_epochs: int = 4,
              batch_size: int = 256) -> dict:
        """
        PPO update step.

        Args:
            obs: Observations [N, obs_dim]
            masks: Action masks [N, action_dim]
            actions: Actions taken [N]
            old_log_probs: Log probs from behavior policy [N]
            advantages: GAE advantages [N]
            returns: Discounted returns [N]
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size

        Returns:
            stats: Dictionary of training statistics
        """
        self.policy.train()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(obs)
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }

        for _ in range(num_epochs):
            # Shuffle data
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get batch
                batch_obs = obs[batch_indices]
                batch_masks = masks[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_masks, batch_actions
                )

                # Compute ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # Policy loss (clipped surrogate objective)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_pred = values.squeeze(-1)
                value_loss = F.mse_loss(value_pred, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Statistics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    clip_fraction = ((ratios - 1.0).abs() > self.clip_epsilon).float().mean().item()

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(-entropy_loss.item())
                stats['total_loss'].append(loss.item())
                stats['approx_kl'].append(approx_kl)
                stats['clip_fraction'].append(clip_fraction)

        # Average statistics
        return {k: np.mean(v) for k, v in stats.items()}

    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def collect_trajectories(env: SplendorEnv, policy: MaskedPolicyNetwork,
                         num_episodes: int, device: str = 'cpu',
                         base_seed: int = 0) -> dict:
    """
    Collect trajectories using current policy.

    Returns:
        batch: Dictionary with trajectory data
    """
    policy.eval()

    # Storage
    all_obs = []
    all_masks = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_dones = []

    for ep in range(num_episodes):
        obs, mask, _ = env.reset(seed=base_seed + ep)

        ep_obs = []
        ep_masks = []
        ep_actions = []
        ep_log_probs = []
        ep_values = []
        ep_rewards = []
        ep_dones = []

        done = False

        while not done:
            # Convert to tensors
            obs_tensor = torch.from_numpy(obs).float().to(device)
            mask_tensor = torch.from_numpy(mask).bool().to(device)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs_tensor, mask_tensor,
                                                           deterministic=False)

            # Execute action
            next_obs, reward, done, next_mask, _ = env.step(action.item())

            # Store transition
            ep_obs.append(obs)
            ep_masks.append(mask)
            ep_actions.append(action.item())
            ep_log_probs.append(log_prob.item())
            ep_values.append(value.item())
            ep_rewards.append(reward)
            ep_dones.append(done)

            # Update
            obs = next_obs
            mask = next_mask

        # Add episode to batch
        all_obs.extend(ep_obs)
        all_masks.extend(ep_masks)
        all_actions.extend(ep_actions)
        all_log_probs.extend(ep_log_probs)
        all_values.extend(ep_values)
        all_rewards.extend(ep_rewards)
        all_dones.extend(ep_dones)

    return {
        'observations': np.array(all_obs, dtype=np.float32),
        'masks': np.array(all_masks, dtype=bool),
        'actions': np.array(all_actions, dtype=np.int64),
        'log_probs': np.array(all_log_probs, dtype=np.float32),
        'values': np.array(all_values, dtype=np.float32),
        'rewards': np.array(all_rewards, dtype=np.float32),
        'dones': np.array(all_dones, dtype=bool),
    }


def train_ppo(num_iterations: int = 1000, episodes_per_iter: int = 10,
             num_players: int = 2, hidden_dim: int = 512,
             learning_rate: float = 3e-4, gamma: float = 0.99,
             gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
             pretrained_path: str = None, device: str = None,
             checkpoint_dir: str = 'ppo_checkpoints', plot: bool = True):
    """
    Train PPO agent.

    Args:
        num_iterations: Number of training iterations
        episodes_per_iter: Episodes per iteration
        num_players: Number of players
        hidden_dim: Hidden layer size
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_epsilon: PPO clip parameter
        pretrained_path: Path to pretrained BC model
        device: Device
        checkpoint_dir: Checkpoint directory
        plot: Plot training curves

    Returns:
        trainer: Trained PPOTrainer
        history: Training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Initialize environment
    env = SplendorEnv(num_players=num_players)

    # Initialize policy
    policy = MaskedPolicyNetwork(env.OBS_DIM, env.ACTION_DIM, hidden_dim)

    # Load pretrained model if provided
    if pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])

    # Initialize trainer
    trainer = PPOTrainer(
        policy, learning_rate=learning_rate, gamma=gamma,
        gae_lambda=gae_lambda, clip_epsilon=clip_epsilon,
        device=device
    )

    # Training history
    history = {
        'iteration': [],
        'avg_reward': [],
        'avg_episode_length': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'approx_kl': [],
    }

    print(f"\nTraining PPO for {num_iterations} iterations...")

    for iteration in tqdm(range(num_iterations), desc="PPO Training"):
        # Collect trajectories
        batch = collect_trajectories(
            env, policy, episodes_per_iter, device,
            base_seed=iteration * episodes_per_iter
        )

        # Compute GAE
        advantages, returns = trainer.compute_gae(
            batch['rewards'], batch['values'],
            batch['dones'], next_value=0.0
        )

        # Convert to tensors
        obs_tensor = torch.from_numpy(batch['observations']).to(device)
        masks_tensor = torch.from_numpy(batch['masks']).to(device)
        actions_tensor = torch.from_numpy(batch['actions']).to(device)
        old_log_probs_tensor = torch.from_numpy(batch['log_probs']).to(device)
        advantages_tensor = torch.from_numpy(advantages).float().to(device)
        returns_tensor = torch.from_numpy(returns).float().to(device)

        # PPO update
        stats = trainer.update(
            obs_tensor, masks_tensor, actions_tensor,
            old_log_probs_tensor, advantages_tensor, returns_tensor
        )

        # Record history
        avg_reward = batch['rewards'].sum() / episodes_per_iter
        avg_length = len(batch['rewards']) / episodes_per_iter

        history['iteration'].append(iteration)
        history['avg_reward'].append(avg_reward)
        history['avg_episode_length'].append(avg_length)
        history['policy_loss'].append(stats['policy_loss'])
        history['value_loss'].append(stats['value_loss'])
        history['entropy'].append(stats['entropy'])
        history['approx_kl'].append(stats['approx_kl'])

        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"\nIteration {iteration+1}/{num_iterations}")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Avg length: {avg_length:.1f}")
            print(f"  Policy loss: {stats['policy_loss']:.4f}")
            print(f"  KL divergence: {stats['approx_kl']:.4f}")

        # Save checkpoint
        if (iteration + 1) % 50 == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_iter_{iteration+1}.pt"
            trainer.save(checkpoint_path)

    # Save final model
    final_path = f"{checkpoint_dir}/final_model.pt"
    trainer.save(final_path)
    print(f"\nTraining complete! Saved to {final_path}")

    # Plot training curves
    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        ax1.plot(history['avg_reward'])
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Avg Reward')
        ax1.set_title('Average Reward per Episode')
        ax1.grid(True)

        ax2.plot(history['avg_episode_length'])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Avg Length')
        ax2.set_title('Average Episode Length')
        ax2.grid(True)

        ax3.plot(history['policy_loss'], label='Policy Loss')
        ax3.plot(history['value_loss'], label='Value Loss')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.set_title('Losses')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(history['entropy'])
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Policy Entropy')
        ax4.grid(True)

        plt.tight_layout()
        plot_path = f"{checkpoint_dir}/training_curves.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training curves to {plot_path}")
        plt.close()

    return trainer, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of iterations (default: 1000)')
    parser.add_argument('--episodes_per_iter', type=int, default=10,
                       help='Episodes per iteration (default: 10)')
    parser.add_argument('--num_players', type=int, default=2,
                       help='Number of players (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension (default: 512)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained BC model')
    parser.add_argument('--checkpoint_dir', type=str, default='ppo_checkpoints',
                       help='Checkpoint directory (default: ppo_checkpoints)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    trainer, history = train_ppo(
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        num_players=args.num_players,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        pretrained_path=args.pretrained,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        plot=True
    )
