"""
Masked Policy Network for Splendor
Neural network with action masking for behavior cloning and PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MaskedPolicyNetwork(nn.Module):
    """
    Policy and value network with action masking.

    Architecture:
    - Input: observation vector + action mask
    - Hidden: 2-layer MLP with LayerNorm and SiLU activation
    - Outputs: policy logits + value estimate
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 512):
        """
        Initialize network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Feature extraction layers
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Small initialization for policy head (for stable early training)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [B, obs_dim]
            mask: Optional action mask [B, action_dim] (True = legal)

        Returns:
            logits: Policy logits [B, action_dim] (masked if mask provided)
            value: Value estimate [B, 1]
        """
        # Feature extraction
        x = self.fc1(obs)
        x = self.ln1(x)
        x = F.silu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.silu(x)

        # Policy logits
        logits = self.policy_head(x)

        # Apply action mask (set illegal actions to large negative value)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        # Value estimate
        value = self.value_head(x)

        return logits, value

    def get_action(self, obs: torch.Tensor, mask: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observation [B, obs_dim] or [obs_dim]
            mask: Action mask [B, action_dim] or [action_dim]
            deterministic: If True, take argmax; else sample

        Returns:
            action: Sampled action [B] or scalar
            log_prob: Log probability of action [B] or scalar
            value: Value estimate [B, 1] or [1]
        """
        # Handle single observation (add batch dimension)
        single_obs = obs.dim() == 1
        if single_obs:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)

        logits, value = self.forward(obs, mask)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        # Remove batch dimension if input was single
        if single_obs:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, mask: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO updates).

        Args:
            obs: Observations [B, obs_dim]
            mask: Action masks [B, action_dim]
            actions: Actions taken [B]

        Returns:
            log_probs: Log probabilities [B]
            values: Value estimates [B, 1]
            entropy: Policy entropy [B]
        """
        logits, values = self.forward(obs, mask)

        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Entropy (only over legal actions)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return action_log_probs, values, entropy


class BCTrainer:
    """Trainer for behavior cloning."""

    def __init__(self, policy: MaskedPolicyNetwork, learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5, device: str = 'cpu'):
        """
        Initialize trainer.

        Args:
            policy: Policy network
            learning_rate: Learning rate
            weight_decay: L2 regularization
            device: Device to train on
        """
        self.policy = policy.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_step(self, obs: torch.Tensor, mask: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[float, float]:
        """
        Single training step.

        Args:
            obs: Observations [B, obs_dim]
            mask: Action masks [B, action_dim]
            actions: Expert actions [B]

        Returns:
            loss: Cross-entropy loss
            accuracy: Prediction accuracy
        """
        self.policy.train()

        # Move to device
        obs = obs.to(self.device)
        mask = mask.to(self.device)
        actions = actions.to(self.device)

        # Forward pass
        logits, _ = self.policy(obs, mask)

        # Masked cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs, actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            pred_actions = torch.argmax(logits, dim=-1)
            accuracy = (pred_actions == actions).float().mean()

        return loss.item(), accuracy.item()

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor,
                actions: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate on validation set.

        Args:
            obs: Observations [B, obs_dim]
            mask: Action masks [B, action_dim]
            actions: Expert actions [B]

        Returns:
            loss: Cross-entropy loss
            accuracy: Prediction accuracy
        """
        self.policy.eval()

        with torch.no_grad():
            obs = obs.to(self.device)
            mask = mask.to(self.device)
            actions = actions.to(self.device)

            logits, _ = self.policy(obs, mask)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs, actions)

            pred_actions = torch.argmax(logits, dim=-1)
            accuracy = (pred_actions == actions).float().mean()

        return loss.item(), accuracy.item()

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
