"""
Behavior Cloning Training Script
Train policy network from heuristic demonstrations.
"""

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from policy_network import MaskedPolicyNetwork, BCTrainer
from env_wrapper import SplendorEnv
from generate_dataset import load_dataset


def train_bc(dataset_path: str, num_epochs: int = 50, batch_size: int = 256,
             learning_rate: float = 3e-4, val_split: float = 0.1,
             hidden_dim: int = 512, device: str = None,
             checkpoint_dir: str = 'checkpoints', plot: bool = True):
    """
    Train behavior cloning model.

    Args:
        dataset_path: Path to dataset (.npz file)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_split: Validation split ratio
        hidden_dim: Hidden layer size
        device: Device ('cuda' or 'cpu', auto-detect if None)
        checkpoint_dir: Directory to save checkpoints
        plot: Whether to plot training curves

    Returns:
        trainer: Trained BCTrainer object
        history: Training history dict
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)

    observations = torch.from_numpy(data['observations']).float()
    masks = torch.from_numpy(data['masks']).bool()
    actions = torch.from_numpy(data['actions']).long()

    print(f"Dataset size: {len(observations)} transitions")
    print(f"Observation shape: {observations.shape}")
    print(f"Action space: {masks.shape[1]}")

    # Create dataset
    full_dataset = TensorDataset(observations, masks, actions)

    # Train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize network
    obs_dim = observations.shape[1]
    action_dim = masks.shape[1]

    policy = MaskedPolicyNetwork(obs_dim, action_dim, hidden_dim)
    trainer = BCTrainer(policy, learning_rate=learning_rate, device=device)

    print(f"\nModel architecture:")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }

    # Training loop
    print("\nTraining...")
    for epoch in range(num_epochs):
        # Training
        train_losses = []
        train_accs = []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_obs, batch_mask, batch_actions in train_bar:
            loss, acc = trainer.train_step(batch_obs, batch_mask, batch_actions)
            train_losses.append(loss)
            train_accs.append(acc)

            train_bar.set_postfix({
                'loss': f"{np.mean(train_losses):.4f}",
                'acc': f"{np.mean(train_accs):.3f}"
            })

        # Validation
        val_losses = []
        val_accs = []

        for batch_obs, batch_mask, batch_actions in val_loader:
            loss, acc = trainer.evaluate(batch_obs, batch_mask, batch_actions)
            val_losses.append(loss)
            val_accs.append(acc)

        # Record history
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accs)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accs)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.3f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.3f}")

        # Save best model
        if epoch_val_acc > history['best_val_acc']:
            history['best_val_acc'] = epoch_val_acc
            history['best_epoch'] = epoch + 1
            best_path = f"{checkpoint_dir}/best_model.pt"
            trainer.save(best_path)
            print(f"  → Saved best model (val_acc: {epoch_val_acc:.3f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
            trainer.save(checkpoint_path)

    # Save final model
    final_path = f"{checkpoint_dir}/final_model.pt"
    trainer.save(final_path)
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {history['best_val_acc']:.3f} (epoch {history['best_epoch']})")

    # Plot training curves
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.axhline(y=history['best_val_acc'], color='r', linestyle='--',
                   label=f'Best Val Acc ({history["best_val_acc"]:.3f})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = f"{checkpoint_dir}/training_curves.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training curves to {plot_path}")
        plt.close()

    return trainer, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train behavior cloning model')
    parser.add_argument('--dataset', type=str, default='bc_data.npz',
                       help='Path to dataset (default: bc_data.npz)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden layer size (default: 512)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu, auto-detect if None)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Disable plotting')

    args = parser.parse_args()

    trainer, history = train_bc(
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        hidden_dim=args.hidden_dim,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        plot=not args.no_plot
    )
