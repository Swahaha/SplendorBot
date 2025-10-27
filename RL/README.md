# Splendor RL Training Pipeline

This directory contains a complete reinforcement learning training pipeline for Splendor, including:
- Environment wrapper with fixed-size observations and action masking
- Heuristic agent implementing strategic decision rules
- Dataset generation for behavior cloning
- Masked policy network with BC and PPO training
- Evaluation framework for comparing agents

## Quick Start

### 1. Generate Dataset (Behavior Cloning)

Generate supervised learning data by running the heuristic agent:

```bash
python generate_dataset.py --num_games 1000 --num_players 2 --output bc_data.npz
```

This will create `bc_data.npz` containing ~50k-100k state-action pairs.

### 2. Train BC Policy

Train a neural network policy from the heuristic demonstrations:

```bash
python train_bc.py --dataset bc_data.npz --epochs 50 --batch_size 256
```

This saves the best model to `checkpoints/best_model.pt`.

### 3. Evaluate Agents

Test different agents against each other:

```bash
# Heuristic vs Random
python evaluate.py --num_games 100 --agent0 heuristic --agent1 random

# BC Policy vs Heuristic
python evaluate.py --num_games 100 --agent0 policy --agent1 heuristic --policy_checkpoint checkpoints/best_model.pt

# BC Policy vs Random (3 players)
python evaluate.py --num_games 100 --num_players 3 --agent0 policy --agent1 random --agent2 random
```

### 4. Train with PPO (Optional)

Further improve the BC policy with reinforcement learning:

```bash
python train_ppo.py --iterations 1000 --episodes_per_iter 10 --pretrained checkpoints/best_model.pt
```

This finetunes the BC policy using PPO and saves to `ppo_checkpoints/`.

## Files Overview

### Core Components

- **`env_wrapper.py`**: Gym-style environment wrapper
  - Fixed-size observation vectors (198 dim)
  - Action space of 45 discrete actions
  - Action masking for legal moves
  - Step-based interface with rewards

- **`heuristic_agent.py`**: Rule-based heuristic agent
  - Winning buy detection
  - Strategic card buying (maximize value score)
  - Intelligent reserving (target cards within reach)
  - Token collection optimization
  - Noble progress tracking

### Training Scripts

- **`generate_dataset.py`**: Dataset generation
  - Runs heuristic agent for N games
  - Saves observations, masks, and actions
  - Produces `.npz` file for BC training
  - Includes verification utilities

- **`train_bc.py`**: Behavior cloning training
  - Masked cross-entropy loss
  - 2-layer MLP with LayerNorm
  - Train/val split with early stopping
  - Saves checkpoints and plots

- **`train_ppo.py`**: PPO reinforcement learning
  - Proximal Policy Optimization
  - GAE advantage estimation
  - Clipped surrogate objective
  - Can initialize from BC policy

### Evaluation

- **`evaluate.py`**: Agent evaluation
  - Head-to-head matchups
  - Win rate and prestige statistics
  - Support for 2-4 players
  - Agents: heuristic, random, policy

- **`policy_network.py`**: Neural network architecture
  - `MaskedPolicyNetwork`: Policy + value heads
  - Action masking in forward pass
  - `BCTrainer`: BC training logic
  - `PPOTrainer`: PPO training logic (in train_ppo.py)

## Architecture Details

### Observation Space (198 dimensions)

The observation is a fixed-size vector encoding the game state from the current player's perspective:

1. **Current player (14)**: tokens(6) + bonuses(5) + prestige(1) + cards(1) + reserved(1)
2. **Other players (42)**: 3 opponents × 14 features (padded for <4 players)
3. **Bank (6)**: Token counts for 6 colors
4. **Market (84)**: 12 cards × 7 features (tier, color, points, 5 costs)
5. **Nobles (30)**: 5 nobles × 6 features (points, 5 requirements)
6. **Reserved cards (21)**: 3 cards × 7 features
7. **Current player ID (1)**: 0-3

### Action Space (45 actions)

Actions are mapped to integer IDs:

- **0-11**: Reserve card from market (3 tiers × 4 slots)
- **12-14**: Reserve from deck top (3 tiers) - *not yet implemented in game*
- **15-26**: Buy card from market (3 tiers × 4 slots)
- **27-29**: Buy card from reserve (up to 3 reserved)
- **30-39**: Take 3 different tokens (C(5,3) = 10 combinations)
- **40-44**: Take 2 same tokens (5 colors)

### Action Masking

Illegal actions are masked out by setting logits to -1e9 before softmax. This ensures:
- Policy only samples legal actions
- No wasted probability mass on invalid moves
- Stable training without constraint violations

### Heuristic Decision Rules

The heuristic agent follows this priority order:

1. **Winning buy**: If any purchase reaches 15+ points, take highest
2. **Best affordable buy**: Maximize `2.0*points + 0.8*noble_progress + 0.5*discount_gain`
3. **Strategic reserve**: Reserve cards within ≤2 missing tokens, maximize reserve score
4. **Take 3 distinct**: Choose triple maximizing weighted token needs
5. **Take 2 same**: If it helps top 2 target cards
6. **Fallback**: First legal action

## Training Pipeline

### Recommended Workflow

```bash
# 1. Generate dataset (1000 games ~ 5-10 minutes)
python generate_dataset.py --num_games 1000 --output bc_data.npz

# 2. Verify dataset
python generate_dataset.py --verify_only bc_data.npz

# 3. Train BC model (50 epochs ~ 5-10 minutes on CPU)
python train_bc.py --dataset bc_data.npz --epochs 50 --batch_size 256

# 4. Evaluate BC vs Heuristic
python evaluate.py --num_games 100 --agent0 policy --agent1 heuristic

# 5. (Optional) Finetune with PPO
python train_ppo.py --iterations 500 --pretrained checkpoints/best_model.pt

# 6. Evaluate PPO vs BC
python evaluate.py --num_games 100 --agent0 policy --agent1 policy \
  --policy_checkpoint ppo_checkpoints/final_model.pt
```

### Expected Performance

- **Random agent**: ~5-10% win rate vs heuristic
- **BC policy**: ~40-60% win rate vs heuristic (depends on dataset size)
- **PPO policy**: ~50-70% win rate vs heuristic (with enough training)
- **Heuristic baseline**: ~70-80% win rate vs random

## Hyperparameters

### Behavior Cloning

```python
epochs = 50
batch_size = 256
learning_rate = 3e-4
hidden_dim = 512
val_split = 0.1
```

### PPO

```python
gamma = 0.99              # Discount factor
gae_lambda = 0.95         # GAE lambda
clip_epsilon = 0.2        # PPO clip parameter
value_coef = 0.5          # Value loss coefficient
entropy_coef = 0.01       # Entropy bonus
learning_rate = 3e-4
episodes_per_iter = 10
```

## Extending the System

### Adding New Agents

```python
class MyAgent:
    def choose_action(self, obs: np.ndarray, mask: np.ndarray, state: Dict = None) -> int:
        # obs: observation vector
        # mask: boolean array of legal actions
        # state: optional game state dict
        legal_actions = np.where(mask)[0]
        # Your logic here
        return selected_action
```

### Custom Reward Shaping

Modify `env_wrapper.py::step()` to add shaped rewards:

```python
# Current: reward = prestige gained
reward = float(prestige_after - prestige_before)

# Custom: add bonuses for card purchases, noble progress, etc.
reward += 0.1 * cards_gained
reward += 0.5 * noble_progress
```

### Potential-Based Shaping

For valid potential-based reward shaping:

```python
def potential(state):
    player = state['players'][current_player]
    return player['prestige_points'] + 0.1 * len(player['played'])

shaped_reward = reward + gamma * potential(next_state) - potential(state)
```

## Troubleshooting

### Dataset Generation is Slow

- Reduce `--num_games` (500-1000 is usually sufficient)
- Use faster hardware
- The C++ game engine is already quite fast

### BC Training Not Converging

- Increase dataset size (2000+ games)
- Reduce learning rate
- Increase model capacity (`--hidden_dim 1024`)
- Check that actions are valid: `python generate_dataset.py --verify bc_data.npz`

### Policy Makes Illegal Moves

This should never happen with proper masking. Check:
- Action mask is correctly applied before sampling
- Mask is boolean tensor with correct shape
- Logits are masked with `-1e9` for illegal actions

### PPO Not Improving

- Start with BC pretrained model
- Reduce clip epsilon (0.1-0.15)
- Increase episodes per iteration
- Check KL divergence (should be < 0.02)
- Use reward shaping to make signal stronger

## Performance Tips

### For Faster Training

1. **Use GPU**: Add `--device cuda` to training scripts
2. **Increase batch size**: `--batch_size 512` (if memory allows)
3. **Reduce dataset size**: 500 games is often enough for BC
4. **Parallel data collection**: Modify `collect_trajectories` to use multiprocessing

### For Better Performance

1. **More diverse data**: Train on 2p, 3p, and 4p games
2. **Larger networks**: `--hidden_dim 1024`
3. **Self-play**: Use PPO with self-play for multi-agent scenarios
4. **Curriculum learning**: Start with simpler scenarios, gradually increase difficulty

## Known Limitations

1. **Noble acquisition not implemented**: Game engine skips noble checks
2. **Reserve from deck**: Not yet supported in move generator
3. **Token return overflow**: Not handled (moves that exceed limit are masked)
4. **Single-agent RL**: PPO currently trains single agent, not multi-agent

## Future Enhancements

- [ ] Multi-agent PPO with self-play
- [ ] MCTS-based planning agents
- [ ] AlphaZero-style training
- [ ] State representation improvements (graph neural networks)
- [ ] Opponent modeling
- [ ] Transfer learning across player counts
- [ ] Distributed training

## References

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **Behavior Cloning**: [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)

---

For questions or issues, please refer to the main project documentation in `../CLAUDE.md`.
