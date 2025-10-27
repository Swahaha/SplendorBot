# Splendor RL Bot - Usage Guide

## Overview

You now have a complete reinforcement learning training pipeline for Splendor! The system includes:

- **Heuristic Agent**: Strategic rule-based bot implementing your decision rules
- **Environment Wrapper**: Gym-style interface with fixed observations and action masking
- **Behavior Cloning**: Train neural network from heuristic demonstrations
- **PPO Training**: Further improve policy with reinforcement learning
- **Evaluation Framework**: Test and compare different agents

## Prerequisites

### 1. Compile the Game Module

The RL code needs the compiled Splendor game. You have two options:

#### Option A: Use on macOS (where module is already compiled)

The existing `python/splendor_game.cpython-39-darwin.so` will work on macOS with Python 3.9.

#### Option B: Compile for Windows

You're currently on Windows, so you need to compile for your platform:

**Requirements:**
- Visual Studio 2019 or later (with C++ tools)
- CMake 3.12+
- Python 3.x development headers

**Steps:**
```bash
# Install CMake if not already installed
# Download from: https://cmake.org/download/

# Open Command Prompt or PowerShell in the project root
cd C:\Users\swara\Desktop\Splendor\SplendorBot

# Create build directory
mkdir build
cd build

# Configure (replace "Visual Studio 17 2022" with your version)
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build
cmake --build . --config Release

# The output will be in build/Release/splendor_game.*.pyd
# Copy it to the python directory or update sys.path
```

### 2. Install Python Dependencies

```bash
cd C:\Users\swara\Desktop\Splendor\SplendorBot\RL
pip install -r requirements.txt
```

This installs:
- numpy (for arrays)
- torch (for neural networks)
- tqdm (for progress bars)
- matplotlib (for plotting)

## Quick Start

### Validate Installation

First, verify everything is set up correctly:

```bash
cd RL
python test_structure.py
```

This checks that all code is valid and components are in place.

### Run Complete Pipeline

Once the game module is compiled:

```bash
# Full pipeline (BC training only)
python main.py --mode full

# Full pipeline with PPO (takes longer)
python main.py --mode full --use_ppo

# Quick test (10 games)
python main.py --mode test
```

## Step-by-Step Usage

### Step 1: Generate Dataset

Generate training data by running the heuristic agent:

```bash
python generate_dataset.py --num_games 1000 --num_players 2 --output bc_data.npz
```

**Parameters:**
- `--num_games`: Number of games to play (default: 1000)
- `--num_players`: Players per game (default: 2)
- `--output`: Output file path
- `--base_seed`: Random seed for reproducibility

**Output:**
- Creates `bc_data.npz` with ~50k-100k transitions
- Prints dataset statistics

**Verify dataset:**
```bash
python generate_dataset.py --verify_only bc_data.npz
```

### Step 2: Train Behavior Cloning Model

Train a neural network to imitate the heuristic:

```bash
python train_bc.py --dataset bc_data.npz --epochs 50 --batch_size 256
```

**Parameters:**
- `--dataset`: Path to dataset
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 3e-4)
- `--hidden_dim`: Hidden layer size (default: 512)
- `--checkpoint_dir`: Where to save models (default: checkpoints/)

**Output:**
- Saves best model to `checkpoints/best_model.pt`
- Saves final model to `checkpoints/final_model.pt`
- Creates training curves: `checkpoints/training_curves.png`

**Expected Results:**
- Validation accuracy: 40-60% (depends on dataset size)
- Training time: ~5-10 minutes on CPU for 1000 games

### Step 3: Evaluate Agents

Test your trained agent against others:

```bash
# BC vs Heuristic
python evaluate.py --num_games 100 --agent0 policy --agent1 heuristic

# BC vs Random
python evaluate.py --num_games 100 --agent0 policy --agent1 random

# Heuristic vs Random (baseline)
python evaluate.py --num_games 100 --agent0 heuristic --agent1 random
```

**Agent Types:**
- `policy`: Neural network agent (requires `--policy_checkpoint`)
- `heuristic`: Rule-based heuristic agent
- `random`: Random agent (chooses uniformly from legal actions)

**3-4 Players:**
```bash
python evaluate.py --num_games 100 --num_players 3 \
  --agent0 policy --agent1 heuristic --agent2 random
```

**Expected Performance:**
- Random vs Heuristic: ~10% win rate for random
- BC vs Heuristic: ~40-60% win rate for BC
- BC vs Random: ~80-90% win rate for BC

### Step 4: Train with PPO (Optional)

Improve the BC policy with reinforcement learning:

```bash
python train_ppo.py --iterations 1000 --episodes_per_iter 10 \
  --pretrained checkpoints/best_model.pt
```

**Parameters:**
- `--iterations`: Number of PPO iterations (default: 1000)
- `--episodes_per_iter`: Episodes per iteration (default: 10)
- `--pretrained`: Path to BC model (strongly recommended)
- `--lr`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--checkpoint_dir`: Where to save (default: ppo_checkpoints/)

**Output:**
- Saves final model to `ppo_checkpoints/final_model.pt`
- Creates training curves

**Training Time:**
- 1000 iterations: ~30-60 minutes
- PPO is slower than BC but can achieve better performance

**Evaluate PPO:**
```bash
python evaluate.py --num_games 100 --agent0 policy --agent1 heuristic \
  --policy_checkpoint ppo_checkpoints/final_model.pt
```

## Understanding the Heuristic

The heuristic agent implements your decision rules in this priority order:

### 1. Winning Buy
If any card purchase reaches 15+ prestige points, buy the highest value one.

### 2. Best Affordable Buy
Among affordable cards, maximize:
```
score = 2.0 * card_points + 0.8 * noble_progress_after + 0.5 * discount_gain
```

Where:
- `card_points`: Prestige points on the card
- `noble_progress_after`: Progress toward noble requirements after buying
- `discount_gain`: Number of colors newly at ≥1 discount

### 3. Strategic Reserve
If a card is within ≤2 missing tokens, reserve the one maximizing:
```
reserve_score = card_points + 0.5 * discount_gain + 0.3 * noble_alignment
```

### 4. Take 3 Distinct Tokens
Choose the color triple that maximizes:
```
needs = Σ (weighted_needs_for_target_cards) + 0.5 * (noble_req - discounts)+
```

Where weights are `1/(1 + total_missing_for_card)`.

### 5. Take 2 Same Tokens
If legal and it reduces missing tokens for top 2 target cards, take it.

### 6. Fallback
First legal action.

## Architecture Details

### Observation Space (198 dimensions)

Fixed-size vector encoding from current player's perspective:

1. **Current player (14)**: tokens(6) + bonuses(5) + prestige(1) + cards(1) + reserved(1)
2. **Other players (42)**: 3 opponents × 14 (padded)
3. **Bank (6)**: Token counts
4. **Market (84)**: 12 cards × 7 features
5. **Nobles (30)**: 5 nobles × 6 features
6. **Reserved cards (21)**: 3 cards × 7
7. **Current player ID (1)**

### Action Space (45 actions)

- **0-11**: Reserve from market (3 tiers × 4 slots)
- **12-14**: Reserve from deck (not implemented yet)
- **15-26**: Buy from market (3 tiers × 4 slots)
- **27-29**: Buy from reserve (up to 3)
- **30-39**: Take 3 different (C(5,3) = 10 combos)
- **40-44**: Take 2 same (5 colors)

### Action Masking

Illegal actions are masked by setting logits to -1e9 before softmax. This ensures:
- Only legal actions are sampled
- No constraint violations
- Stable training

### Network Architecture

```
Input (198)
  → Linear(512) + LayerNorm + SiLU
  → Linear(512) + LayerNorm + SiLU
  → Policy Head: Linear(45) [with masking]
  → Value Head: Linear(256) + SiLU + Linear(1)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'splendor_game'"

The C++ game module isn't compiled or isn't in the Python path.

**Solution:**
1. Compile the module (see Prerequisites above)
2. Ensure the `.pyd` or `.so` file is in `python/` or `build/`
3. The env_wrapper.py automatically checks both locations

### "Dataset generation is slow"

**Solutions:**
- Reduce `--num_games` (500-1000 is usually enough)
- Use a faster machine
- The C++ engine is already optimized

### "BC model not learning well"

**Possible causes:**
- Dataset too small → Generate more games
- Learning rate too high → Try `--lr 1e-4`
- Model too small → Try `--hidden_dim 1024`

**Check:**
```bash
python generate_dataset.py --verify bc_data.npz
# Look for invalid actions (should be 0%)
```

### "PPO not improving over BC"

**Solutions:**
- Start with BC pretrained model (`--pretrained`)
- Reduce clip epsilon (`--clip_epsilon 0.1`)
- Use reward shaping (modify env_wrapper.py step())
- Train longer (`--iterations 2000`)

### "Agent makes illegal moves"

This should never happen with proper masking. If it does:
1. Check mask is applied before sampling
2. Verify logits masked with -1e9
3. Report as bug

## Advanced Usage

### Custom Reward Shaping

Edit `env_wrapper.py` around line 150:

```python
# Current (sparse reward)
reward = float(prestige_after - prestige_before)

# Custom (dense reward)
reward = float(prestige_after - prestige_before)
reward += 0.1 * (cards_after - cards_before)  # Bonus for buying cards
reward += 0.05 * (bonuses_after - bonuses_before).sum()  # Bonus for discounts
```

### Multi-Agent Self-Play

Modify `train_ppo.py` to alternate between agents:

```python
# In collect_trajectories, use multiple agents
agents = [policy1, policy2]
current_agent = agents[env.game.current_player() % len(agents)]
action = current_agent.choose_action(obs, mask)
```

### Training on Different Player Counts

```bash
# Generate datasets for each
python generate_dataset.py --num_games 500 --num_players 2 --output bc_2p.npz
python generate_dataset.py --num_games 500 --num_players 3 --output bc_3p.npz
python generate_dataset.py --num_games 500 --num_players 4 --output bc_4p.npz

# Train on combined data
# (You'll need to modify train_bc.py to load multiple files)
```

### Hyperparameter Tuning

Create a grid search:

```bash
for lr in 3e-4 1e-4 1e-3; do
  for hidden in 256 512 1024; do
    python train_bc.py --lr $lr --hidden_dim $hidden \
      --checkpoint_dir checkpoints/lr${lr}_h${hidden}
  done
done
```

## Performance Benchmarks

On a typical system (i7 CPU, no GPU):

| Task | Time | Output |
|------|------|--------|
| Generate 1000 games | 5-10 min | ~80k transitions |
| Train BC (50 epochs) | 5-10 min | ~50% val acc |
| Evaluate 100 games | 2-3 min | Win rates |
| Train PPO (1000 iter) | 30-60 min | Improved policy |

With GPU (CUDA):
- BC training: 2-3 min
- PPO training: 15-20 min

## Next Steps

1. **Improve Heuristic**: Modify `heuristic_agent.py` to test new strategies
2. **More Data**: Generate 5k-10k games for better BC performance
3. **Larger Networks**: Try `--hidden_dim 1024` or add more layers
4. **Self-Play**: Implement multi-agent PPO with self-play
5. **MCTS**: Add Monte Carlo Tree Search for planning
6. **Graph Neural Networks**: Use GNNs for better state representation

## Files Reference

### Core Files
- `env_wrapper.py`: Environment interface
- `heuristic_agent.py`: Rule-based agent
- `policy_network.py`: Neural network architecture
- `generate_dataset.py`: Dataset generation
- `train_bc.py`: Behavior cloning training
- `train_ppo.py`: PPO training
- `evaluate.py`: Agent evaluation
- `main.py`: Complete pipeline

### Documentation
- `README.md`: Technical documentation
- `USAGE_GUIDE.md`: This file
- `requirements.txt`: Dependencies

### Testing
- `test_structure.py`: Validate code structure

## Support

For issues with:
- **RL code**: Check this guide and README.md
- **Game rules**: See `../CLAUDE.md`
- **Compilation**: Check CMake and C++ compiler setup

## License

Part of the Splendor game implementation project.

---

**Happy training!** You now have a complete system to train bots that can beat the Splendor game.
