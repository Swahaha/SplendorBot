# Quick Reference Card

## One-Line Commands

```bash
# Complete pipeline (after compiling game module)
python main.py --mode full

# Generate dataset
python generate_dataset.py --num_games 1000

# Train BC
python train_bc.py --dataset bc_data.npz --epochs 50

# Evaluate
python evaluate.py --num_games 100 --agent0 policy --agent1 heuristic

# Train PPO
python train_ppo.py --iterations 1000 --pretrained checkpoints/best_model.pt

# Validate code
python test_structure.py
```

## File Purposes

| File | Purpose |
|------|---------|
| `env_wrapper.py` | Environment with obs/action/mask interface |
| `heuristic_agent.py` | Strategic rule-based bot |
| `generate_dataset.py` | Create BC training data |
| `policy_network.py` | Neural network with masking |
| `train_bc.py` | Behavior cloning training |
| `train_ppo.py` | PPO reinforcement learning |
| `evaluate.py` | Test and compare agents |
| `main.py` | Complete pipeline |

## Key Numbers

- **Observation dim**: 198
- **Action dim**: 45
- **Hidden dim**: 512 (default)
- **Dataset size**: ~80k transitions (1000 games)
- **Training time BC**: 5-10 min (CPU)
- **Training time PPO**: 30-60 min (CPU)

## Agent Types

```python
# Heuristic (rule-based)
agent = HeuristicAgent(env)

# Random (baseline)
agent = RandomAgent()

# Policy (neural network)
agent = load_policy_agent('checkpoints/best_model.pt', obs_dim, action_dim)
```

## Heuristic Priority

1. Winning buy (15+ points)
2. Best affordable buy (2*pts + 0.8*nobles + 0.5*discount)
3. Strategic reserve (≤2 missing tokens)
4. Take 3 distinct (maximize needs)
5. Take 2 same (if helps targets)
6. Fallback (first legal)

## Action Space

- 0-11: Reserve from market
- 12-14: Reserve from deck (not implemented)
- 15-26: Buy from market
- 27-29: Buy from reserve
- 30-39: Take 3 different
- 40-44: Take 2 same

## Common Issues

| Problem | Solution |
|---------|----------|
| Module not found | Compile game module |
| Slow generation | Reduce --num_games |
| BC not learning | Increase dataset size |
| PPO not improving | Use --pretrained BC model |
| Invalid actions | Check masking (should never happen) |

## Typical Workflow

```bash
# 1. Setup
pip install -r requirements.txt
# Compile game module (see USAGE_GUIDE.md)

# 2. Generate data
python generate_dataset.py --num_games 1000

# 3. Train BC
python train_bc.py --dataset bc_data.npz

# 4. Evaluate
python evaluate.py --agent0 policy --agent1 heuristic

# 5. (Optional) Train PPO
python train_ppo.py --pretrained checkpoints/best_model.pt

# 6. Final eval
python evaluate.py --agent0 policy --agent1 heuristic \
  --policy_checkpoint ppo_checkpoints/final_model.pt
```

## Performance Targets

| Agent | vs Random | vs Heuristic |
|-------|-----------|--------------|
| Heuristic | 90% | 50% |
| BC Policy | 85% | 45-55% |
| PPO Policy | 90% | 50-60% |

## Hyperparameters

### BC Training
```python
epochs = 50
batch_size = 256
learning_rate = 3e-4
hidden_dim = 512
val_split = 0.1
```

### PPO Training
```python
iterations = 1000
episodes_per_iter = 10
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
```

## File Locations

- **Datasets**: `bc_data.npz`
- **BC models**: `checkpoints/best_model.pt`
- **PPO models**: `ppo_checkpoints/final_model.pt`
- **Plots**: `checkpoints/training_curves.png`

## Debugging

```python
# Check observation
obs, mask, _ = env.reset()
print(f"Obs shape: {obs.shape}")  # Should be (198,)
print(f"Legal actions: {mask.sum()}")  # Should be > 0

# Check dataset
python generate_dataset.py --verify_only bc_data.npz

# Test heuristic
python main.py --mode test
```

## Extending

### Custom reward
Edit `env_wrapper.py:150`:
```python
reward += 0.1 * bonus_for_cards
```

### Better network
Edit `policy_network.py:16`:
```python
hidden_dim = 1024  # Larger network
```

### More heuristics
Edit `heuristic_agent.py`:
```python
def _my_custom_rule(self, ...):
    # Your logic
    return action_id
```

---

**For full details, see:**
- `USAGE_GUIDE.md` - Complete usage instructions
- `README.md` - Technical documentation
- `SUMMARY.md` - Implementation overview
