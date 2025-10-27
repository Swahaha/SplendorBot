"""
Test script to validate code structure without requiring compiled game module.
"""

import sys
import ast
import os
from pathlib import Path


def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_all_files():
    """Test all Python files in the RL directory."""
    rl_dir = Path(__file__).parent

    files_to_test = [
        'env_wrapper.py',
        'heuristic_agent.py',
        'generate_dataset.py',
        'policy_network.py',
        'train_bc.py',
        'train_ppo.py',
        'evaluate.py',
        'main.py',
    ]

    print("=" * 80)
    print("TESTING CODE STRUCTURE")
    print("=" * 80)

    all_passed = True

    for filename in files_to_test:
        filepath = rl_dir / filename
        if not filepath.exists():
            print(f"[FAIL] {filename}: File not found")
            all_passed = False
            continue

        passed, message = check_file_syntax(filepath)
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {filename}: {message}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 80)

    if all_passed:
        print("[SUCCESS] All files have valid syntax!")
    else:
        print("[FAILED] Some files have errors")

    print("=" * 80)

    return all_passed


def check_structure():
    """Check directory structure."""
    rl_dir = Path(__file__).parent

    print("\n" + "=" * 80)
    print("DIRECTORY STRUCTURE")
    print("=" * 80)

    required_files = {
        'env_wrapper.py': 'Environment wrapper with observation and action space',
        'heuristic_agent.py': 'Heuristic agent implementing decision rules',
        'generate_dataset.py': 'Dataset generation for behavior cloning',
        'policy_network.py': 'Neural network with action masking',
        'train_bc.py': 'Behavior cloning training script',
        'train_ppo.py': 'PPO training script',
        'evaluate.py': 'Agent evaluation framework',
        'main.py': 'Main entry point for pipeline',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Documentation',
    }

    for filename, description in required_files.items():
        filepath = rl_dir / filename
        exists = filepath.exists()
        status = "[OK]" if exists else "[MISS]"
        print(f"{status} {filename:25s} - {description}")

    print("=" * 80)


def check_key_components():
    """Check that key components are properly defined."""
    print("\n" + "=" * 80)
    print("KEY COMPONENTS")
    print("=" * 80)

    checks = []

    # Check env_wrapper
    try:
        code = open('env_wrapper.py', 'r').read()
        checks.append(('SplendorEnv class', 'class SplendorEnv' in code))
        checks.append(('reset method', 'def reset(' in code))
        checks.append(('step method', 'def step(' in code))
        checks.append(('_get_observation', 'def _get_observation' in code))
        checks.append(('_get_action_mask', 'def _get_action_mask' in code))
    except:
        checks.append(('env_wrapper.py', False))

    # Check heuristic_agent
    try:
        code = open('heuristic_agent.py', 'r').read()
        checks.append(('HeuristicAgent class', 'class HeuristicAgent' in code))
        checks.append(('choose_action method', 'def choose_action(' in code))
        checks.append(('_check_winning_buy', 'def _check_winning_buy' in code))
        checks.append(('_best_affordable_buy', 'def _best_affordable_buy' in code))
        checks.append(('_strategic_reserve', 'def _strategic_reserve' in code))
        checks.append(('_best_take3', 'def _best_take3' in code))
        checks.append(('_best_take2', 'def _best_take2' in code))
    except:
        checks.append(('heuristic_agent.py', False))

    # Check policy_network
    try:
        code = open('policy_network.py', 'r').read()
        checks.append(('MaskedPolicyNetwork', 'class MaskedPolicyNetwork' in code))
        checks.append(('BCTrainer', 'class BCTrainer' in code))
        checks.append(('forward method', 'def forward(' in code))
        checks.append(('get_action method', 'def get_action(' in code))
        checks.append(('evaluate_actions', 'def evaluate_actions' in code))
    except:
        checks.append(('policy_network.py', False))

    # Check training scripts
    try:
        code = open('train_bc.py', 'r').read()
        checks.append(('train_bc function', 'def train_bc(' in code))
    except:
        checks.append(('train_bc.py', False))

    try:
        code = open('train_ppo.py', 'r').read()
        checks.append(('PPOTrainer class', 'class PPOTrainer' in code))
        checks.append(('compute_gae', 'def compute_gae(' in code))
        checks.append(('collect_trajectories', 'def collect_trajectories(' in code))
    except:
        checks.append(('train_ppo.py', False))

    # Check evaluation
    try:
        code = open('evaluate.py', 'r').read()
        checks.append(('PolicyAgent', 'class PolicyAgent' in code))
        checks.append(('RandomAgent', 'class RandomAgent' in code))
        checks.append(('evaluate_agents', 'def evaluate_agents(' in code))
    except:
        checks.append(('evaluate.py', False))

    for component, exists in checks:
        status = "[OK]" if exists else "[MISS]"
        print(f"{status} {component}")

    all_passed = all(exists for _, exists in checks)

    print("=" * 80)

    if all_passed:
        print("[SUCCESS] All key components are present!")
    else:
        print("[FAILED] Some components are missing")

    return all_passed


def print_summary():
    """Print summary of what's been created."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
The complete RL training pipeline has been created with the following components:

1. ENVIRONMENT WRAPPER (env_wrapper.py)
   - Fixed-size observation space (198 dimensions)
   - Discrete action space (45 actions)
   - Action masking for legal moves
   - Reward: prestige points gained per turn

2. HEURISTIC AGENT (heuristic_agent.py)
   - Winning buy detection
   - Strategic card buying (value maximization)
   - Intelligent card reservation
   - Optimized token collection
   - Noble progress tracking

3. DATASET GENERATION (generate_dataset.py)
   - Run heuristic agent for N games
   - Save observations, masks, and actions
   - Dataset verification utilities

4. POLICY NETWORK (policy_network.py)
   - MaskedPolicyNetwork: 2-layer MLP with LayerNorm
   - Action masking in forward pass
   - Policy + value heads for PPO
   - BCTrainer for behavior cloning

5. TRAINING SCRIPTS
   - train_bc.py: Behavior cloning from heuristic
   - train_ppo.py: PPO reinforcement learning

6. EVALUATION (evaluate.py)
   - Head-to-head agent matchups
   - Win rate and performance statistics
   - Support for multiple agent types

7. DOCUMENTATION
   - README.md: Comprehensive guide
   - requirements.txt: Dependencies

USAGE:
------
To use this pipeline, you need to:

1. Compile the Splendor game module for your platform:
   - Install CMake and a C++ compiler
   - Run: mkdir -p build && cd build && cmake .. && make
   - This creates splendor_game module

2. Install Python dependencies:
   pip install -r requirements.txt

3. Run the pipeline:
   python main.py --mode full

ALTERNATIVELY (if you can't compile):
-------------------------------------
You can run the pipeline on a system where the game is already compiled
(like macOS with the existing .so file), or compile it for Windows.

For Windows compilation, you need:
- Visual Studio with C++ tools
- CMake
- Python development headers

Then run: cmake -G "Visual Studio 17 2022" .. && cmake --build . --config Release
""")

    print("=" * 80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)

    # Run tests
    syntax_ok = test_all_files()
    check_structure()
    components_ok = check_key_components()
    print_summary()

    print("\n" + "=" * 80)
    if syntax_ok and components_ok:
        print("[SUCCESS] CODE VALIDATION SUCCESSFUL")
        print("\nAll code is syntactically correct and components are in place.")
        print("Ready to use once the game module is compiled for your platform.")
    else:
        print("[FAILED] VALIDATION FAILED")
        print("\nSome issues were found. Please check the output above.")
    print("=" * 80)
