"""Simple test to isolate environment issue"""
import sys
import os

# Set up paths
print("Setting up paths...")
rl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, rl_dir)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
print(f"Changed to: {os.getcwd()}")

# Add paths for module
python_path = os.path.join(project_root, 'python')
build_path = os.path.join(project_root, 'build', 'Release')
sys.path.insert(0, python_path)
sys.path.insert(0, build_path)

# Import game module
print("Importing splendor_game...")
try:
    import splendor_game as sg
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    print(f"Python path: {sys.path[:5]}")
    sys.exit(1)

# Create game
print("Creating game...")
try:
    game = sg.SplendorGame(2, 42)
    print("Game created!")
except Exception as e:
    print(f"Game creation failed: {e}")
    sys.exit(1)

# Get state
print("Getting state...")
try:
    state = game.state_summary()
    print(f"State retrieved! Current player: {state['current_player']}")
    print(f"Bank: {state['bank']}")
except Exception as e:
    print(f"State retrieval failed: {e}")
    sys.exit(1)

# Get legal moves
print("Getting legal moves...")
try:
    moves = game.legal_moves()
    print(f"Got {len(moves)} legal moves")
except Exception as e:
    print(f"Legal moves failed: {e}")
    sys.exit(1)

print("\nAll tests passed!")
