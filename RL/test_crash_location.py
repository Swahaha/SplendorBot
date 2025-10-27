"""Narrow down where the crash happens"""
import sys
import os

# Set up paths
rl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, rl_dir)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)

python_path = os.path.join(project_root, 'python')
build_path = os.path.join(project_root, 'build', 'Release')
sys.path.insert(0, python_path)
sys.path.insert(0, build_path)

print("Test 1: Import splendor_game...")
import splendor_game as sg
print("  OK")

print("\nTest 2: Create game...")
game = sg.SplendorGame(2, 42)
print("  OK")

print("\nTest 3: Get state_summary...")
state = game.state_summary()
print(f"  OK - current_player: {state['current_player']}")

print("\nTest 4: Get legal_moves...")
moves = game.legal_moves()
print(f"  OK - {len(moves)} moves")

print("\nTest 5: Get first move details...")
if moves:
    first_move = moves[0]
    print(f"  OK - first move: {first_move}")

print("\nTest 6: Access state['players']...")
players = state['players']
print(f"  OK - {len(players)} players")

print("\nTest 7: Access player[0]...")
player0 = players[0]
print(f"  OK - player0 keys: {list(player0.keys())}")

print("\nTest 8: Access player['reserved']...")
reserved = player0['reserved']
print(f"  OK - reserved: {len(reserved)} cards")

print("\nTest 9: Access player['played']...")
played = player0['played']
print(f"  OK - played: {len(played)} cards")

print("\nTest 10: Check if played cards are valid...")
for i, card in enumerate(played):
    print(f"  Card {i}: valid={card.get('valid', '?')}, bonus_color={card.get('bonus_color', '?')}")
print("  OK")

print("\nTest 11: Access state['nobles']...")
nobles = state['nobles']
print(f"  OK - {len(nobles)} nobles")

print("\nTest 12: Access noble details...")
if nobles:
    noble0 = nobles[0]
    print(f"  OK - noble0 keys: {list(noble0.keys())}")
    print(f"  Noble 0 requirements: {noble0.get('requirements', '?')}")

print("\nAll tests passed!")
