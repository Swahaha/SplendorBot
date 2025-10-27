"""
Debug script to find game states where mask is empty but should have legal moves.
"""

import sys
sys.path.append('../python')

import numpy as np
from env_wrapper import SplendorEnv
import json

def debug_action_mask(env, state):
    """Debug why action mask might be empty."""

    print("\n" + "="*80)
    print("DEBUGGING ACTION MASK")
    print("="*80)

    # Get legal moves from game
    legal_moves = env.game.legal_moves()
    print(f"\nGame reports {len(legal_moves)} legal moves:")

    for i, move in enumerate(legal_moves[:10]):  # Show first 10
        action_type, payload = move
        print(f"\n  Move {i}:")
        print(f"    Type: {action_type}")
        print(f"    Payload: {payload}")

        # Try to convert to action ID
        try:
            action_ids = env._game_move_to_action_ids(action_type, payload, state)
            print(f"    Action IDs: {action_ids}")

            # Check if these are in the mask
            mask = env._get_action_mask()
            for aid in action_ids:
                if aid is not None:
                    print(f"      ID {aid}: {'MASKED' if not mask[aid] else 'OK'}")
        except Exception as e:
            print(f"    ERROR converting: {e}")
            import traceback
            traceback.print_exc()

    # Show current game state summary
    print("\n" + "-"*80)
    print("GAME STATE:")
    print(f"  Current player: {state['current_player']}")
    print(f"  Bank: {state['bank']}")

    current_player = state['current_player']
    player = state['players'][current_player]
    print(f"\n  Player {current_player}:")
    print(f"    Tokens: {player['tokens']}")
    print(f"    Prestige: {player['prestige_points']}")
    print(f"    Cards: {len(player['played'])}")
    print(f"    Reserved: {len(player['reserved'])}")

    print("\n  Market:")
    for tier_idx, tier in enumerate(state['market']):
        valid_cards = [c for c in tier if c['valid']]
        print(f"    Tier {tier_idx + 1}: {len(valid_cards)} cards")

    print("\n" + "="*80)


def find_empty_mask_state():
    """Run games until we find a state with empty mask."""

    env = SplendorEnv(num_players=2)

    # Import heuristic agent
    from heuristic_agent import HeuristicAgent
    agent = HeuristicAgent(env)

    print("Searching for problematic game states...")
    print("(Using heuristic agent)")
    print("(Running up to 100 games, checking each step)\n")

    for game_num in range(100):
        print(f"Game {game_num + 1}...", end=" ")

        obs, mask, info = env.reset(seed=game_num)
        done = False
        step = 0

        while not done and step < 200:
            step += 1

            # Check mask
            legal_count = mask.sum()
            game_legal_count = len(env.game.legal_moves())

            # Found a mismatch!
            if legal_count == 0 and game_legal_count > 0:
                print(f"\n\nFOUND PROBLEM at step {step}!")
                print(f"  Mask says: {legal_count} legal actions")
                print(f"  Game says: {game_legal_count} legal moves")

                state = env.game.state_summary()

                # Save the state
                with open('problematic_state.json', 'w') as f:
                    json.dump(state, f, indent=2)
                print(f"\nState saved to: problematic_state.json")

                # Debug it
                debug_action_mask(env, state)

                return game_num, step, state

            # Also check for mismatches where counts differ
            if legal_count != game_legal_count:
                print(f"\n\nFOUND MISMATCH at step {step}!")
                print(f"  Mask says: {legal_count} legal actions")
                print(f"  Game says: {game_legal_count} legal moves")

                state = env.game.state_summary()
                debug_action_mask(env, state)

                return game_num, step, state

            # Take action if possible
            if legal_count > 0:
                state = env.game.state_summary()
                action = agent.choose_action(obs, mask, state)  # Use heuristic
                obs, reward, done, mask, info = env.step(action)
            else:
                # No legal moves
                if game_legal_count > 0:
                    # This is the problem!
                    print(f"\n\nFOUND PROBLEM at step {step}!")
                    state = env.game.state_summary()
                    debug_action_mask(env, state)
                    return game_num, step, state
                else:
                    # Both agree - no legal moves
                    done = True

        print(f"OK ({step} steps)")

    print("\nNo problems found in 100 games!")
    return None


def test_specific_moves():
    """Test conversion of specific move types."""

    print("\n" + "="*80)
    print("TESTING MOVE CONVERSION")
    print("="*80)

    env = SplendorEnv(num_players=2)
    obs, mask, info = env.reset(seed=42)
    state = env.game.state_summary()
    legal_moves = env.game.legal_moves()

    # Group moves by type
    move_types = {}
    for move in legal_moves:
        action_type = move[0]
        if action_type not in move_types:
            move_types[action_type] = []
        move_types[action_type].append(move)

    print(f"\nMove types found:")
    for action_type, moves in move_types.items():
        type_names = {
            0: "RESERVE_CARD",
            1: "BUY_CARD_FROM_MARKET",
            2: "BUY_CARD_FROM_RESERVE",
            3: "GET_3_FROM_BANK",
            4: "GET_2_FROM_BANK"
        }
        print(f"  Type {action_type} ({type_names.get(action_type, 'UNKNOWN')}): {len(moves)} moves")

        # Test conversion of first move of each type
        if moves:
            move = moves[0]
            action_type, payload = move
            print(f"    Testing: {payload}")

            try:
                action_ids = env._game_move_to_action_ids(action_type, payload, state)
                print(f"    -> Action IDs: {action_ids}")

                # Check mask
                for aid in action_ids:
                    if aid is not None and 0 <= aid < len(mask):
                        print(f"       ID {aid}: mask[{aid}] = {mask[aid]}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--find', action='store_true', help='Find problematic state')
    parser.add_argument('--test-conversion', action='store_true', help='Test move conversion')
    parser.add_argument('--both', action='store_true', help='Run both tests')

    args = parser.parse_args()

    if args.both or (not args.find and not args.test_conversion):
        # Default: run both
        test_specific_moves()
        result = find_empty_mask_state()

        if result is None:
            print("\n✓ All tests passed! No issues found.")
    elif args.test_conversion:
        test_specific_moves()
    elif args.find:
        find_empty_mask_state()
