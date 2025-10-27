"""
Script to recreate and analyze the "no legal moves" bug in the C++ game.

This script:
1. Recreates the exact game state where the bug occurs
2. Shows what legal moves SHOULD be available
3. Explains why the C++ move generator is failing
"""

import sys
sys.path.append('../python')

import numpy as np
from env_wrapper import SplendorEnv


def recreate_bug():
    """Recreate the exact sequence that leads to the bug."""

    print("="*80)
    print("RECREATING THE BUG")
    print("="*80)

    env = SplendorEnv(num_players=2)
    obs, mask, info = env.reset(seed=42)

    print("\nStarting game with seed=42")
    print("Taking first legal action each turn until bug appears...\n")

    step = 0
    while True:
        step += 1

        legal_count = mask.sum()
        game_legal_count = len(env.game.legal_moves())

        print(f"Step {step}:")
        print(f"  Mask legal actions: {legal_count}")
        print(f"  Game legal moves: {game_legal_count}")

        if legal_count == 0:
            print(f"\n{'='*80}")
            print("BUG TRIGGERED AT STEP", step)
            print("="*80)
            return env, step

        # Take first legal action
        legal_actions = np.where(mask)[0]
        action = legal_actions[0]
        print(f"  Taking action {action}")

        obs, reward, done, mask, info = env.step(action)

        if done:
            print("\nGame finished normally")
            return None, step


def analyze_state(env):
    """Analyze the game state and determine what legal moves SHOULD exist."""

    state = env.game.state_summary()
    current_player = state['current_player']
    player = state['players'][current_player]
    bank = state['bank']
    market = state['market']

    print("\n" + "="*80)
    print("GAME STATE ANALYSIS")
    print("="*80)

    print(f"\nCurrent Player: {current_player}")
    print(f"\nPlayer State:")
    print(f"  Tokens: {player['tokens']}")
    print(f"    White:  {player['tokens'][0]}")
    print(f"    Blue:   {player['tokens'][1]}")
    print(f"    Green:  {player['tokens'][2]}")
    print(f"    Red:    {player['tokens'][3]}")
    print(f"    Black:  {player['tokens'][4]}")
    print(f"    Gold:   {player['tokens'][5]}")
    print(f"  Total tokens: {sum(player['tokens'])}")
    print(f"  Prestige: {player['prestige_points']}")
    print(f"  Played cards: {len(player['played'])}")
    print(f"  Reserved cards: {len(player['reserved'])}")

    print(f"\nBank:")
    print(f"  White:  {bank[0]}")
    print(f"  Blue:   {bank[1]}")
    print(f"  Green:  {bank[2]}")
    print(f"  Red:    {bank[3]}")
    print(f"  Black:  {bank[4]}")
    print(f"  Gold:   {bank[5]}")

    print(f"\nMarket:")
    for tier_idx, tier in enumerate(market):
        print(f"  Tier {tier_idx + 1}:")
        valid_cards = [c for c in tier if c['valid']]
        print(f"    Valid cards: {len(valid_cards)}")
        for i, card in enumerate(valid_cards[:2]):  # Show first 2
            print(f"      Card {i}: {card['prestige_points']} pts, "
                  f"cost={card['cost']}, bonus={card['bonus_color']}")

    print("\n" + "="*80)
    print("LEGAL MOVES ANALYSIS")
    print("="*80)

    should_be_legal = []

    # Check: Can we take 2 same tokens?
    print("\n1. TAKE 2 SAME TOKENS:")
    colors = ['White', 'Blue', 'Green', 'Red', 'Black']
    player_total = sum(player['tokens'])

    for color_idx, color_name in enumerate(colors):
        bank_count = bank[color_idx]
        if bank_count >= 4:
            # Can take 2 if bank has 4+
            if player_total + 2 <= 10:  # Hand limit
                print(f"  [YES] Can take 2 {color_name} (bank has {bank_count})")
                should_be_legal.append(f"Take 2 {color_name}")
            else:
                print(f"  [NO] Cannot take 2 {color_name} (would exceed hand limit)")
        else:
            print(f"  [NO] Cannot take 2 {color_name} (bank only has {bank_count}, need 4+)")

    # Check: Can we take 3 different tokens?
    print("\n2. TAKE 3 DIFFERENT TOKENS:")
    available_colors = [i for i in range(5) if bank[i] > 0]
    print(f"  Colors with tokens in bank: {[colors[i] for i in available_colors]}")

    if len(available_colors) >= 3:
        if player_total + 3 <= 10:
            print(f"  [YES] Can take 3 different tokens")
            from itertools import combinations
            for combo in list(combinations(available_colors, 3))[:3]:  # Show first 3
                combo_names = [colors[i] for i in combo]
                should_be_legal.append(f"Take 3 different: {combo_names}")
                print(f"    Example: {combo_names}")
        else:
            print(f"  [NO] Cannot take 3 (would exceed hand limit)")
    else:
        print(f"  [NO] Cannot take 3 (only {len(available_colors)} colors available)")

    # Check: Can we reserve a card?
    print("\n3. RESERVE A CARD:")
    total_reserved = len(player['reserved'])
    if total_reserved >= 3:
        print(f"  [NO] Cannot reserve (already have 3 reserved cards)")
    else:
        # Check each tier
        can_reserve_any = False
        for tier_idx, tier in enumerate(market):
            valid_cards = [c for c in tier if c['valid']]
            if valid_cards:
                will_get_gold = bank[5] > 0 and player_total < 10
                print(f"  [YES] Can reserve from Tier {tier_idx + 1} "
                      f"({len(valid_cards)} cards available)")
                if will_get_gold:
                    print(f"    (and will receive 1 gold token)")
                should_be_legal.append(f"Reserve from Tier {tier_idx + 1}")
                can_reserve_any = True

        if not can_reserve_any:
            print(f"  [NO] No cards available to reserve")

    # Check: Can we buy a card from market?
    print("\n4. BUY CARD FROM MARKET:")

    # Calculate bonuses
    bonuses = [0] * 5
    for card in player['played']:
        if card['valid']:
            bonuses[card['bonus_color']] += 1

    print(f"  Player bonuses: {bonuses}")
    print(f"  (White={bonuses[0]}, Blue={bonuses[1]}, Green={bonuses[2]}, "
          f"Red={bonuses[3]}, Black={bonuses[4]})")

    can_buy_any = False
    for tier_idx, tier in enumerate(market):
        for slot_idx, card in enumerate(tier):
            if not card['valid']:
                continue

            # Check if affordable
            cost = card['cost']
            gold_needed = 0
            affordable = True

            for color in range(5):
                effective_cost = max(0, cost[color] - bonuses[color])
                if player['tokens'][color] < effective_cost:
                    gold_needed += effective_cost - player['tokens'][color]

            if gold_needed > player['tokens'][5]:
                affordable = False

            if affordable:
                print(f"  [YES] Can buy Tier {tier_idx + 1} Slot {slot_idx}: "
                      f"{card['prestige_points']} pts, cost={cost}")
                should_be_legal.append(f"Buy Tier {tier_idx + 1} Slot {slot_idx}")
                can_buy_any = True

    if not can_buy_any:
        print(f"  [NO] Cannot afford any cards")

    # Check: Can we buy from reserve?
    print("\n5. BUY CARD FROM RESERVE:")
    if not player['reserved']:
        print(f"  [NO] No reserved cards")
    else:
        for idx, card in enumerate(player['reserved']):
            if not card['valid']:
                continue

            cost = card['cost']
            gold_needed = 0
            affordable = True

            for color in range(5):
                effective_cost = max(0, cost[color] - bonuses[color])
                if player['tokens'][color] < effective_cost:
                    gold_needed += effective_cost - player['tokens'][color]

            if gold_needed > player['tokens'][5]:
                affordable = False

            if affordable:
                print(f"  [YES] Can buy reserved card {idx}: "
                      f"{card['prestige_points']} pts, cost={cost}")
                should_be_legal.append(f"Buy reserved card {idx}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nC++ game reports: {len(env.game.legal_moves())} legal moves")
    print(f"Should have: {len(should_be_legal)} legal moves")

    if should_be_legal:
        print(f"\nLegal moves that SHOULD be available:")
        for i, move in enumerate(should_be_legal[:10], 1):
            print(f"  {i}. {move}")
    else:
        print(f"\nNo legal moves should be available (game is stuck)")

    print("\n" + "="*80)
    print("BUG DIAGNOSIS")
    print("="*80)

    if len(should_be_legal) > 0:
        print("""
The C++ move generator is INCORRECTLY returning zero legal moves.

Possible causes:
1. Bug in move_generator.cpp - not checking all move types
2. Bug in token hand limit checking (player has 8 tokens, limit is 10)
3. Bug in bank availability checking
4. Bug in affordability calculations

The game should NOT be stuck - there are valid moves available!

RECOMMENDATION:
This is a bug in the C++ game engine that needs to be fixed in:
  src/move_generator.cpp

As a workaround, our Python code should:
1. Detect when game reports 0 moves but isn't terminal
2. Treat this as a terminal state (game stuck/deadlocked)
3. Log a warning about the C++ bug
        """)
    else:
        print("""
This appears to be a legitimate deadlock state where no moves are possible.
However, the game should have been marked as terminal.

This is still a bug - the game should detect deadlocks and mark them as terminal.
        """)


if __name__ == '__main__':
    print(__doc__)

    # Recreate the bug
    env, step = recreate_bug()

    if env is None:
        print("\nBug did not occur in this run")
    else:
        # Analyze the state
        analyze_state(env)

        print("\n" + "="*80)
        print("You can now examine the game state saved in the 'env' variable")
        print("="*80)
