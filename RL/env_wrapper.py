"""
Splendor Environment Wrapper for RL Training
Provides a standardized interface with observation vectors, action masks, and step logic.
"""

import sys
import os

# IMPORTANT: Change working directory to project root so C++ can find data/cards.csv
# But keep the RL directory in sys.path so imports still work
rl_dir = os.path.dirname(os.path.abspath(__file__))
if rl_dir not in sys.path:
    sys.path.insert(0, rl_dir)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if os.path.exists(os.path.join(project_root, 'data', 'cards.csv')):
    os.chdir(project_root)

# Try to import from multiple possible locations
try:
    import splendor_game as sg
except ImportError:
    # Try build directory
    build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
    if os.path.exists(build_path):
        sys.path.insert(0, build_path)

    # Try python directory
    python_path = os.path.join(os.path.dirname(__file__), '..', 'python')
    if os.path.exists(python_path):
        sys.path.insert(0, python_path)

    import splendor_game as sg

import numpy as np
from typing import Tuple, Dict, List, Any


class SplendorEnv:
    """
    Gym-style environment wrapper for Splendor game.

    Observation: Fixed-size numpy array encoding game state
    Actions: Integer IDs from 0 to max_actions-1
    Action Mask: Boolean array indicating legal actions
    """

    # Action space configuration
    MAX_RESERVE_MARKET = 12  # 3 tiers × 4 slots
    MAX_RESERVE_DECK = 3     # 3 tiers (reserve from top of deck)
    MAX_BUY_MARKET = 12      # 3 tiers × 4 slots
    MAX_BUY_RESERVE = 3      # Up to 3 reserved cards
    MAX_TAKE3 = 10           # C(5,3) = 10 combinations of 3 different colors
    MAX_TAKE2 = 5            # 5 colors to take 2 from

    # Total action space size
    ACTION_DIM = (MAX_RESERVE_MARKET + MAX_RESERVE_DECK + MAX_BUY_MARKET +
                  MAX_BUY_RESERVE + MAX_TAKE3 + MAX_TAKE2)

    # Observation configuration (per-player perspective)
    # Player state: 6 tokens + 5 bonuses + 1 prestige + 1 card_count + 1 reserved_count = 14
    # Other players: 14 × 3 = 42 (for 4 players max)
    # Bank: 6
    # Market: 12 cards × 7 features (tier, color, points, 5 costs) = 84
    # Nobles: 5 × 6 features (points, 5 requirements) = 30
    # Reserved cards: 3 × 7 features = 21
    # Current player indicator: 1
    OBS_DIM = 14 + 42 + 6 + 84 + 30 + 21 + 1  # = 198

    def __init__(self, num_players: int = 2, seed: int = 0):
        """
        Initialize Splendor environment.

        Args:
            num_players: Number of players (2-4)
            seed: Random seed for reproducibility
        """
        self.num_players = num_players
        self.seed = seed
        self.game = None
        self._action_map = None
        self._build_action_mappings()

    def _build_action_mappings(self):
        """Build mappings between action IDs and game moves."""
        self._action_map = {}
        action_id = 0

        # Reserve from market: 12 actions
        for tier in range(3):
            for slot in range(4):
                self._action_map[action_id] = ('reserve_market', tier, slot)
                action_id += 1

        # Reserve from deck: 3 actions
        for tier in range(3):
            self._action_map[action_id] = ('reserve_deck', tier)
            action_id += 1

        # Buy from market: 12 actions
        for tier in range(3):
            for slot in range(4):
                self._action_map[action_id] = ('buy_market', tier, slot)
                action_id += 1

        # Buy from reserve: 3 actions
        for idx in range(3):
            self._action_map[action_id] = ('buy_reserve', idx)
            action_id += 1

        # Take 3 different: 10 combinations (without gold)
        from itertools import combinations
        for combo in combinations(range(5), 3):  # 5 dev colors
            self._action_map[action_id] = ('take3', combo)
            action_id += 1

        # Take 2 same: 5 actions
        for color in range(5):  # 5 dev colors
            self._action_map[action_id] = ('take2', color)
            action_id += 1

        assert action_id == self.ACTION_DIM, f"Action mapping mismatch: {action_id} != {self.ACTION_DIM}"

    def reset(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            obs: Observation array (OBS_DIM,)
            mask: Action mask (ACTION_DIM,)
            info: Additional information dict
        """
        if seed is None:
            seed = self.seed

        self.game = sg.SplendorGame(self.num_players, seed)
        obs = self._get_observation()
        mask = self._get_action_mask()
        info = {'current_player': self.game.current_player()}

        return obs, mask, info

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict]:
        """
        Execute action and return next state.

        Args:
            action_id: Integer action ID

        Returns:
            next_obs: Next observation
            reward: Reward (prestige points gained this turn)
            done: Whether episode is terminal
            next_mask: Next action mask
            info: Additional information
        """
        # Get current player and their prestige before move
        current_player = self.game.current_player()
        state_before = self.game.state_summary()
        prestige_before = state_before['players'][current_player]['prestige_points']

        # Convert action ID to game move
        game_move = self._action_id_to_game_move(action_id, state_before)

        # Execute move
        self.game.perform_move(game_move)

        # Get new state
        state_after = self.game.state_summary()
        done = self.game.is_terminal()

        # Calculate reward (prestige gained)
        prestige_after = state_after['players'][current_player]['prestige_points']
        reward = float(prestige_after - prestige_before)

        # Get next observation and mask
        next_obs = self._get_observation()
        next_mask = self._get_action_mask()

        info = {
            'current_player': self.game.current_player(),
            'previous_player': current_player,
            'prestige_gained': reward,
            'terminal': done
        }

        return next_obs, reward, done, next_mask, info

    def _action_id_to_game_move(self, action_id: int, state: Dict) -> Tuple:
        """Convert action ID to game move tuple."""
        action_type, *params = self._action_map[action_id]

        if action_type == 'reserve_market':
            tier, slot = params
            return (0, {'tier': tier + 1, 'slot': slot})  # RESERVE_CARD = 0

        elif action_type == 'reserve_deck':
            # Not yet supported in game - fallback to first legal move
            # This will be masked out anyway
            return self.game.legal_moves()[0]

        elif action_type == 'buy_market':
            tier, slot = params
            return (1, {'tier': tier + 1, 'slot': slot})  # BUY_CARD_FROM_MARKET = 1

        elif action_type == 'buy_reserve':
            idx = params[0]
            current_player = self.game.current_player()
            reserved = state['players'][current_player]['reserved']
            if idx < len(reserved):
                return (2, {'index': idx, 'card': reserved[idx]})  # BUY_CARD_FROM_RESERVE = 2
            return self.game.legal_moves()[0]  # Fallback

        elif action_type == 'take3':
            combo = params[0]
            tokens = [0] * 5
            for color in combo:
                tokens[color] = 1
            return (3, tokens)  # GET_3_FROM_BANK = 3

        elif action_type == 'take2':
            color = params[0]
            tokens = [0] * 5
            tokens[color] = 2
            return (4, tokens)  # GET_2_FROM_BANK = 4

        # Should never reach here
        return self.game.legal_moves()[0]

    def _get_observation(self) -> np.ndarray:
        """
        Convert game state to fixed-size observation vector.

        Observation encoding (from current player's perspective):
        - Current player state (14): tokens(6) + bonuses(5) + prestige(1) + card_count(1) + reserved_count(1)
        - Other players (42): 3 × 14 for up to 3 opponents (padded with zeros)
        - Bank (6): token counts
        - Market (84): 12 cards × 7 (tier, bonus_color, points, cost_w, cost_b, cost_g, cost_r, cost_bk)
        - Nobles (30): 5 × 6 (points, req_w, req_b, req_g, req_r, req_bk)
        - Reserved cards (21): 3 × 7 (same as market cards)
        - Current player indicator (1): 0-3
        """
        state = self.game.state_summary()
        current_player = state['current_player']

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        idx = 0

        # Current player state (14)
        player = state['players'][current_player]
        obs[idx:idx+6] = player['tokens']  # 6 token types
        idx += 6

        bonuses = self._get_player_bonuses(player)
        obs[idx:idx+5] = bonuses  # 5 dev colors
        idx += 5

        obs[idx] = player['prestige_points']
        idx += 1

        obs[idx] = len(player['played'])  # card count
        idx += 1

        obs[idx] = len(player['reserved'])  # reserved count
        idx += 1

        # Other players (42 for 3 opponents)
        for p_idx in range(self.num_players):
            if p_idx == current_player:
                continue

            other_player = state['players'][p_idx]
            obs[idx:idx+6] = other_player['tokens']
            idx += 6

            other_bonuses = self._get_player_bonuses(other_player)
            obs[idx:idx+5] = other_bonuses
            idx += 5

            obs[idx] = other_player['prestige_points']
            idx += 1

            obs[idx] = len(other_player['played'])
            idx += 1

            obs[idx] = len(other_player['reserved'])
            idx += 1

        # Pad for missing players (if < 4 players)
        idx = 14 + 42  # Jump to after all player slots

        # Bank (6)
        obs[idx:idx+6] = state['bank']
        idx += 6

        # Market (84): 12 cards × 7 features
        for tier_idx, tier in enumerate(state['market']):
            for card in tier:
                if card['valid']:
                    obs[idx] = card['tier']
                    obs[idx+1] = card['bonus_color']
                    obs[idx+2] = card['prestige_points']
                    obs[idx+3:idx+8] = card['cost']  # 5 costs
                idx += 7

        # Nobles (30): 5 × 6 features
        for i in range(5):
            if i < len(state['nobles']):
                noble = state['nobles'][i]
                obs[idx] = noble['prestige_points']
                obs[idx+1:idx+6] = noble['requirements']  # 5 requirements
            idx += 6

        # Reserved cards (21): 3 × 7 features
        for i in range(3):
            if i < len(player['reserved']):
                card = player['reserved'][i]
                obs[idx] = card['tier']
                obs[idx+1] = card['bonus_color']
                obs[idx+2] = card['prestige_points']
                obs[idx+3:idx+8] = card['cost']
            idx += 7

        # Current player indicator
        obs[idx] = current_player
        idx += 1

        assert idx == self.OBS_DIM, f"Observation size mismatch: {idx} != {self.OBS_DIM}"

        return obs

    def _get_player_bonuses(self, player_state: Dict) -> np.ndarray:
        """Calculate player's bonuses from played cards."""
        bonuses = np.zeros(5, dtype=np.float32)
        for card in player_state['played']:
            if card['valid']:
                bonuses[card['bonus_color']] += 1
        return bonuses

    def _get_action_mask(self) -> np.ndarray:
        """
        Generate boolean mask for legal actions.

        Returns:
            mask: Boolean array of shape (ACTION_DIM,)
        """
        mask = np.zeros(self.ACTION_DIM, dtype=bool)
        legal_moves = self.game.legal_moves()
        state = self.game.state_summary()

        for move in legal_moves:
            action_type, payload = move
            action_ids = self._game_move_to_action_ids(action_type, payload, state)
            for aid in action_ids:
                if aid is not None and 0 <= aid < self.ACTION_DIM:
                    mask[aid] = True

        return mask

    def _game_move_to_action_ids(self, action_type: int, payload: Any, state: Dict) -> List[int]:
        """Convert game move to action ID(s)."""
        action_ids = []

        if action_type == 0:  # RESERVE_CARD
            tier = payload['tier'] - 1  # Convert to 0-indexed
            slot = payload['slot']
            action_id = tier * 4 + slot  # First 12 actions
            action_ids.append(action_id)

        elif action_type == 1:  # BUY_CARD_FROM_MARKET
            tier = payload['tier'] - 1
            slot = payload['slot']
            action_id = 15 + tier * 4 + slot  # After reserve actions (12+3=15)
            action_ids.append(action_id)

        elif action_type == 2:  # BUY_CARD_FROM_RESERVE
            idx = payload['index']
            action_id = 27 + idx  # After buy market (15+12=27)
            action_ids.append(action_id)

        elif action_type == 3:  # GET_3_FROM_BANK
            # Find which combination of 3 this is
            tokens = payload
            colors = [i for i, count in enumerate(tokens) if count > 0]
            if len(colors) == 3:
                # Map to combination index
                from itertools import combinations
                combos = list(combinations(range(5), 3))
                combo_tuple = tuple(sorted(colors))
                if combo_tuple in combos:
                    combo_idx = combos.index(combo_tuple)
                    action_id = 30 + combo_idx  # After buy reserve (27+3=30)
                    action_ids.append(action_id)

        elif action_type == 4:  # GET_2_FROM_BANK
            tokens = payload
            color = next((i for i, count in enumerate(tokens) if count == 2), None)
            if color is not None:
                action_id = 40 + color  # After take3 (30+10=40)
                action_ids.append(action_id)

        return action_ids

    def render(self):
        """Print current game state (for debugging)."""
        state = self.game.state_summary()
        print(f"\n=== Splendor Game State ===")
        print(f"Current Player: {state['current_player']}")
        print(f"Bank: {state['bank']}")

        for i, player in enumerate(state['players']):
            print(f"\nPlayer {i}:")
            print(f"  Tokens: {player['tokens']}")
            print(f"  Prestige: {player['prestige_points']}")
            print(f"  Cards: {len(player['played'])}")
            print(f"  Reserved: {len(player['reserved'])}")
