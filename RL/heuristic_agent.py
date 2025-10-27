"""
Heuristic Agent for Splendor
Implements strategic decision-making rules for baseline performance.
"""

import numpy as np
from typing import Tuple, Dict, List, Any
from env_wrapper import SplendorEnv


class HeuristicAgent:
    """
    Rule-based heuristic agent for Splendor.

    Decision hierarchy:
    1. Winning buy: Buy card that reaches 15+ points
    2. Best affordable buy: Maximize value score
    3. Strategic reserve: Reserve high-value cards within reach
    4. Take 3 distinct: Maximize token needs
    5. Take 2 same: Target specific card needs
    6. Fallback: First legal action
    """

    def __init__(self, env: SplendorEnv):
        self.env = env
        self.num_players = env.num_players

    def choose_action(self, obs: np.ndarray, mask: np.ndarray, state: Dict = None) -> int:
        """
        Choose action based on heuristic rules.

        Args:
            obs: Observation vector
            mask: Boolean action mask
            state: Optional game state dict (for efficiency)

        Returns:
            action_id: Integer action ID, or None if no legal moves (turn should be skipped)
        """
        # Check if there are any legal moves
        if not np.any(mask):
            # No legal moves - turn will be skipped
            return None

        if state is None:
            state = self.env.game.state_summary()

        current_player = state['current_player']
        player = state['players'][current_player]

        # Get legal actions categorized by type
        legal_moves = self.env.game.legal_moves()
        buy_market_moves, buy_reserve_moves, reserve_moves, take3_moves, take2_moves = \
            self._categorize_moves(legal_moves, state)

        # 1. Winning buy: If any buy reaches 15+ points, take highest
        winning_action = self._check_winning_buy(buy_market_moves, buy_reserve_moves, player, mask, state)
        if winning_action is not None:
            return winning_action

        # 2. Best affordable buy: Maximize value score
        best_buy_action = self._best_affordable_buy(buy_market_moves, buy_reserve_moves, player, mask, state)
        if best_buy_action is not None:
            return best_buy_action

        # 3. Strategic reserve: Reserve valuable cards within reach
        reserve_action = self._strategic_reserve(reserve_moves, player, mask, state)
        if reserve_action is not None:
            return reserve_action

        # 4. Take 3 distinct: Maximize token needs
        take3_action = self._best_take3(take3_moves, player, mask, state)
        if take3_action is not None:
            return take3_action

        # 5. Take 2 same: Target specific needs
        take2_action = self._best_take2(take2_moves, player, mask, state)
        if take2_action is not None:
            return take2_action

        # 6. Fallback: First legal action
        legal_ids = np.where(mask)[0]
        return legal_ids[0] if len(legal_ids) > 0 else None

    def _categorize_moves(self, legal_moves: List[Tuple], state: Dict) -> Tuple:
        """Categorize moves by type."""
        buy_market = []
        buy_reserve = []
        reserve = []
        take3 = []
        take2 = []

        for move in legal_moves:
            action_type, payload = move
            if action_type == 1:  # BUY_CARD_FROM_MARKET
                buy_market.append(move)
            elif action_type == 2:  # BUY_CARD_FROM_RESERVE
                buy_reserve.append(move)
            elif action_type == 0:  # RESERVE_CARD
                reserve.append(move)
            elif action_type == 3:  # GET_3_FROM_BANK
                take3.append(move)
            elif action_type == 4:  # GET_2_FROM_BANK
                take2.append(move)

        return buy_market, buy_reserve, reserve, take3, take2

    def _check_winning_buy(self, buy_market: List, buy_reserve: List,
                          player: Dict, mask: np.ndarray, state: Dict) -> int:
        """Check if any buy reaches 15+ points; return highest."""
        current_prestige = player['prestige_points']
        best_action = None
        best_points = -1

        # Check market buys
        for move in buy_market:
            _, payload = move
            card = state['market'][payload['tier'] - 1][payload['slot']]
            if not card['valid']:
                continue

            total_prestige = current_prestige + card['prestige_points']
            if total_prestige >= 15 and card['prestige_points'] > best_points:
                action_id = self._move_to_action_id(move, state)
                if action_id is not None and mask[action_id]:
                    best_action = action_id
                    best_points = card['prestige_points']

        # Check reserve buys
        for move in buy_reserve:
            _, payload = move
            card = payload['card']
            if not card['valid']:
                continue

            total_prestige = current_prestige + card['prestige_points']
            if total_prestige >= 15 and card['prestige_points'] > best_points:
                action_id = self._move_to_action_id(move, state)
                if action_id is not None and mask[action_id]:
                    best_action = action_id
                    best_points = card['prestige_points']

        return best_action

    def _best_affordable_buy(self, buy_market: List, buy_reserve: List,
                            player: Dict, mask: np.ndarray, state: Dict) -> int:
        """Find best buy maximizing: 2*points + 0.8*noble_progress + 0.5*discount_gain."""
        bonuses = self._get_bonuses(player)
        best_action = None
        best_score = -float('inf')

        # Check market buys
        for move in buy_market:
            _, payload = move
            card = state['market'][payload['tier'] - 1][payload['slot']]
            if not card['valid']:
                continue

            action_id = self._move_to_action_id(move, state)
            if action_id is None or not mask[action_id]:
                continue

            score = self._calculate_buy_score(card, bonuses, state)
            if score > best_score:
                best_score = score
                best_action = action_id

        # Check reserve buys
        for move in buy_reserve:
            _, payload = move
            card = payload['card']
            if not card['valid']:
                continue

            action_id = self._move_to_action_id(move, state)
            if action_id is None or not mask[action_id]:
                continue

            score = self._calculate_buy_score(card, bonuses, state)
            if score > best_score:
                best_score = score
                best_action = action_id

        return best_action if best_score > -float('inf') else None

    def _calculate_buy_score(self, card: Dict, current_bonuses: np.ndarray, state: Dict) -> float:
        """Calculate value score for buying a card."""
        points = card['prestige_points']

        # Discount gain: number of colors newly at ≥1 discount
        new_bonuses = current_bonuses.copy()
        new_bonuses[card['bonus_color']] += 1
        discount_gain = np.sum((current_bonuses == 0) & (new_bonuses >= 1))

        # Noble progress: how much closer to nobles
        noble_progress = self._calculate_noble_progress(new_bonuses, state)

        score = 2.0 * points + 0.8 * noble_progress + 0.5 * discount_gain
        return score

    def _calculate_noble_progress(self, bonuses: np.ndarray, state: Dict) -> float:
        """Calculate progress toward nobles (average proximity to requirements)."""
        if len(state['nobles']) == 0:
            return 0.0

        progress = 0.0
        for noble in state['nobles']:
            req = np.array(noble['requirements'])
            # Calculate how many requirements are met
            satisfied = np.minimum(bonuses, req)
            progress += np.sum(satisfied) / (np.sum(req) + 1e-6)

        return progress / len(state['nobles'])

    def _strategic_reserve(self, reserve_moves: List, player: Dict,
                          mask: np.ndarray, state: Dict) -> int:
        """Reserve card if within ≤2 missing tokens; maximize reserve_score."""
        if len(player['reserved']) >= 3:
            return None  # Already at reserve limit

        bonuses = self._get_bonuses(player)
        tokens = np.array(player['tokens'][:5])  # Exclude gold

        best_action = None
        best_score = -float('inf')

        for move in reserve_moves:
            _, payload = move
            tier = payload['tier'] - 1
            slot = payload['slot']
            card = state['market'][tier][slot]

            if not card['valid']:
                continue

            # Check if within 2 missing tokens
            cost = np.array(card['cost'])
            effective_cost = np.maximum(0, cost - bonuses)
            missing = np.maximum(0, effective_cost - tokens)
            total_missing = np.sum(missing)

            if total_missing <= 2:
                # Calculate reserve score
                points = card['prestige_points']
                discount_gain = 1 if bonuses[card['bonus_color']] == 0 else 0
                noble_alignment = self._noble_alignment(card['bonus_color'], state)

                reserve_score = points + 0.5 * discount_gain + 0.3 * noble_alignment

                if reserve_score > best_score:
                    action_id = self._move_to_action_id(move, state)
                    if action_id is not None and mask[action_id]:
                        best_score = reserve_score
                        best_action = action_id

        return best_action if best_score > -float('inf') else None

    def _noble_alignment(self, bonus_color: int, state: Dict) -> float:
        """Check how well this bonus color aligns with noble requirements."""
        if len(state['nobles']) == 0:
            return 0.0

        alignment = 0.0
        for noble in state['nobles']:
            if noble['requirements'][bonus_color] > 0:
                alignment += noble['requirements'][bonus_color]

        return alignment / len(state['nobles'])

    def _best_take3(self, take3_moves: List, player: Dict,
                   mask: np.ndarray, state: Dict) -> int:
        """Take 3 distinct maximizing summed token needs."""
        if len(take3_moves) == 0:
            return None

        bonuses = self._get_bonuses(player)
        tokens = np.array(player['tokens'][:5])

        # Find prospective target cards (cheapest affordable or near-affordable)
        target_cards = self._find_target_cards(player, state, max_targets=3)

        best_action = None
        best_score = -float('inf')

        for move in take3_moves:
            _, payload = move
            colors = [i for i, count in enumerate(payload) if count > 0]

            # Calculate needs score
            score = self._calculate_token_needs(colors, target_cards, tokens, bonuses, state)

            if score > best_score:
                action_id = self._move_to_action_id(move, state)
                if action_id is not None and mask[action_id]:
                    best_score = score
                    best_action = action_id

        return best_action if best_score > -float('inf') else None

    def _best_take2(self, take2_moves: List, player: Dict,
                   mask: np.ndarray, state: Dict) -> int:
        """Take 2 same if it reduces missing tokens for top targets."""
        if len(take2_moves) == 0:
            return None

        bonuses = self._get_bonuses(player)
        tokens = np.array(player['tokens'][:5])

        # Find top 2 target cards
        target_cards = self._find_target_cards(player, state, max_targets=2)
        if len(target_cards) == 0:
            return None

        best_action = None
        best_reduction = 0

        for move in take2_moves:
            _, payload = move
            color = next((i for i, count in enumerate(payload) if count == 2), None)
            if color is None:
                continue

            # Check if this color reduces missing tokens for targets
            reduction = 0
            for card in target_cards:
                cost = np.array(card['cost'])
                before_missing = max(0, cost[color] - bonuses[color] - tokens[color])
                after_missing = max(0, cost[color] - bonuses[color] - (tokens[color] + 2))
                reduction += before_missing - after_missing

            if reduction > best_reduction:
                action_id = self._move_to_action_id(move, state)
                if action_id is not None and mask[action_id]:
                    best_reduction = reduction
                    best_action = action_id

        return best_action if best_reduction > 0 else None

    def _find_target_cards(self, player: Dict, state: Dict, max_targets: int = 3) -> List[Dict]:
        """Find cheapest prospective cards to target."""
        bonuses = self._get_bonuses(player)
        tokens = np.array(player['tokens'][:5])
        gold = player['tokens'][5]

        candidates = []

        # Check market cards
        for tier_idx, tier in enumerate(state['market']):
            for card in tier:
                if not card['valid']:
                    continue

                cost = np.array(card['cost'])
                effective_cost = np.maximum(0, cost - bonuses)
                missing = np.maximum(0, effective_cost - tokens)
                total_missing = np.sum(missing) - min(gold, np.sum(missing))

                # Include cards that are affordable or near-affordable
                if total_missing <= 4:
                    candidates.append({
                        'card': card,
                        'missing': total_missing,
                        'value': card['prestige_points'] + 0.5
                    })

        # Check reserved cards
        for card in player['reserved']:
            if not card['valid']:
                continue

            cost = np.array(card['cost'])
            effective_cost = np.maximum(0, cost - bonuses)
            missing = np.maximum(0, effective_cost - tokens)
            total_missing = np.sum(missing) - min(gold, np.sum(missing))

            if total_missing <= 4:
                candidates.append({
                    'card': card,
                    'missing': total_missing,
                    'value': card['prestige_points'] + 1.0  # Prefer reserved cards
                })

        # Sort by value/missing ratio and missing count
        candidates.sort(key=lambda x: (-x['value'] / (1 + x['missing']), x['missing']))

        return [c['card'] for c in candidates[:max_targets]]

    def _calculate_token_needs(self, colors: List[int], target_cards: List[Dict],
                               tokens: np.ndarray, bonuses: np.ndarray, state: Dict) -> float:
        """Calculate token needs score for given colors."""
        score = 0.0

        for card in target_cards:
            cost = np.array(card['cost'])
            effective_cost = np.maximum(0, cost - bonuses)
            missing = np.maximum(0, effective_cost - tokens)
            total_missing = np.sum(missing)

            # Weight by inverse of total missing (prioritize closer cards)
            weight = 1.0 / (1 + total_missing)

            # Sum contribution of chosen colors
            for color in colors:
                score += weight * missing[color]

        # Add noble requirements contribution
        for color in colors:
            for noble in state['nobles']:
                need = max(0, noble['requirements'][color] - bonuses[color])
                score += 0.5 * need

        return score

    def _get_bonuses(self, player: Dict) -> np.ndarray:
        """Calculate player's bonuses from played cards."""
        bonuses = np.zeros(5, dtype=np.float32)
        for card in player['played']:
            if card['valid']:
                bonuses[card['bonus_color']] += 1
        return bonuses

    def _move_to_action_id(self, move: Tuple, state: Dict) -> int:
        """Convert game move to action ID."""
        action_type, payload = move
        action_ids = self.env._game_move_to_action_ids(action_type, payload, state)
        return action_ids[0] if action_ids else None
