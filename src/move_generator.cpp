#include "move_generator.h"
#include <pybind11/stl.h>
#include <algorithm>

namespace splendor {

py::list MoveGenerator::GetLegalMoves(const SplendorGame& game) {
    py::list moves;
    
    GenerateBuyFromMarketMoves(game, moves);
    GenerateBuyFromReserveMoves(game, moves);
    GenerateReserveCardMoves(game, moves);
    GenerateTakeTokensMoves(game, moves);
    
    return moves;
}

void MoveGenerator::GenerateBuyFromMarketMoves(const SplendorGame& game, py::list& moves) {
    const auto& player = game.players_[game.current_player_];
    const auto discounts = player.get_bonuses();
    
    for (int tier = 0; tier < TIERS; ++tier) {
        for (int slot = 0; slot < SLOTS_PER_TIER; ++slot) {
            const Card& card = game.market_[tier][slot];
            if (!card.valid) continue;
            
            // Check if player can afford the card
            int gold_needed = 0;
            
            for (int color = 0; color < DEV_COLORS; ++color) {
                int required = std::max(0, (int)card.cost[color] - (int)discounts[color]);
                if (player.tokens[color] < required) {
                    gold_needed += required - player.tokens[color];
                }
            }
            
            if (gold_needed <= player.tokens[GOLD]) {
                py::dict payload;
                payload["tier"] = tier + 1;  // 1-indexed for Python
                payload["slot"] = slot;
                moves.append(py::make_tuple((int)BUY_CARD_FROM_MARKET, payload));
            }
        }
    }
}

void MoveGenerator::GenerateBuyFromReserveMoves(const SplendorGame& game, py::list& moves) {
    const auto& player = game.players_[game.current_player_];
    const auto discounts = player.get_bonuses();
    
    for (size_t i = 0; i < player.reserved.size(); ++i) {
        const Card& card = player.reserved[i];
        
        // Check if player can afford the card
        int gold_needed = 0;
        // bool can_afford = true;
        
        for (int color = 0; color < DEV_COLORS; ++color) {
            int required = std::max(0, (int)card.cost[color] - (int)discounts[color]);
            if (player.tokens[color] < required) {
                gold_needed += required - player.tokens[color];
            }
        }
        
        if (gold_needed <= player.tokens[GOLD]) {
            py::dict payload;
            payload["index"] = (int)i;
            payload["card"] = StateSerializer::CardToDict(card);
            moves.append(py::make_tuple((int)BUY_CARD_FROM_RESERVE, payload));
        }
    }
}

void MoveGenerator::GenerateReserveCardMoves(const SplendorGame& game, py::list& moves) {
    const auto& player = game.players_[game.current_player_];
    
    // Can only reserve if under the limit
    if (player.reserved.size() >= RESERVED_LIMIT) {
        return;
    }
    
    for (int tier = 0; tier < TIERS; ++tier) {
        for (int slot = 0; slot < SLOTS_PER_TIER; ++slot) {
            const Card& card = game.market_[tier][slot];
            if (!card.valid) continue;
            
            py::dict payload;
            payload["tier"] = tier + 1;  // 1-indexed for Python
            payload["slot"] = slot;
            moves.append(py::make_tuple((int)RESERVE_CARD, payload));
        }
    }
    
    // TODO: Add reserve from deck moves
}

void MoveGenerator::GenerateTakeTokensMoves(const SplendorGame& game, py::list& moves) {
    const auto& player = game.players_[game.current_player_];
    const int current_tokens = player.total_tokens();
    
    // Generate "take 3 different tokens" moves
    if (current_tokens <= TOKEN_HAND_LIMIT - 3) {
        for (int a = 0; a < DEV_COLORS; ++a) {
            if (game.bank_[a] == 0) continue;
            for (int b = a + 1; b < DEV_COLORS; ++b) {
                if (game.bank_[b] == 0) continue;
                for (int c = b + 1; c < DEV_COLORS; ++c) {
                    if (game.bank_[c] == 0) continue;
                    
                    py::list token_vec;
                    for (int color = 0; color < DEV_COLORS; ++color) {
                        if (color == a || color == b || color == c) {
                            token_vec.append(1);
                        } else {
                            token_vec.append(0);
                        }
                    }
                    moves.append(py::make_tuple((int)GET_3_FROM_BANK, token_vec));
                }
            }
        }
    }
    
    // Generate "take 2 same tokens" moves
    if (current_tokens <= TOKEN_HAND_LIMIT - 2) {
        for (int color = 0; color < DEV_COLORS; ++color) {
            // Need at least 4 tokens in bank to take 2
            if (game.bank_[color] >= 4) {
                py::list token_vec;
                for (int c = 0; c < DEV_COLORS; ++c) {
                    token_vec.append(c == color ? 2 : 0);
                }
                moves.append(py::make_tuple((int)GET_2_FROM_BANK, token_vec));
            }
        }
    }
}

}