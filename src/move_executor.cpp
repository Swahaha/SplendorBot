#include "move_executor.h"
#include "game_initializer.h"
#include <algorithm>

namespace splendor {

void MoveExecutor::PerformMove(SplendorGame& game, py::object move_obj) {
    auto move = move_obj.cast<py::tuple>();
    int action = move[0].cast<int>();
    py::object payload = move[1];

    switch (action) {
        case RESERVE_CARD: {
            int tier = payload["tier"].cast<int>() - 1;
            int slot = payload["slot"].cast<int>();
            ReserveCard(game, tier, slot);
            break;
        }
        case BUY_CARD_FROM_MARKET: {
            int tier = payload["tier"].cast<int>() - 1;
            int slot = payload["slot"].cast<int>();
            BuyCardFromMarket(game, tier, slot);
            break;
        }
        default:
            throw std::runtime_error("not implemented yet");
    }

    // CheckNobles(game); WE SKIP THIS FOR NOW
    game.end_turn();
}

void MoveExecutor::ReserveCard(SplendorGame& game, int tier, int slot) {
    auto& player = game.players_[game.current_player_];
    if (player.reserved.size() >= RESERVED_LIMIT) return;
    
    Card card = game.market_[tier][slot];
    if (!card.valid) return;
    
    player.reserved.push_back(card);
    game.market_[tier][slot] = Card();
    
    if (game.bank_[GOLD] > 0 and player.total_tokens() < TOKEN_HAND_LIMIT) {
        game.bank_[GOLD]--;
        player.tokens[GOLD]++;
    }
    
    GameInitializer::DrawIntoSlot(game, tier, slot);
}

void BuyCardFromMarket(SplendorGame& game, int tier, int slot){
    throw std::runtime_error("not implemented yet");
}

bool MoveExecutor::CanAfford(const PlayerState& player, const Card& card, 
                            const std::array<uint8_t, DEV_COLORS>& discounts) {
    int gold_needed = 0;
    for (int i = 0; i < DEV_COLORS; ++i) {
        int required = std::max(0, (int)card.cost[i] - (int)discounts[i]);
        if (player.tokens[i] < required) {
            gold_needed += required - player.tokens[i];
        }
    }
    return gold_needed <= player.tokens[GOLD];
}

// ... other move executor implementations

} 