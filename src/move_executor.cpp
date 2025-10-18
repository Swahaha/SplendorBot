#include "move_executor.h"
#include "game_initializer.h"
#include "move_generator.h"
#include <algorithm>
#include <pybind11/stl.h>
#include <iostream>

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
        case BUY_CARD_FROM_RESERVE: {
            // uint8_t points = payload["prestige_points"].cast<uint8_t>();
            // Color col = payload["bonus_color"].cast<Color>();
            // std::array<uint8_t, DEV_COLORS> cost = payload["cost"].cast<std::array<uint8_t, DEV_COLORS>>();
            // uint8_t tier = payload["tier"].cast<uint8_t>();
            uint8_t idx = payload["index"].cast<uint8_t>();
            // Card card = Card(points, col, cost, tier, id);
            BuyCardFromReserve(game, idx);
            break;
        }
        case GET_3_FROM_BANK: 
        case GET_2_FROM_BANK:
        {
            auto take = payload.cast<std::vector<int>>();
            auto& player = game.players_[game.current_player_];
            for (size_t i = 0; i < take.size(); i++){
                player.tokens[i] += take[i];
                game.bank_[i] -= take[i];
            }
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

void MoveExecutor::BuyCardFromReserve(SplendorGame& game, uint8_t idx){
    auto& player = game.players_[game.current_player_];
    auto& card = player.reserved[idx];
    const auto discounts = player.get_bonuses();
    std::vector<int> costs(COLOR_COUNT, 0);

    int gold_needed = 0;
            
    for (int color = 0; color < DEV_COLORS; ++color) {
        int required = std::max(0, (int)card.cost[color] - discounts[color]);
        if (player.tokens[color] < required) {
            gold_needed += required - player.tokens[color];
        }
        costs[color] = -std::max(-required, -player.tokens[color]);
    }

    costs[GOLD] = gold_needed;

    for (size_t i = 0; i < costs.size(); i++){
        player.tokens[i] -= costs[i];
        game.bank_[i] += costs[i];
    }

    player.played_cards.push_back(card);

    auto& v = player.reserved;
    v[idx] = v.back();
    v.pop_back();
    player.prestige_points += card.prestige_points;
}

void MoveExecutor::BuyCardFromMarket(SplendorGame& game, int tier, int slot){
    auto& player = game.players_[game.current_player_];
    const auto discounts = player.get_bonuses();
    Card card = game.market_[tier][slot];
    std::vector<int> costs(COLOR_COUNT, 0);

    int gold_needed = 0;
            
    for (int color = 0; color < DEV_COLORS; ++color) {
        int required = std::max(0, (int)card.cost[color] - discounts[color]);
        if (player.tokens[color] < required) {
            gold_needed += required - player.tokens[color];
        }
        costs[color] = -std::max(-required, -player.tokens[color]);
    }

    costs[GOLD] = gold_needed;

    for (size_t i = 0; i < costs.size(); i++){
        player.tokens[i] -= costs[i];
        game.bank_[i] += costs[i];
    }

    player.played_cards.push_back(card);
    game.market_[tier][slot] = Card();
    player.prestige_points += card.prestige_points;
    GameInitializer::DrawIntoSlot(game, tier, slot);
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



} 