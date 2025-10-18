#include "state_serializer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

namespace splendor {

py::dict StateSerializer::StateSummary(const SplendorGame& game) {
    py::dict d;

    // std::cout << "Current player in StateSummary: " << game.current_player_ << std::endl;

    SerializeBank(game, d);
    SerializeMarket(game, d);
    SerializeNobles(game, d);
    SerializePlayers(game, d);
    
    d["current_player"] = game.current_player_;
    d["num_players"] = game.num_players_;
    d["game_over"] = game.check_game_over();

    return d;
}

void StateSerializer::SerializeBank(const SplendorGame& game, py::dict& summary) {
    py::list bank;
    for (int c = 0; c < COLOR_COUNT; ++c) {
        bank.append(game.bank_[c]);
    }
    summary["bank"] = bank;
}

void StateSerializer::SerializeMarket(const SplendorGame& game, py::dict& summary) {
    py::list market_list;
    
    for (int tier = 0; tier < TIERS; ++tier) {
        py::list tier_list;
        for (int slot = 0; slot < SLOTS_PER_TIER; ++slot) {
            tier_list.append(CardToDict(game.market_[tier][slot]));
        }
        market_list.append(tier_list);
    }
    
    summary["market"] = market_list;
}

void StateSerializer::SerializeNobles(const SplendorGame& game, py::dict& summary) {
    py::list nobles_list;
    
    for (const auto& noble : game.nobles_) {
        nobles_list.append(NobleToDict(noble));
    }
    
    summary["nobles"] = nobles_list;
}

void StateSerializer::SerializePlayers(const SplendorGame& game, py::dict& summary) {
    py::list players_list;
    
    for (const auto& player : game.players_) {
        py::dict player_dict;
        
        py::list tokens;
        for (int k = 0; k < COLOR_COUNT; ++k) {
            tokens.append(player.tokens[k]);
        }
        player_dict["tokens"] = tokens;
        player_dict["prestige_points"] = player.prestige_points;
        
        py::list bonuses;
        auto bonus_counts = player.get_bonuses();
        for (int k = 0; k < DEV_COLORS; ++k) {
            bonuses.append(bonus_counts[k]);
        }
        player_dict["bonuses"] = bonuses;
        player_dict["reserved"] = CardsToList(player.reserved);
        player_dict["played"] = CardsToList(player.played_cards);
        player_dict["nobles_owned"] = static_cast<int>(player.nobles_owned.size());
        player_dict["total_tokens"] = player.total_tokens();
        players_list.append(player_dict);
    }
    
    summary["players"] = players_list;
}

py::dict StateSerializer::CardToDict(const Card& card) {
    py::dict cd;
    
    cd["prestige_points"] = card.prestige_points;
    cd["bonus_color"] = static_cast<int>(card.bonus_color);
    cd["tier"] = card.tier;
    cd["valid"] = card.valid;
    
    py::list cost;
    for (int k = 0; k < DEV_COLORS; ++k) {
        cost.append(card.cost[k]);
    }
    cd["cost"] = cost;
    
    return cd;
}

py::list StateSerializer::CardsToList(const std::vector<Card>& cards) {
    py::list out;
    
    for (const auto& card : cards) {
        out.append(CardToDict(card));
    }
    
    return out;
}

py::dict StateSerializer::NobleToDict(const Noble& noble) {
    py::dict nd;
    
    nd["prestige_points"] = noble.prestige_points;
    
    py::list requirements;
    for (int k = 0; k < DEV_COLORS; ++k) {
        requirements.append(noble.req[k]);
    }
    nd["requirements"] = requirements;
    
    return nd;
}

} 