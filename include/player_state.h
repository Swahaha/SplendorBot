#pragma once
#include "types.h"
#include <vector>

namespace splendor {

struct PlayerState {
    std::array<uint8_t, COLOR_COUNT> tokens{};
    uint8_t prestige_points = 0;
    std::vector<Card> reserved;
    std::vector<Card> played_cards;
    std::vector<Noble> nobles_owned;
    
    PlayerState() = default;
    
    int total_tokens() const;
    std::array<uint8_t, DEV_COLORS> get_bonuses() const;
};

} // namespace splendor