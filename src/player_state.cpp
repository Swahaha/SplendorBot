#include "player_state.h"

namespace splendor {

int PlayerState::total_tokens() const {
    int sum = 0;
    for (auto t : tokens) sum += t;
    return sum;
}

std::array<uint8_t, DEV_COLORS> PlayerState::get_bonuses() const {
    std::array<uint8_t, DEV_COLORS> bonuses{};
    for (const auto& card : played_cards) {
        if (card.bonus_color != GOLD) {
            bonuses[static_cast<int>(card.bonus_color)]++;
        }
    }
    return bonuses;
}

} 