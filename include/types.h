#pragma once
#include "constants.h"
#include <array>
#include <vector>
#include <cstdint>

namespace splendor {

struct Card {
    uint8_t prestige_points = 0;
    Color bonus_color = WHITE;
    std::array<uint8_t, DEV_COLORS> cost{};
    uint8_t tier = 1;
    bool valid = false;
    
    Card() = default;
    Card(uint8_t points, Color color, std::array<uint8_t, DEV_COLORS> c, uint8_t t = false);
};

struct Noble {
    std::array<uint8_t, DEV_COLORS> req{};
    uint8_t prestige_points = 0;
    
    Noble() = default;
    Noble(std::array<uint8_t, DEV_COLORS> r, uint8_t points);
};

} 