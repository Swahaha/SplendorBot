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
    uint8_t id;
    bool valid = false;

    bool operator==(const Card& other) const {
        return id == other.id; // compare the right fields for identity
    }
    
    Card() = default;
    Card(uint8_t points, Color color, std::array<uint8_t, DEV_COLORS> c, uint8_t t, uint8_t i);
};

struct Noble {
    std::array<uint8_t, DEV_COLORS> req{};
    uint8_t prestige_points = 0;
    
    Noble() = default;
    Noble(std::array<uint8_t, DEV_COLORS> r, uint8_t points);
};

} 