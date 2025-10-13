#include "types.h"

namespace splendor {

Card::Card(uint8_t points, Color color, std::array<uint8_t, DEV_COLORS> c, uint8_t t)
    : prestige_points(points), bonus_color(color), cost(c), tier(t), valid(true) {}

Noble::Noble(std::array<uint8_t, DEV_COLORS> r, uint8_t points)
    : req(r), prestige_points(points) {}

}