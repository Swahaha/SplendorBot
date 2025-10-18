#include "types.h"

namespace splendor {

Card::Card(uint8_t points, Color color, std::array<uint8_t, DEV_COLORS> c, uint8_t t, uint8_t id)
    : prestige_points(points), bonus_color(color), cost(c), tier(t), id(id), valid(true) {}

Noble::Noble(std::array<uint8_t, DEV_COLORS> r, uint8_t points)
    : req(r), prestige_points(points) {}

}