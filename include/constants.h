#pragma once
#include <cstdint>

namespace splendor {

constexpr int DEV_COLORS = 5; // number of dev token colors
constexpr int COLOR_COUNT = DEV_COLORS + 1; // total colors also includes gold
constexpr int NUM_NOBLES = 5; // num nobles in play
constexpr int TIERS = 3;
constexpr int SLOTS_PER_TIER = 4;
constexpr int RESERVED_LIMIT = 3;
constexpr int TOKEN_HAND_LIMIT = 10; // how many tokens you are allowed in hand
constexpr int PRESTIGE_POINTS_TO_WIN = 15; 

enum Color : uint8_t { WHITE=0, BLUE=1, GREEN=2, RED=3, BLACK=4, GOLD=5 };
enum ACTIONS : uint8_t { 
    RESERVE_CARD=0, 
    BUY_CARD_FROM_MARKET=1, 
    BUY_CARD_FROM_RESERVE=2, 
    GET_3_FROM_BANK=3, 
    GET_2_FROM_BANK=4 
};

}