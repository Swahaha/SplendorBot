#pragma once
#include "game_state.h"

namespace splendor {

class MoveExecutor {
public:
    static void PerformMove(SplendorGame& game, py::object move_obj);
    
private:
    static void ReserveCard(SplendorGame& game, int tier, int slot);
    static void BuyCardFromMarket(SplendorGame& game, int tier, int slot);
    static void BuyCardFromReserve(SplendorGame& game, int index);
    static bool PurchaseCard(SplendorGame& game, PlayerState& player, const Card& card);
    static void TakeTokens3(SplendorGame& game, const std::array<uint8_t, DEV_COLORS>& tokens);
    static void TakeTokens2(SplendorGame& game, const std::array<uint8_t, DEV_COLORS>& tokens);
    static void CheckNobles(SplendorGame& game);
    
    static bool CanAfford(const PlayerState& player, const Card& card, 
                         const std::array<uint8_t, DEV_COLORS>& discounts);
};

} // namespace splendor