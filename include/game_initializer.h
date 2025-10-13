#pragma once
#include "game_state.h"

namespace splendor {

class GameInitializer {
public:
    static void InitializeGame(SplendorGame& game);
    static void DrawIntoSlot(SplendorGame& game, int tier, int slot);
private:
    static void InitializeBank(SplendorGame& game);
    static void InitializeDecks(SplendorGame& game);
    static void FillMarket(SplendorGame& game);
    static void InitializeNobles(SplendorGame& game);
    static void InitializePlayers(SplendorGame& game);
};

} 