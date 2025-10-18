#pragma once
#include "game_state.h"
#include "state_serializer.h"

namespace splendor {

class MoveGenerator {
public:
    static py::list GetLegalMoves(const SplendorGame& game);
    
private:
    static void GenerateBuyFromMarketMoves(const SplendorGame& game, py::list& moves);
    static void GenerateBuyFromReserveMoves(const SplendorGame& game, py::list& moves);
    static void GenerateReserveCardMoves(const SplendorGame& game, py::list& moves);
    static void GenerateTakeTokensMoves(const SplendorGame& game, py::list& moves);
};

} 