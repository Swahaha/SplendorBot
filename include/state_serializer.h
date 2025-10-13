#pragma once
#include "game_state.h"

namespace splendor {

class StateSerializer {
public:
    static py::dict StateSummary(const SplendorGame& game);
    static py::dict CardToDict(const Card& card);
    static py::list CardsToList(const std::vector<Card>& cards);
    static py::dict NobleToDict(const Noble& noble);
    
private:
    static void SerializeBank(const SplendorGame& game, py::dict& summary);
    static void SerializeMarket(const SplendorGame& game, py::dict& summary);
    static void SerializeNobles(const SplendorGame& game, py::dict& summary);
    static void SerializePlayers(const SplendorGame& game, py::dict& summary);
};

} 