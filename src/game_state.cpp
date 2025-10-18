#include "game_state.h"
#include "game_initializer.h"
#include "move_executor.h"
#include "move_generator.h"
#include "state_serializer.h"

namespace splendor {

SplendorGame::SplendorGame(int num_players, uint64_t seed) 
    : num_players_(num_players), current_player_(0), rng_(seed) {
    if (num_players < 2 || num_players > 4) throw std::invalid_argument("num players must be between 2 and 4");
    GameInitializer::InitializeGame(*this);
}

py::dict SplendorGame::state_summary() const {
    return StateSerializer::StateSummary(*this);
}

py::list SplendorGame::legal_moves() const {
    return MoveGenerator::GetLegalMoves(*this);
}

void SplendorGame::perform_move(py::object move_obj) {
    MoveExecutor::PerformMove(*this, move_obj);
}

bool SplendorGame::is_terminal() const {
    return check_game_over();
}

bool SplendorGame::check_game_over() const {
    for (const auto& player : players_) {
        if (player.prestige_points >= PRESTIGE_POINTS_TO_WIN) {
            return true;
        }
    }
    return false;
}

}