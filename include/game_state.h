#pragma once
#include "player_state.h"
#include "constants.h"
#include <random>
#include <vector>
#include <array>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace splendor {

class SplendorGame {
public:
    explicit SplendorGame(int num_players, uint64_t seed = 0);
    
    // Public interface
    int num_players() const { return num_players_; }
    int current_player() const { return current_player_; }
    void end_turn() { current_player_ = (current_player_ + 1) % num_players_; }

    py::dict state_summary() const;
    py::list get_legal_moves() const;
    void perform_move(py::object move_obj);
    bool is_terminal() const;

    // Friend classes for internal operations
    friend class GameInitializer;
    friend class MoveExecutor;
    friend class MoveGenerator;
    friend class StateSerializer;

private:
    // Core game state
    int num_players_;
    int current_player_;
    std::array<uint8_t, COLOR_COUNT> bank_{};
    std::array<std::vector<Card>, TIERS> decks_;
    std::array<std::array<Card, SLOTS_PER_TIER>, TIERS> market_{};
    std::vector<Noble> nobles_;
    std::vector<PlayerState> players_;
    std::mt19937_64 rng_;

    bool check_game_over() const;
};

}