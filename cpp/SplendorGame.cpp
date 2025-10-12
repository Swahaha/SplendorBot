#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <cstdint> 

namespace py = pybind11;

enum Color : uint8_t { WHITE=0, BLUE=1, GREEN=2, RED=3, BLACK=4, GOLD=5};
constexpr int DEV_COLORS = 5;
constexpr int COLOR_COUNT = DEV_COLORS + 1;
constexpr int NUM_NOBLES = 5;
constexpr int TIERS = 3;
constexpr int SLOTS_PER_TIER = 4;
constexpr int RESERVED_LIMIT = 3;
constexpr int TOKEN_HAND_LIMIT = 10;
constexpr int PRESTIGE_POINTS_TO_WIN = 15;

struct Card{
    uint8_t prestige_points = 0;
    Color col = WHITE;
    std::array<uint8_t, DEV_COLORS> cost{};
    uint8_t tier = 1;
};

struct Noble{
    std::array<uint8_t, DEV_COLORS> req{};
    uint8_t prestige_points = 0;
};

struct PlayerState{
    std::array<uint8_t, COLOR_COUNT> tokens{};
    uint8_t prestige_points = 0;
    std::vector<Card> reserved;
    std::vector<Card> played_cards;
    std::vector<Noble> nobles_owned;
};

class SplendorGame {
public:
    explicit SplendorGame(int num_players, uint64_t seed = 0) 
        : num_players_(num_players), current_player_(0), rng_(seed)
    {
        if (num_players < 2 || num_players > 4){
            throw std::invalid_argument("num players must be btwn 2 and 4");
        }
        initialize_game();
    }

    int num_players() const { return num_players_; }
    int current_player() const { return current_player_; }

    void end_turn() { current_player_ = (current_player_ + 1) % num_players_; }

    py::dict state_summary() const {
        py::dict d;

        py::list bank;
        for (int c = 0; c < COLOR_COUNT; ++c) bank.append(bank_[c]);
        d["bank"] = bank;

        py::list players_list;
        for (const auto& p : players_) {
            py::dict pd;
            py::list tokens; 
            for (int k=0;k<COLOR_COUNT;++k) tokens.append(p.tokens[k]);
            pd["tokens"] = tokens;
            pd["prestige_points"] = p.prestige_points;
            players_list.append(pd);
        }
        d["players"] = players_list;
        d["current_player"] = current_player_;
        return d;
    }

    int get_legal_moves(){
        return 0;
    }

    void perform_move(){
        current_player_ += 1 % num_players_;
    }

    int deepcopy(){
        return 0;
    }

    void draw_into_slot(int tier /*1..3*/, int slot /*0..SLOTS_PER_TIER-1*/) {
        auto& deck = decks_[tier-1];
        auto& face = market_[tier-1][slot];
        if (deck.empty()) { face = Card{}; return; }
        face = deck.back();
        deck.pop_back();
    }
private:
    int num_players_;
    int current_player_;
    std::array<uint8_t, COLOR_COUNT> bank_{};
    std::array<std::vector<Card>, TIERS> decks_; // the decks
    std::array<std::array<Card, SLOTS_PER_TIER>, TIERS> market_{};
    std::array<Noble, NUM_NOBLES> nobles_{};
    std::vector<PlayerState> players_;
    std::mt19937_64 rng_; // mersenne twister random number

    void initialize_game() {

        uint8_t base = (num_players_ == 2) ? 4 : (num_players_ == 3) ? 5 : 7;
        for (int c=0;c<DEV_COLORS;++c) bank_[c] = base;
        bank_[GOLD] = 5;

        decks_[0] = {
            Card{1, WHITE, { {2,0,0,0,0} }, 1},
            Card{1, BLUE,  { {0,2,0,0,0} }, 1},
            Card{0, GREEN, { {1,1,1,0,0} }, 1},
            Card{0, GREEN, { {1,1,1,0,0} }, 1},
        };
        decks_[1] = {
            Card{1, WHITE, { {2,0,0,0,0} }, 1},
            Card{1, BLUE,  { {0,2,0,0,0} }, 1},
            Card{0, GREEN, { {1,1,1,0,0} }, 1},
            Card{0, GREEN, { {1,1,1,0,0} }, 1},
        };
        decks_[2] = {
            Card{1, WHITE, { {2,0,0,0,0} }, 1},
            Card{1, BLUE,  { {0,2,0,0,0} }, 1},
            Card{0, GREEN, { {1,1,1,0,0} }, 1},
            Card{0, GREEN, { {1,1,1,0,0} }, 1},
        };

        for (auto& d : decks_) std::shuffle(d.begin(), d.end(), rng_);

        for (int t=1; t<=TIERS; ++t)
            for (int s=0; s<SLOTS_PER_TIER; ++s)
                draw_into_slot(t, s);

        nobles_.fill(Noble{});
        nobles_[0] = Noble{ { {3,3,3,0,0} }, 3};
        nobles_[1] = Noble{ { {0,3,3,3,0} }, 3};

        players_.assign(num_players_, PlayerState{});
    }
};

PYBIND11_MODULE(SplendorGame, m) {
    m.doc() = "Minimal Splendor game state core";
    py::class_<SplendorGame>(m, "SplendorGame")
        .def(py::init<int, uint64_t>(), py::arg("num_players"), py::arg("seed") = 0)
        .def("num_players", &SplendorGame::num_players)
        .def("current_player", &SplendorGame::current_player)
        .def("end_turn", &SplendorGame::end_turn)
        .def("state_summary", &SplendorGame::state_summary);
}