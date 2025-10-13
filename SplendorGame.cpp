#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <cstdint> 

namespace py = pybind11;

enum Color : uint8_t { WHITE=0, BLUE=1, GREEN=2, RED=3, BLACK=4, GOLD=5};
enum ACTIONS : uint8_t { 
    RESERVE_CARD=0, 
    BUY_CARD_FROM_MARKET=1, 
    BUY_CARD_FROM_RESERVE=2, 
    GET_3_FROM_BANK=3, 
    GET_2_FROM_BANK=4
};

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
    Color bonus_color = WHITE;
    std::array<uint8_t, DEV_COLORS> cost{};
    uint8_t tier = 1;

    Card(uint8_t points, Color color, std::array<uint8_t, DEV_COLORS> c, uint8_t t) 
        : prestige_points(points), bonus_color(color), cost(c), tier(t) {}
};

struct Noble{
    std::array<uint8_t, DEV_COLORS> req{};
    uint8_t prestige_points = 0;

    Noble(std::array<uint8_t, DEV_COLORS> r, uint8_t points) 
        : req(r), prestige_points(points) {}
};

struct PlayerState{
    std::array<uint8_t, COLOR_COUNT> tokens{};
    uint8_t prestige_points = 0;
    std::vector<Card> reserved;
    std::vector<Card> played_cards;
    std::vector<Noble> nobles_owned;

    int total_tokens() const {
        int sum = 0;
        for (auto t : tokens) sum += t;
        return sum;
    }

    std::array<uint8_t, DEV_COLORS> get_bonuses() const {
        std::array<uint8_t, DEV_COLORS> bonuses{};
        for (const auto& card : played_cards) bonuses[static_cast<int>(card.bonus_color)]++;
        return bonuses;
    }
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
            for (int k = 0; k < COLOR_COUNT; ++k) tokens.append(p.tokens[k]);
            pd["tokens"] = tokens;
            pd["prestige_points"] = p.prestige_points;
            pd["reserved"] = cards_to_list(p.reserved);
            pd["played"] = cards_to_list(p.played_cards);

            players_list.append(pd);
        }
        d["players"] = players_list;

        d["current_player"] = current_player_;
        return d;
    }

    py::list get_legal_moves() const {
        py::list moves;
        const auto& me = players_[current_player_];
        const int hand = token_sum(me);
        const auto disc = discounts_of(me);

        // ---- BUY from market: (1, {"tier": t, "slot": s}) ----
        for (int t=1; t<=TIERS; ++t){
            for (int s=0; s<SLOTS_PER_TIER; ++s){
                const Card& c = market_[t-1][s];
                if (!slot_present(c)) continue;
                uint8_t gold_needed=0; std::array<uint8_t,DEV_COLORS> pay{};
                if (can_afford(me, c, disc, gold_needed, pay)){
                    py::dict payload;
                    payload["tier"] = t;
                    payload["slot"] = s;
                    moves.append(py::make_tuple((int)BUY_CARD_FROM_MARKET, payload));
                }
            }
        }

        // ---- BUY from reserve: (2, {"index": i, "card": {...}}) ----
        for (size_t i=0; i<me.reserved.size(); ++i){
            const Card& c = me.reserved[i];
            uint8_t gold_needed=0; std::array<uint8_t,DEV_COLORS> pay{};
            if (can_afford(me, c, disc, gold_needed, pay)){
                py::dict payload;
                payload["index"] = (int)i;
                payload["card"] = card_to_dict(c);
                moves.append(py::make_tuple((int)BUY_CARD_FROM_RESERVE, payload));
            }
        }

        // ---- RESERVE a face-up card: (0, {"tier": t, "slot": s}) ----
        if (me.reserved.size() < RESERVED_LIMIT){
            for (int t=1; t<=TIERS; ++t){
                for (int s=0; s<SLOTS_PER_TIER; ++s){
                    const Card& c = market_[t-1][s];
                    if (!slot_present(c)) continue;
                    py::dict payload;
                    payload["tier"] = t;
                    payload["slot"] = s;
                    moves.append(py::make_tuple((int)RESERVE_CARD, payload));
                }
            }
            // (Reserve-from-top is easy to add later)
        }

        // ---- GET 3 distinct tokens: (3, [w,b,g,r,blk]) ----
        // Only if the bank has those colors and the hand limit allows +3
        if (hand <= TOKEN_HAND_LIMIT - 3){
            for (int a=0; a<DEV_COLORS; ++a){
                if (bank_[a] == 0) continue;
                for (int b=a+1; b<DEV_COLORS; ++b){
                    if (bank_[b] == 0) continue;
                    for (int c=b+1; c<DEV_COLORS; ++c){
                        if (bank_[c] == 0) continue;
                        py::list vec;
                        for (int k=0;k<DEV_COLORS;++k) vec.append( (k==a || k==b || k==c) ? 1 : 0 );
                        moves.append(py::make_tuple((int)GET_3_FROM_BANK, vec));
                    }
                }
            }
        }

        // ---- GET 2 of the same color: (4, [w,b,g,r,blk]) ----
        // Only if bank[color] >= 4 and hand limit allows +2
        if (hand <= TOKEN_HAND_LIMIT - 2){
            for (int c=0; c<DEV_COLORS; ++c){
                if (bank_[c] >= 4){
                    py::list vec;
                    for (int k=0;k<DEV_COLORS;++k) vec.append(k==c ? 2 : 0);
                    moves.append(py::make_tuple((int)GET_2_FROM_BANK, vec));
                }
            }
        }

        return moves;
    }

    void perform_move(){
        end_turn();
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
        initialize_bank();
        initialize_decks();
        fill_in_market();
        initialize_nobles();
        initialize_players();
    }

    void initialize_bank() {
        int token_count = (num_players_ == 2) ? 4 : (num_players_ == 3) ? 5 : 7;
        for (int i = 0; i < DEV_COLORS; ++i) bank_[i] = token_count;
        bank_[GOLD] = 5;
    }

    void initialize_decks() {
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
    }

    void fill_in_markets(){
        for (auto& d : decks_) std::shuffle(d.begin(), d.end(), rng_);

        for (int t=1; t<=TIERS; ++t)
            for (int s=0; s<SLOTS_PER_TIER; ++s)
                draw_into_slot(t, s);
    }

    void initialize_nobles(){
        nobles_.fill(Noble{});
        nobles_[0] = Noble{ { {3,3,3,0,0} }, 3};
        nobles_[1] = Noble{ { {0,3,3,3,0} }, 3};
    }

    void initialize_players(){
        players_.assign(num_players_, PlayerState{});
    }

    static py::dict card_to_dict(const Card& c) {
        py::dict cd;
        cd["points"] = c.points;
        cd["bonus_color"] = c.bonus_color;   // 0..4 (color of permanent discount)
        cd["tier"] = c.tier;                 // 1..3
        py::list cost;
        for (int k = 0; k < DEV_COLORS; ++k) cost.append(c.cost[k]);
        cd["cost"] = cost;
        cd["valid"] = c.valid;
        return cd;
    }

    static py::list cards_to_list(const std::vector<Card>& v) {
        py::list out;
        out.attr("reserve")(v.size());       // optional prealloc hint to CPython
        for (const auto& c : v) out.append(card_to_dict(c));
        return out;
    }
};

PYBIND11_MODULE(SplendorGame, m) {
    m.doc() = "Minimal Splendor game state core";
    py::class_<SplendorGame>(m, "SplendorGame")
        .def(py::init<int, uint64_t>(), py::arg("num_players"), py::arg("seed") = 0)
        .def("num_players", &SplendorGame::num_players)
        .def("current_player", &SplendorGame::current_player)
        .def("end_turn", &SplendorGame::end_turn)
        .def("state_summary", &SplendorGame::state_summary)
        .def("get_legal_moves", &SplendorGame::get_legal_moves);
}