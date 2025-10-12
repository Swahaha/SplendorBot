#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <iostream>
#include <random>

namespace py = pybind11;

class SplendorGame {
public:
    // constructor
    explicit SplendorGame(int num_players, int seed = 0) : num_players(num_players), current_player(0), rng(seed)
    {
        if (num_players < 2 || num_players > 4){
            throw std::invalid_argument("num players must be btwn 2 and 4");
        }
        // initialize_game();
    }

    int get_state(){
        return 0;
    }

    int get_legal_moves(){
        return 0;
    }

    int get_players(){
        return 0;
    }

    void perform_move(){
    }

    int deepcopy(){
        return 0;
    }
private:
    int num_players;
    int current_player;
    std::vector<std::unordered_map<std::string, int>> players;
    std::mt19937 rng; // mersenne twister random number

    // void initialize_game() {
    //     for (int i = 0; i < num_players; ++i){
    //         players.push_back({"id", i}, {"score", 0}, {"tokens", 0});
    //     }
    // }
};

PYBIND11_MODULE(SplendorGame, m) {
    py::class_<SplendorGame, std::shared_ptr<SplendorGame>>(m, "SplendorGame")
        .def(py::init<int, int>(), py::arg("num_players"), py::arg("seed") = 0)
        .def("get_state", &SplendorGame::get_state)
        .def("get_legal_moves", &SplendorGame::get_legal_moves)
        .def("get_players", &SplendorGame::get_players)
        .def("perform_move", &SplendorGame::perform_move)
        .def("deepcopy", &SplendorGame::deepcopy);
}