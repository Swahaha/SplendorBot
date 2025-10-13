#include "game_state.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace splendor;

PYBIND11_MODULE(splendor_game, m) {
    m.doc() = "Splendor game implementation.";
    
    py::class_<SplendorGame>(m, "SplendorGame")
        .def(py::init<int, uint64_t>(), py::arg("num_players"), py::arg("seed") = 0)
        .def("num_players", &SplendorGame::num_players)
        .def("current_player", &SplendorGame::current_player)
        .def("end_turn", &SplendorGame::end_turn)
        .def("state_summary", &SplendorGame::state_summary)
        .def("get_legal_moves", &SplendorGame::get_legal_moves)
        .def("perform_move", &SplendorGame::perform_move)
        .def("is_terminal", &SplendorGame::is_terminal);
}