cmake_minimum_required(VERSION 3.12)
project(SplendorGame)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

include_directories(include)

pybind11_add_module(splendor_game
    src/player_state.cpp
    src/game_state.cpp
    src/game_initializer.cpp
    src/move_executor.cpp
    src/move_generator.cpp
    src/state_serializer.cpp
    src/types.cpp
    src/csv_parser.cpp
    src/bindings.cpp
)

target_compile_options(splendor_game PRIVATE -Wall -Wextra)