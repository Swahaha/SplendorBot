cmake_minimum_required(VERSION 3.12)
project(SplendorGame)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python first
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 using Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

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

# Compiler-specific warnings
if(MSVC)
    target_compile_options(splendor_game PRIVATE /W4)
else()
    target_compile_options(splendor_game PRIVATE -Wall -Wextra)
endif()