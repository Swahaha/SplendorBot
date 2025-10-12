cmake_minimum_required(VERSION 3.15)
project(SplendorBot)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add the Python module
pybind11_add_module(SplendorGame ${CMAKE_SOURCE_DIR}/cpp/SplendorGame.cpp)