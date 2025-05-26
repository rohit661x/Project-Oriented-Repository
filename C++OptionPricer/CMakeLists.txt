cmake_minimum_required(VERSION 3.10)
project(MonteCarloOptionPricer)

set(CMAKE_CXX_STANDARD 17) # Or 14, depending on your compiler and preferences

# Define include directories
include_directories(include)

# Add source files
add_executable(MonteCarloOptionPricer
    src/main.cpp
    src/OptionPricing.cpp
)