cmake_minimum_required(VERSION 3.25)
project(LeMonHashWrapper)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(ext/LeMonHash)

add_library(lemonhash_wrapper STATIC cpp/wrapper.cpp)
target_link_libraries(lemonhash_wrapper PRIVATE LeMonHash)
target_include_directories(lemonhash_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/opt/homebrew/include)
