cmake_minimum_required(VERSION 3.15)
project(GrafiteWrapper)

set(CMAKE_CXX_STANDARD 17)

set(GRAFITE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ext/grafite)

# Detect Homebrew prefix
if(EXISTS "/opt/homebrew")
    set(BREW_PREFIX "/opt/homebrew")
else()
    set(BREW_PREFIX "/usr/local")
endif()

add_library(grafite_wrapper STATIC cpp/wrapper.cpp)

# sdsl-lite is header-only (add_custom_target); sux is header-only too
target_include_directories(grafite_wrapper PRIVATE
    ${GRAFITE_ROOT}/include
    ${GRAFITE_ROOT}/lib/sdsl-lite/include
    ${GRAFITE_ROOT}/lib/sux
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BREW_PREFIX}/include
)

target_compile_options(grafite_wrapper PRIVATE -O3 -march=native)
target_compile_definitions(grafite_wrapper PRIVATE SUCCINCT_LIB_SUX)
