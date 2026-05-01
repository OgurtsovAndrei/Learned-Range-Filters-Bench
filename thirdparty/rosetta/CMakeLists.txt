cmake_minimum_required(VERSION 3.15)
project(RosettaWrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect Homebrew prefix (macOS).
if(EXISTS "/opt/homebrew")
    set(BREW_PREFIX "/opt/homebrew")
else()
    set(BREW_PREFIX "/usr/local")
endif()

set(ROSETTA_EXT ${CMAKE_CURRENT_SOURCE_DIR}/ext)

add_library(rosetta_wrapper STATIC
    cpp/wrapper.cpp
    ${ROSETTA_EXT}/dst.cpp
    ${ROSETTA_EXT}/MurmurHash3.cpp
)

target_include_directories(rosetta_wrapper PRIVATE
    ${ROSETTA_EXT}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BREW_PREFIX}/include
)

target_compile_options(rosetta_wrapper PRIVATE -O3)
