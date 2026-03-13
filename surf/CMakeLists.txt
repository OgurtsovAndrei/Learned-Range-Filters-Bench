cmake_minimum_required(VERSION 3.15)
project(SuRFWrapper)

set(CMAKE_CXX_STANDARD 17)

set(SURF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ext/SuRF)

if(EXISTS "/opt/homebrew")
    set(BREW_PREFIX "/opt/homebrew")
else()
    set(BREW_PREFIX "/usr/local")
endif()

# SuRF is header-only; only the wrapper needs compiling
add_library(surf_wrapper STATIC cpp/wrapper.cpp)

target_include_directories(surf_wrapper PRIVATE
    # ARM shims MUST come before system includes to shadow emmintrin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/arm_shims
    ${SURF_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BREW_PREFIX}/include
)

target_compile_options(surf_wrapper PRIVATE -O3 -Wno-register)
