cmake_minimum_required(VERSION 3.15)
project(SNARFWrapper)

set(CMAKE_CXX_STANDARD 17)

set(SNARF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ext/SNARF)

if(EXISTS "/opt/homebrew")
    set(BREW_PREFIX "/opt/homebrew")
else()
    set(BREW_PREFIX "/usr/local")
endif()

add_library(snarf_wrapper STATIC cpp/wrapper.cpp)

target_include_directories(snarf_wrapper PRIVATE
    ${SNARF_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BREW_PREFIX}/include
)

target_compile_options(snarf_wrapper PRIVATE -O3 -w -fpermissive)
