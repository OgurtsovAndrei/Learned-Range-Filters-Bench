cmake_minimum_required(VERSION 3.15)
project(SNARFWrapper)

set(CMAKE_CXX_STANDARD 17)

set(SNARF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ext/SNARF)

add_library(snarf_wrapper STATIC cpp/wrapper.cpp)

target_include_directories(snarf_wrapper PRIVATE
    ${SNARF_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(snarf_wrapper PRIVATE -O3 -w -fpermissive)
