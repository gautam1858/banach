cmake_minimum_required(VERSION 3.16)
project(banach-trace CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_DIR "" CACHE PATH "Path to a llama.cpp checkout")
if(NOT LLAMA_DIR)
  message(FATAL_ERROR "Set -DLLAMA_DIR=/path/to/llama.cpp (see tracer/README.md for the pinned commit)")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)

add_executable(banach-trace banach-trace.cpp)
target_link_libraries(banach-trace PRIVATE llama ggml)
target_include_directories(banach-trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../engine)
target_compile_options(banach-trace PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
