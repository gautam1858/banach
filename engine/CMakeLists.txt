cmake_minimum_required(VERSION 3.16)
project(banach-engine CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_DIR "" CACHE PATH "Path to a llama.cpp checkout")
if(NOT LLAMA_DIR)
  message(FATAL_ERROR "Set -DLLAMA_DIR=/path/to/llama.cpp (see engine/README.md for the tested commit)")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
add_executable(banach-engine banach-engine.cpp)
target_link_libraries(banach-engine PRIVATE llama ggml Threads::Threads)
target_compile_options(banach-engine PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
