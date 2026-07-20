# cpp_tokenizer/CMakeLists.txt — build BPETokenizer sebagai Python extension module
cmake_minimum_required(VERSION 3.10)
project(bpe_tokenizer_py)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Ambil path CMake config pybind11 langsung dari instalasi pip
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

pybind11_add_module(bpe_tokenizer_py
    tokenizer_binding.cpp
    bpe_tokenizer.cpp
)