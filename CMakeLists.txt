cmake_minimum_required(VERSION 3.24)
project(paterson-stockmeyer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures BEFORE any target uses CUDA.
# "90a" = Hopper arch-specific (TMA/wgmma), "100a" = Blackwell arch-specific.
# "90-real"/"100-real" would exclude PTX; plain "90"/"100" include PTX for forward compat.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90;90a;100;100a" CACHE STRING "CUDA architectures to build for")
endif()

add_subdirectory(benchmarks)