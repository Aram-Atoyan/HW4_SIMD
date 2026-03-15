cmake_minimum_required(VERSION 4.1)
project(HPC C)

set(CMAKE_C_STANDARD 11)

add_executable(HPC
        dna.c
        lowercase.c
        gray.c)
