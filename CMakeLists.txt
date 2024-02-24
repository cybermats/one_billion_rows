cmake_minimum_required(VERSION 3.28)
project(one_billion_rows C)

set(CMAKE_C_STANDARD 23)

if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_compile_options(-g -pg)

add_executable(one_billion_rows main.c)
