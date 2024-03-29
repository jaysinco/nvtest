cmake_minimum_required(VERSION 3.19)

project(nvtest LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

set(MY_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

if(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    add_link_options(-rdynamic)
elseif(WIN32)
    add_compile_definitions(
        WINVER=0x0601
        _WIN32_WINNT=0x0602
        NOMINMAX
    )
    # add_compile_options(/utf-8 /permissive-)
endif()

add_subdirectory(src)
