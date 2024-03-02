
add_executable(nvtest
    main.cpp
    hello_world.cu
    check_device.cu
    sum_matrix.cu
    reduce_integer.cu
    nested_hello_world.cu
    global_variable.cu
)
target_link_libraries(nvtest PUBLIC
    CUDA::cudart
)

add_executable(deviceQuery
    deviceQuery.cpp
)
target_link_libraries(deviceQuery PUBLIC
    CUDA::cudart
)
