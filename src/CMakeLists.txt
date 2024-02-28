
add_executable(nvtest
    main.cpp
    hello_world.cu
    check_device.cu
    sum_matrix.cu
    reduce_integer.cu
    nested_hello_world.cu
    global_variable.cu
)

add_executable(deviceQuery
    deviceQuery.cpp
)

target_include_directories(deviceQuery PUBLIC
    /usr/local/cuda/include/
)

target_link_libraries(deviceQuery PUBLIC
    /usr/local/cuda/lib64/libcudart.so
)