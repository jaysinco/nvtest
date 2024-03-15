add_executable(nvtest
    main.cpp
    stb_image_write.cpp
    stb_image.cpp

    hello_world.cu
    check_device.cu
    sum_matrix.cu
    reduce_integer.cu
    nested_hello_world.cu
    global_variable.cu
    test_cufft.cu
    julia_set.cu
    dot_product.cu
    ray_tracing.cu
    txi_gaussian.cu
)
target_link_libraries(nvtest PUBLIC
    CUDA::cudart_static
    CUDA::cufft
)

add_executable(deviceQuery
    deviceQuery.cpp
)
target_link_libraries(deviceQuery PUBLIC
    CUDA::cudart_static
)
