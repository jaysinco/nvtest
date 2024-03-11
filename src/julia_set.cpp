#include "./fwd.cuh"
#include "stb_image_write.h"
#include <stdio.h>

int julia_set(int argc, char** argv)
{
    int image_width = 2560;
    int image_height = 1440;
    int channel_num = 3;
    int pixels_size = image_width * image_height * channel_num;

    uint8_t* pixels = new uint8_t[pixels_size];
    fill_julia_set(image_width, image_height, pixels);
    stbi_write_jpg("julia_set.jpg", image_width, image_height, channel_num, pixels, 100);
    delete[] pixels;

    return 0;
}