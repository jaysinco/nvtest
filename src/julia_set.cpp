#include "./fwd.cuh"
#include "stb_image_write.h"
#include <stdio.h>

int julia_set(int argc, char** argv)
{
    int image_size = 720;
    int channel_num = 3;
    char* pixels = new char[image_size * image_size * channel_num];

    int index = 0;
    for (int j = image_size - 1; j >= 0; --j) {
        for (int i = 0; i < image_size; ++i) {
            float r = (float)i / (float)image_size;
            float g = (float)j / (float)image_size;
            float b = 0.2f;
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);

            pixels[index++] = ir;
            pixels[index++] = ig;
            pixels[index++] = ib;
        }
    }

    stbi_write_jpg("julia_set.jpg", image_size, image_size, channel_num, pixels, 100);

    delete[] pixels;

    return 0;
}