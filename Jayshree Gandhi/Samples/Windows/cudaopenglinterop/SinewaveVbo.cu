#include <stdio.h>
#include <math.h>
#include <cuda.h>

__global__ void sinewave_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float animTime)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    float frequency = 4.0f;
    float w = sinf(frequency * u + animTime) * cosf(frequency * v + animTime) * 0.5f;

    pos[y * width + x] = make_float4(u, w, v, 1.0f);
}

void launchCudaKernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time)
{
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);

    sinewave_vbo_kernel<<<grid, block>>>(pos, mesh_width, mesh_height, time);
}
