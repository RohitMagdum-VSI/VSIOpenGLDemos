#include "cuda_runtime.h"
#include "./cuda_kernel.cuh"

__global__ void vectorAdditionKernel(double* A, double* B, double* C, int arraySize) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < arraySize) {
        // Add a and b.
        C[threadID] = A[threadID] + B[threadID];
    }
}

__global__ void calculate_vertices(float4* pos, unsigned int width, unsigned int height, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    __syncthreads();
    float v = y / (float)height;
    __syncthreads();
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    float freq = 4.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    pos[y * width + x] = make_float4(u, w, v, 1.0f);
}


/**
 * Wrapper function for the CUDA kernel function.
 * @param A Array A.
 * @param B Array B.
 * @param C Sum of array elements A and B directly across.
 * @param arraySize Size of arrays A, B, and C.
 */
//void kernel(double* A, double* B, double* C, int arraySize) 
void kernel(int mesh_width, int mesh_height, void *pVbo, float fAnimate)
{

    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //MessageBox(NULL,TEXT("RunCuda Before Kernel"),TEXT("MSG"),MB_OK);
    calculate_vertices <<< grid, block >>> ((float4 *)pVbo, mesh_width, mesh_height, fAnimate);

}
