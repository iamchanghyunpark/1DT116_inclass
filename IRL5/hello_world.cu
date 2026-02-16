// File to prepare at home. (Making sure CUDA is available, and
// you can use it on your machine before the session.

#include <stdio.h>
#include <stdlib.h>
#include "common_cuda.h"

//The very basic kernel function. Notice the conditional statement! Why is
//that needed?
__global__ void increment_by_n_minus_i(float *array, size_t N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
        array[i] += N-i;
}

int main () {
    const int N = 64;
    //const int N = 2*1024*1024;
    float *array;
    float *cu_array;
    if (cudaMallocHost(&array, sizeof(float)*N) != cudaSuccess) {
        printf("Error allocating host memory for the array\n");
        return 1;
    }

    if (cudaMalloc(&cu_array, sizeof(float)*N) != cudaSuccess) {
        printf("Error allocating GPU memory for the array\n");
        return 1;
    }

    printf("Filling an array of N=%d floating point elements\n", N);

    for (int i = 0; i < N; i++) {
        array[i] = i;
    }

    printf("Array content before: [");
    for (int i = 0; i < N; i++) {
        printf("%3.1f%s", array[i], i==(N-1) ? "]\n":", ");
    }

    printf("Now incrementing each value of the element by N-i! (everything should be equal to N \n");

    //Copying from the host array to device cu_array.
    if (cudaMemcpy(cu_array, array, N*sizeof(float), cudaMemcpyHostToDevice)
            != cudaSuccess) {
        printf("Copying from host to device failed\n");
        return 1;
    }

    //Setup the kernel call (grid) dimensions
    int threadPerBlock = 1024;
    int numBlocks = N / threadPerBlock;
    if (N % threadPerBlock)
        numBlocks++;

    //Let the device to the work on the cu_array
    increment_by_n_minus_i<<<numBlocks, threadPerBlock>>>(cu_array, N);

    //Copying from device cu_array back to host array.
    if (cudaMemcpy(array, cu_array, N*sizeof(float), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
        printf("Copying from device to host failed\n");
        return 1;
    }

    printf("Array content after: [");
    for (int i = 0; i < N; i++) {
        printf("%3.1f%s", array[i], i==(N-1) ? "]\n":", ");
    }

    return 0;
}
