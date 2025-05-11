// Random test
#include <iostream>
#include <math.h>

#define MAX_THREAD_PER_BLOCK 1024

// Kernel function to add the elements of two arrays
__global__ void add(float *x, float *y, float *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = x[i] + y[i];
}

int main(void) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = 1 << 25;
    float *x, *y, *res;
    srand(time(0));
    // Allocate Unified Memory accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&res, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = rand() % 50;
        y[i] = rand() % 50;
    }
    // Run kernel on 1M elements on the GPU
    cudaEventRecord(start);
    add<<<N / MAX_THREAD_PER_BLOCK, MAX_THREAD_PER_BLOCK>>>(x, y, res);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    int n_error = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(x[i] + y[i] - res[i]) > 0.1) {
            n_error++;
        }
    }
    std::cout << "N error: " << n_error << std::endl;
    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(res);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Time: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
