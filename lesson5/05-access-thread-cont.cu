// ES 6.1
#include <iostream>
#include <math.h>

#define MAX_THREAD_PER_BLOCK 1024
#define MAX_WARP 32
#define N_BLOCK 256

#define CYCLES 64
#define WARMUP_CYCLES 32

// Kernel function to add the elements of two arrays
__global__ void add_B(float *x, float *y, float *res, int len) {
    int n_threads = blockDim.x * gridDim.x;
    int action_per_thread = len / n_threads;
    int i_start = blockDim.x * blockIdx.x + threadIdx.x;
    int start = action_per_thread * i_start;

    for (int i = start; i < start + action_per_thread; i++) {
        res[i] = x[i] + y[i];
    }
}

// Kernel function to add the elements of two arrays
__global__ void add_A(float *x, float *y, float *res) {
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
    double sum_times = 0;

    // Allocate Unified Memory accessible from CPU or GPU
    for (int cycle = -WARMUP_CYCLES; cycle < CYCLES; cycle++) {
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
        add_B<<<N_BLOCK, MAX_THREAD_PER_BLOCK>>>(x, y, res, N);
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

        if (n_error > 0) {
            std::cout << "There are " << n_error << " errors in the array (cycle " << cycle << ")" << std::endl;
            break;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (cycle >= 0) {
            printf("Kernel Time (id %d): %f ms\n", cycle, milliseconds);
            sum_times += milliseconds;
        }
    }

    printf("\nKernel Time (average): %f ms\n\n\n", sum_times / CYCLES);

    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x);
    cudaFree(y);
    cudaFree(res);
}
