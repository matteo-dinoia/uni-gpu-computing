// #define ENABLE_NVTX
// #define ENABLE_CPU_BASELINE
// #define DEBUG_PRINTS
#include <driver_types.h>
#include <iterator>
#include <sys/types.h>
#define ENABLE_CORRECTNESS_CHECK

#define EXIT_INCORRECT_DISTANCES 10

#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "../distributed_mmio/include/mmio.h"
#include "../distributed_mmio/include/mmio_utils.h"

#include "../include/bfs_baseline.cuh"
#include "../include/cli.hpp"
#include "../include/colors.h"
#include "../include/mt19937-64.hpp"
#include "../include/utils.cuh"

__global__ void bfs_kernel_push(
    const uint32_t M,
    const uint32_t* row_indices, // CSR row offsets
    const uint32_t* col_indices, // CSR column indices (neighbors)
    int* distances, // Output distances array
    const uint32_t current_level, // BFS level (depth)
    uint32_t* cont // BFS level (depth)
)
{
    const int n_threads = gridDim.x * blockDim.x;
    const int per_thread = (int)ceil(M / (float)n_threads);
    const int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    const int wrap_start_id = start_i / warpSize * warpSize;
    const int start = wrap_start_id * per_thread + (start_i - wrap_start_id);
    const int incr = warpSize;

    uint32_t flag = 0;
    for (int i = 0; i < per_thread; i++) {
        const int edge = start + i * incr;
        if (edge >= M)
            break;

        const uint32_t node = row_indices[edge];
        if (distances[node] != current_level)
            continue;

        const uint32_t neighbor = col_indices[edge];
        if (distances[neighbor] == -1) {
            distances[neighbor] = current_level + 1;
            flag = 1;
        }
    }

    if (flag == 1 && cont[0] == 0) {
        cont[0] = 1;
    }
}

void gpu_bfs(
    const uint32_t N,
    const uint32_t M,
    const uint32_t* h_rowidx,
    const uint32_t* h_colidx,
    const uint32_t source,
    int* h_distances)
{
    float tot_time = 0.0;
    CUDA_TIMER_INIT(H2D_copy)

    // Allocate and copy graph to device
    uint32_t* d_row_indices;
    uint32_t* d_col_indices;
    CHECK_CUDA(cudaMalloc(&d_row_indices, M * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_row_indices, h_rowidx, M * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate memory for distances and frontier queues
    int* d_distances;
    uint32_t* d_cont;
    CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_distances + source, 0, sizeof(int))); // set to 0
    CHECK_CUDA(cudaMalloc(&d_cont, sizeof(uint32_t)));

    CUDA_TIMER_STOP(H2D_copy)
#ifdef DEBUG_PRINTS
    CUDA_TIMER_PRINT(H2D_copy)
#endif
    tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
    CUDA_TIMER_DESTROY(H2D_copy)

    uint32_t level = 0;
    uint32_t cont = 1;

    uint32_t block_size = 512;
    //        uint32_t num_blocks = 256;
    uint32_t num_blocks = 256;
    // CEILING(M, block_size) / 256;

    // Main BFS loop
    CPU_TIMER_INIT(BASELINE_BFS)
    while (cont) {

#ifdef DEBUG_PRINTS
        printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n", is_placeholder ? "" : " BASELINE", level, current_frontier_size);
#endif
#ifdef ENABLE_NVTX
        // Mark start of level in NVTX
        nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
#endif
        CHECK_CUDA(cudaMemset(d_cont, 0, sizeof(uint32_t)));
        bfs_kernel_push<<<num_blocks, block_size>>>(
            M,
            d_row_indices,
            d_col_indices,
            d_distances,
            level,
            d_cont);

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&cont, d_cont, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        level++;

#ifdef ENABLE_NVTX
        // End NVTX range for level
        nvtxRangePop();
#endif
    }
    CPU_TIMER_STOP(BASELINE_BFS)
#ifdef DEBUG_PRINTS
    CPU_TIMER_PRINT(BASELINE_BFS)
#endif
    tot_time += CPU_TIMER_ELAPSED(BASELINE_BFS);

    CUDA_TIMER_INIT(D2H_copy)
    CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TIMER_STOP(D2H_copy)
#ifdef DEBUG_PRINTS
    CUDA_TIMER_PRINT(D2H_copy)
#endif
    tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
    CUDA_TIMER_DESTROY(D2H_copy);

    printf("\n[OUT] Total BFS time: %f ms\n" RESET, tot_time);

    // Free device memory
    cudaFree(d_row_indices);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
    cudaFree(d_cont);
}

int main(int argc, char** argv)
{
    int return_code = EXIT_SUCCESS;

    Cli_Args args;
    init_cli();
    if (parse_args(argc, argv, &args) != 0) {
        return -1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        fprintf(stderr, "No GPU available: device_count=%d\n", device_count);
        return EXIT_FAILURE;
    }
    cudaSetDevice(0);

    CPU_TIMER_INIT(MTX_read)
    COO_local<uint32_t, float>* coo = Distr_MMIO_COO_local_read<uint32_t, float>(args.filename);
    CSR_local<uint32_t, float>* csr = Distr_MMIO_CSR_local_read<uint32_t, float>(args.filename);
    if (coo == NULL || csr == NULL) {
        printf("Failed to import graph from file [%s]\n", args.filename);
        return -1;
    }
    CPU_TIMER_STOP(MTX_read)
    printf("\n[OUT] MTX file read time: %f ms\n", CPU_TIMER_ELAPSED(MTX_read));
    printf("Graph size: %.3fM vertices, %.3fM edges\n", coo->nrows / 1e6, coo->nnz / 1e6);

    GraphCSR graph;
    graph.row_ptr = csr->row_ptr;
    graph.col_idx = csr->col_idx;
    graph.num_vertices = csr->nrows;
    graph.num_edges = csr->nnz;
    // print_graph_csr(graph);

    uint32_t* sources = generate_sources(&graph, args.runs, graph.num_vertices, args.source);
    int* distances_gpu_baseline = (int*)malloc(graph.num_vertices * sizeof(int));
    int* distances = (int*)malloc(graph.num_vertices * sizeof(int));
    bool correct = true;

    for (int source_i = 0; source_i < args.runs; source_i++) {
        uint32_t source = sources[source_i];
        printf("\n[OUT] -- BFS iteration #%u, source=%u --\n", source_i, source);

        // Run the BFS baseline
        gpu_bfs_baseline(graph.num_vertices, graph.num_edges, graph.row_ptr, graph.col_idx, source, distances_gpu_baseline, false);

#ifdef ENABLE_NVTX
        nvtxRangePushA("Complete BFS");
#endif
        gpu_bfs(graph.num_vertices, graph.num_edges, coo->row, coo->col, source, distances);
#ifdef ENABLE_NVTX
        nvtxRangePop();
#endif

        bool match = true;
#ifdef ENABLE_CORRECTNESS_CHECK
        for (uint32_t i = 0; i < graph.num_vertices; ++i) {
            if (distances_gpu_baseline[i] != distances[i]) {
                printf("Mismatch at node %u: Baseline distance = %d, Your distance = %d\n", i, distances_gpu_baseline[i], distances[i]);
                match = false;
                break;
            }
        }
        if (match) {
            printf(BRIGHT_GREEN "Correctness OK\n" RESET);
        } else {
            printf(BRIGHT_RED "GPU and CPU BFS results do not match for source node %u.\n" RESET, source);
            return_code = EXIT_INCORRECT_DISTANCES;
            correct = false;
        }
#endif

#ifdef ENABLE_CPU_BASELINE
        int cpu_distances[graph.num_vertices];

        CPU_TIMER_INIT(CPU_BFS)
        cpu_bfs_baseline(graph.num_vertices, graph.row_ptr, graph.col_idx, source, cpu_distances);
        CPU_TIMER_CLOSE(CPU_BFS)

        match = true;
        for (uint32_t i = 0; i < graph.num_vertices; ++i) {
            if (distances_gpu_baseline[i] != cpu_distances[i]) {
                printf("Mismatch at node %u: GPU distance = %d, CPU distance = %d\n", i, distances_gpu_baseline[i], cpu_distances[i]);
                match = false;
                break;
            }
        }
        if (match) {
            printf(BRIGHT_GREEN "[CPU] Correctness OK\n" RESET);
        } else {
            printf(BRIGHT_RED "GPU and CPU BFS results do not match for source node %u.\n" RESET, source);
            return_code = EXIT_INCORRECT_DISTANCES;
        }
#endif
    }

    if (correct)
        printf("\n[OUT] ALL RESULTS ARE CORRECT\n");
    else
        printf(BRIGHT_RED "\nSOME RESULTS ARE WRONG\n" RESET);

    Distr_MMIO_COO_local_destroy(&coo);
    free(sources);
    free(distances_gpu_baseline);
    free(distances);

    return return_code;
}
