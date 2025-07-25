#include "include/matrix_parser.hpp"
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

int filter(IDXTYPE row, IDXTYPE col, IDXTYPE nrow, IDXTYPE ncol, int np) {
    int cols_per_process = ceil(ncol / np);
    std::cout << col % cols_per_process << std::endl;
    return col / cols_per_process;
}

// Uso mpirun -np 2 ./bin/exercise_1 /home/matteo/Code/gpu/lessons/06-multi-gpu/multiGPU-SpMV/datasets/mycielskian3.mtx
int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    COO<IDXTYPE, VALTYPE> *coo = (COO<IDXTYPE, VALTYPE> *)my_mtx_parser(argc, argv, "coo", 0, filter, rank, np);
    fprintf(stdout, "Loaded COO matrix with %d non-zeros.\n", coo->nnz);

    for (int i = 0; i < coo->nnz; i++) {
        std::cout << coo->rows_idx[i] << " " << coo->cols_idx[i] << " " << coo->values[i] << std::endl;
    }

    free(coo);

    MPI_Finalize();
    return 0;
}
