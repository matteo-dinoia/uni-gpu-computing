#include <mpi.h>
#include <stdio.h>
#define LEN 5

int main(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int other[size] = {};

    int dest, tag = 0, source;
    MPI_Status status;
    int partial_sum = 0, bkp = 0;
    int sum = 0;
    if (rank == 0) {
        int arr[LEN] = { 1, 2, 3, 4, 5 };
        for (int i = 0; i < LEN; i++) {
            partial_sum += arr[i];
        }

    } else {
        int arr[LEN] = { 2, 3, 4, 5, 6 };
        for (int i = 0; i < LEN; i++) {
            partial_sum += arr[i];
        }
    }

    printf("Partial sum %d from process %d\n", partial_sum, rank);
    MPI_Allgather(&partial_sum, 1, MPI_INT, &other, 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < size; i++) {
        sum += other[i];
    }

    printf("Sum is %d, %d from process %d\n", partial_sum, sum, rank);

    MPI_Finalize();
}
