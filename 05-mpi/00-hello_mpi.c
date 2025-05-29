#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
	// Initialize MPI
	MPI_Init(&argc, &argv);
	int rank, size;
	// Get process ID
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Get total number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("%d %d\n", rank, size);

	MPI_Finalize();
	return 0;
}
