#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char message[100];
    int dest, tag = 0, source;
    MPI_Status status;
    if (rank == 0) {
        sprintf(message, "Greetings from process %d !\0", rank);
        MPI_Send(message, sizeof(message), MPI_CHAR, 1, tag, MPI_COMM_WORLD);
    } else {
        MPI_Recv(message, 100, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
        printf("%s\n", message);
    }
    MPI_Finalize();
}
