#include <mpi.h>
#include <stdio.h>

#include <sys/time.h>

#define TIMER_DEF(n) struct timeval temp_1_##n = { 0, 0 }, temp_2_##n = { 0, 0 }
#define TIMER_START(n) gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n) gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_TIME(n, op) \
    TIMER_START(n);       \
    op;                   \
    TIMER_STOP(n)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec - temp_1_##n.tv_sec) * 1.e6 + (temp_2_##n.tv_usec - temp_1_##n.tv_usec))

int main(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dest, tag = 0, source;
    MPI_Status status;
    if (rank == 0) {

        TIMER_DEF(0);
        TIMER_TIME(0, {
            MPI_Send(NULL, 0, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(NULL, 0, MPI_CHAR, 1, tag, MPI_COMM_WORLD, &status);
        });
        printf("%fms", TIMER_ELAPSED(0) / 1e3 / 2);

    } else {
        MPI_Recv(NULL, 0, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
        MPI_Send(NULL, 0, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
