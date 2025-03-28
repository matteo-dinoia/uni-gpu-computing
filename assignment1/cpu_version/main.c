#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <strings.h>

#include "include/my_time_lib.h"

#define M_TYPE double
#define OK true
#define ERR false

#define WARM_CYCLES 10
#define TEST_CYCLES 40

// Function to skip comments and read metadata from the MTX file
bool read_mtx_header(FILE *file, int *rows, int *cols, int *non_zeros)
{
    int ch;

    // Skip any initial whitespace or comments
    while ((ch = fgetc(file)) == '%') {
        while ((ch = fgetc(file)) != EOF && ch != '\n');  // Discards the entire line.
    }
    ungetc(ch, file);

    // Read the matrix dimensions and non-zero entries
    if (fscanf(file, "%d %d %d", rows, cols, non_zeros) != 3) {
        return false;
    }

    return true;
}

bool read_mtx_data(FILE *file, int *coords_x, int *coords_y, double *vals, int LEN)
{
    if (coords_x == NULL ||  coords_y == NULL || vals == NULL) {
        printf("Null pointer in read mtx data is null");
        return false;
    }
    int row, col;
    double value;
    int i = 0;

    while (fscanf(file, "%d %d %lf", &col, &row, &value) == 3) {
        // Store the entry (adjust 1-based index to 0-based)
        coords_x[i] = row - 1;
        coords_y[i] = col - 1;
        vals[i] = value;
        i++;
    }

    return i == LEN;
}

// ASSUME it is zeroed the res vector
void gemm_sparse(const int *cx, const int *cy, const M_TYPE* vals, const M_TYPE* vec, M_TYPE* res, const int NON_ZERO)
{
    if (cx == NULL || cy == NULL || vals == NULL || vec == NULL || res == NULL) {
        printf("NULL pointeri in GEMM sparse\n");
        return;
    }

    for (int i = 0; i < NON_ZERO; i++){
        const int row = cy[i];
        res[row] += vec[row] * vals[i];
    }
}

void gen_random_vec(M_TYPE *m, const int SIZE)
{
    if (m == NULL){
        printf("NULL pointer in random\n");
        return;
    }

    for (int i = 0; i < SIZE; i++){
        m[i] = rand() % 10;
    }
}

void print_m(const M_TYPE* res, const int M, const int N)
{
    if (res == NULL){
        printf("NULL pointer in print\n");
        return;
    }

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            printf("%.4f\t", res[i * N + j]);
        }
        printf("\n");
    }
}

void print_vec(const M_TYPE* res, const int N)
{
    if (res == NULL){
        printf("NULL pointer in print\n");
        return;
    }

    for (int i = 0; i < N; i++){
        printf("%.4f\t", res[i]);
    }
    printf("\n");
}

void print_sparse(const int *cx, const int *cy, const M_TYPE* vals, const int NON_ZERO)
{
    for (int i = 0; i < NON_ZERO; i++){
        printf("%.4f in (%d, %d)\n", vals[i], cx[i], cy[i]);
    }
    printf("\n");
}

void print_info(const int *cx, const int *cy, const M_TYPE* vals, const M_TYPE* vec, const M_TYPE* res, const int NON_ZERO, const int ROWS) {
    printf("\nA:\n");
    print_sparse(cx, cy, vals, NON_ZERO);
    printf("\nB:\n");
    print_vec(vec, ROWS);
    printf("\nRES:\n");
    print_vec(res, ROWS);
}

// TODO FINISH EX BEFORE
int main()
{
    int ROWS, COLS, NON_ZERO;
    FILE *file;

    TIMER_DEF(1);
    TIMER_START(1);
    {
        file = fopen("input.mtx", "r");
        const bool status = read_mtx_header(file, &ROWS, &COLS, &NON_ZERO);
        if (status == ERR) return -1;
    }
    TIMER_STOP(1);
    printf("%fms\n", TIMER_ELAPSED(1) / 1.e6);

    int *coords_x = calloc(NON_ZERO, sizeof(int));
    int *coords_y = calloc(NON_ZERO, sizeof(int));
    double *vals = calloc(NON_ZERO, sizeof(double));
    double *vec = calloc(ROWS, sizeof(double));
    double *res = calloc(ROWS, sizeof(double));
    double *times = calloc(TEST_CYCLES, sizeof(double));

    TIMER_DEF(2);
    TIMER_START(2);
    {
        const bool status = read_mtx_data(file, coords_x, coords_y, vals, NON_ZERO);
        if (status == ERR) return -1;
    }
    TIMER_STOP(2);
    printf("%fms\n", TIMER_ELAPSED(2) / 1.e6);

    srand(time(0) * 5);
    gen_random_vec(vec, ROWS);

    TIMER_DEF(0);
    for (int i = -WARM_CYCLES; i < TEST_CYCLES; i++) {
        // Reset
        bzero(res, ROWS * sizeof(double));

        // TEST
        TIMER_START(0);
        gemm_sparse(coords_x, coords_y, vals, vec, res, NON_ZERO);
        TIMER_STOP(0);

        // Stats
        if (i >= 0)
            times[i] = TIMER_ELAPSED(0) / 1.e6;
    }

    double average = 0, variance = 0;
    for (int i = 0; i < TEST_CYCLES; i++) {
        average += times[i] / TEST_CYCLES;
    }
    for (int i = 0; i < TEST_CYCLES; i++) {
        const double diff = times[i] - average;
        variance += diff * diff / TEST_CYCLES;
    }
    const double std_dev = sqrt(variance);

    // FREE
    free(coords_x);
    free(coords_y);
    free(vals);
    free(vec);
    free(res);
    free(times);


    const double flops1 = (2 * NON_ZERO) / average;
    printf("MY IMP TIME avarage %fms (%f MFLOP/S) over %d size and %d tests [with std. dev. = %f]\n",
            average * 1e3, flops1 / 1e6, NON_ZERO, TEST_CYCLES, std_dev);

    return 0;
}
