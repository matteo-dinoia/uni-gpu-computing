#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "include/my_time_lib.h"
#include <cblas.h>
#include <stdbool.h>

#define M_TYPE double
//Implement a simple GEMM algorithm that multiplies two matrices of size NxN.
//• The algorithm takes the dimension of the matrices that is a power of 2.
//• For example, ./gemm 10 multiplies two matrices of 2 10 x 2 10 .
//• Measure the FLOPS of your implementation by using -00 –O1 –O2 –O3 options.

void gemm(const M_TYPE* m1, const M_TYPE* m2, M_TYPE* res, const int N){
    if (res == NULL || m1 == NULL || m2 == NULL){
        printf("NULL pointeri in GEMM\n");
        return;
    }

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            M_TYPE sum = 0;
            for (int c = 0; c < N; c++){
                sum += m1[i * N + c] * m2[c * N + j];
            }
            res[i * N + j] = sum;
        }
    }
}

// Implement a simple GEMM algorithm that multiplies two matrices of size NxN by
// splitting the orginal matrix into blocks.
// • The block_size is such that NxN is a multiple of block_size X block_size
// • Measure the FLOS of your sequential implementations
// MEASURE CACHE HIT AND MISSES
void gemm2(const M_TYPE* m1, const M_TYPE* m2, M_TYPE* res, const int N, const int CS){
    if (res == NULL || m1 == NULL || m2 == NULL){
        printf("NULL pointeri in GEMM\n");
        return;
    }

    const int CHUNCKS = N / CS;
    for (int ci = 0; ci < N; ci += CS){
        for (int cj = 0; cj < N; cj += CS){
            for (int ck = 0; ck < N; ck += CS){
                //---------------
                for (int i = 0; i < CS; i++){
                    for (int j = 0; j < CS; j++){
                        for (int k = 0; k < CS; k++){
                            res[(ci + i) * N + (cj + j)] +=
                                m1[(ci + i) * N + (ck + k)]
                                * m2[(ck + k) * N + (cj + j)];
                        }
                    }
                }
                // --------
            }
        }
    }
}

void gen_random_matrix(M_TYPE *m, const int M, const int N){
    if (m == NULL){
        printf("NULL pointer in random\n");
        return;
    }

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            m[i * N + j] = rand() % 10;
        }
    }
}

bool check_eq(const M_TYPE* check_res, const M_TYPE* res, const int M, const int N){
    if (check_res == NULL || res == NULL){
        printf("NULL pointer in check_eq\n");
        return false;
    }

    for (int j = 0; j < M * N; j++){
        if (check_res[j] != res[j])
            return false;
    }
    return true;
}

void print_m(const M_TYPE* res, const int M, const int N)
{
    if (res == NULL){
        printf("NULL pointer in print\n");
        return;
    }

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            printf("%.0f\t", res[i * N + j]);
        }
        printf("\n");
    }
}

int main(){
    int exp = 0;

    printf("Insert matrix size: ");
    scanf("%d", &exp);
    const int N = (int) pow(2, exp);

    M_TYPE *m1 = calloc(N * N, sizeof(M_TYPE));
    M_TYPE *m2 = calloc(N * N, sizeof(M_TYPE));
    M_TYPE *res = calloc(N * N, sizeof(M_TYPE));
    M_TYPE *res2 = calloc(N * N, sizeof(M_TYPE));
    M_TYPE *check_res = calloc(N * N, sizeof(M_TYPE));

    srand(time(0) * 5);
    gen_random_matrix(m1, N, N);
    gen_random_matrix(m2, N, N);

    TIMER_DEF(0);
    TIMER_START(0);
    gemm(m1, m2, res, N);
    TIMER_STOP(0);

    TIMER_DEF(1);
    TIMER_START(1);
    gemm2(m1, m2, res2, N, (int) pow(2, exp / 2));
    TIMER_STOP(1);

    TIMER_DEF(9);
    TIMER_START(9);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, m1, N, m2, N, 0.0, check_res, N);
    TIMER_STOP(9);

    bool fail = !check_eq(check_res, res, N, N) || !check_eq(check_res, res2, N, N);
    if (fail){
        printf("Calculation is wrong\n");
        printf("\nA:\n");
        print_m(m1, N, N);
        printf("\nB:\n");
        print_m(m2, N, N);
        printf("\nEXPECTED:\n");
        print_m(check_res, N, N);
        printf("\nRES:\n");
        print_m(res, N, N);
        printf("\nRES2:\n");
        print_m(res2, N, N);
    }

    free(m1);
    free(m2);
    free(res);
    free(res2);
    free(check_res);

    if (fail)
        return 1;

    const M_TYPE sec1 = TIMER_ELAPSED(0) / 1e6;
    const M_TYPE flops1 = N * N * N / sec1;
    printf("MY IMP TIME %f (%f MFLOP/S)\n", sec1, flops1 / 1e6);

    const M_TYPE sec2 = TIMER_ELAPSED(1) / 1e6;
    const M_TYPE flops2 = N * N * N / sec2;
    printf("MY IMP TIME %f (%f MFLOP/S)\n", sec2, flops2 / 1e6);

    const M_TYPE secE = TIMER_ELAPSED(9) / 1e6;
    const M_TYPE flopsE = N * N * N / secE;
    printf("REF IMP TIME  %f (%f MFLOP/S)\n", secE, flopsE / 1e6);
    return 0;
}

