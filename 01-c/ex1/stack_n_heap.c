#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define NPROBS 8
#define WARM_UP 4

#define LEN (1024 * 1024 * 512)
#define N 5
#define M 5

void sum_s1(int *a, int *b, int *sum){
    for(int i = 0; i < LEN; i++){
        sum[i] = a[i] + b[i];
    }
}

void sum_s2(int *a, int *b, int *sum){
    for(int i = 0; i < LEN; i++){
        sum[i] = a[i] + b[i];
    }
}

int* sum_s3(int *a, int *b){
    int *sum = (int *) calloc(LEN, sizeof(int));

    for(int i = 0; i < LEN; i++){
        sum[i] = a[i] + b[i];
    }
    return sum;
}


#define PRINT_RESULT_VECTOR( V, NAME ) {    \
    printf("%2s: ", NAME);                  \
    for (int i=0; i<LEN; i++)               \
        printf("%4d ", V[i]);               \
    printf("\n");                           \
}

#define PRINT_RESULT_MATRIX(MAT, NAME) {    \
    printf("%2s matrix:\n\t", NAME);        \
    for (int i=0; i<N; i++) {               \
        for (int j=0; j<M; j++)             \
            printf("%4d ", MAT[i*M+j]);     \
        printf("\n\t");                     \
    }                                       \
    printf("\n");                           \
}

// -------- uncomment these two lines when solutions are published --------
// #include "../../solutions/lab1_sol.cu"
// #define RESULTS
// ------------------------------------------------------------------------

#ifndef SOLUTION_STACKVEC_1
#define SOLUTION_STACKVEC_1 { }
#endif

#ifndef SOLUTION_HEAPVEC_1
#define SOLUTION_HEAPVEC_1 { }
#endif

#ifndef SOLUTION_HEAPVEC_2
#define SOLUTION_HEAPVEC_2 { }
#endif

#ifndef SOLUTION_STACKVEC_2
#define SOLUTION_STACKVEC_2 { }
#endif

#ifndef SOLUTION_STACKMAT_1
#define SOLUTION_STACKMAT_1 { }
#endif

#ifndef SOLUTION_HEAPMAT_1
#define SOLUTION_HEAPMAT_1 { }
#endif

#ifndef SOLUTION_HEAPMAT_2
#define SOLUTION_HEAPMAT_2 { }
#endif

#ifndef SOLUTION_STACKMAT_2
#define SOLUTION_STACKMAT_2 { }
#endif

int main(void) {
    // ---------- for timing ----------
    float CPU_times[NPROBS];
    for (int i=0; i<NPROBS; i++)
        CPU_times[i] = 0.0;

    struct timeval temp_1, temp_2;
    // --------------------------------

#ifdef RESULTS
    printf("You are now running the \x1B[31mSOLUTION CODE\x1B[37m:\n");
#else
    printf("You are now running \x1B[31mYOUR CODE\x1B[37m:\n");
#endif
    // ---------------------- Stack vectors 1 ----------------------
    /* Generate three stack vectors a, and b of length "LEN" such
     * that for each i in {0, 1, ... LEN-1} a[i] = i, b[i] = 100 * i.
     * Then compute the vector c = a + b.
     */

    int a[LEN];
    int b[LEN];
    for(int i = 0; i < LEN; i++){
        a[i] = i;
        b[i] = -i;
    }

#if 0
        int sum1[LEN];
        sum_s1(a, b, sum1);
        //PRINT_RESULT_VECTOR(sum1, "SUM1");


#endif
    // ---------------------- Heap vectors 1 -----------------------
    /* Compute the same result as c but in a heap vector c1 allocated
     * in the main but computed in a function out of the main.
     */
#ifdef RESULTS
    SOLUTION_HEAPVEC_1
    PRINT_RESULT_VECTOR(c1, "c1")
#else
        int *sum2 = (int *) calloc(sizeof(int), LEN);
        for(int i = -WARM_UP; i<NPROBS; i++) {
            gettimeofday(&temp_1, (struct timezone *)0);
            sum_s2(a, b, sum2);
            gettimeofday(&temp_2, (struct timezone *)0);
            if (i>0)
                CPU_times[i] = ((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1.tv_usec));
        }

        {
            float mean = 0;
            for (int i = 0; i < NPROBS; i++) {
                mean += ((float) CPU_times[i]) / NPROBS;
            }
            printf("MEAN %fms", mean / 1e3);
        }
        //PRINT_RESULT_VECTOR(sum2, "SUM2");


#endif
#if 0
    // ---------------------- Heap vectors 2 -----------------------
    /* Compute the same result as c and c1 in a heap vector c2
     * which, this time, is allocated in the function out of the main
     */
#ifdef RESULTS
    SOLUTION_HEAPVEC_2
    PRINT_RESULT_VECTOR(c2, "c2")
#else
        int *sum3 = sum_s3(a, b);
        //PRINT_RESULT_VECTOR(sum3, "SUM3");


#endif
    // ---------------------- Stack vectors 2 ----------------------
    /* Is it possible to compute the c vector as a stack vector of
     * the out-main function and then return it to the main?
     */
#ifdef RESULTS
    SOLUTION_STACKVEC_2
#else
        // NO


#endif


    /* Now, do the same 4 previous exercises but with the three
     * matrices A, B, C. All the matrices has N rows and M columns.
     * Moreover:
     *   1) A[i][j] = i + j
     *   2) B[i][j] = (i + j) * 100
     *   3) C = A + B
     *
     * What are the differences when you change from vectors to
     * matrices?
     */
    // --------------------- Stack matrices ----------------------
#ifdef RESULTS
        SOLUTION_STACKMAT_1
        PRINT_RESULT_MATRIX(((int*)C), "C")
#else
        int A[N * M];
        int B[N * M];
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                A[j * M + i] = i + j;
                B[j * M + i] = (i + j) * 100;
            }
        }

        int SUM[N * M];
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                SUM[j * M + i] = A[j * M + i] + B[j * M + i];
            }
        }

        PRINT_RESULT_MATRIX(((int *) SUM), "SUM M1");


#endif

    // --------------------- Heap matrices -----------------------
#ifdef RESULTS
        SOLUTION_HEAPMAT_1
        PRINT_RESULT_MATRIX(C1, "C1")
#else

        /* |========================================| */
        /* |           Put here your code           | */
        /* |========================================| */


#endif

    // --------------------- Heap matrices -----------------------
#ifdef RESULTS
        SOLUTION_HEAPMAT_2
        PRINT_RESULT_MATRIX(C2, "C2")
#else
        /* |========================================| */
        /* |           Put here your code           | */
        /* |========================================| */


#endif

    // --------------------- Stack matrices ----------------------
#ifdef RESULTS
        SOLUTION_STACKMAT_2
#else
        // NO


#endif

    for (int i=0; i<NPROBS; i++) {
        printf("Problem %d runs in %9.8f CPU time\n", i, CPU_times[i]);
    }
    printf("\n");

    return(0);
#endif
}
