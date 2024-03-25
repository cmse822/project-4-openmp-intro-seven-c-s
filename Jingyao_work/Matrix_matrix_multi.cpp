#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000 // Matrix size

void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    double A[N][N], B[N][N], C[N][N];
    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    double start_time = omp_get_wtime();
    matrix_multiply(A, B, C);
    double end_time = omp_get_wtime();

    printf("Time taken: %f seconds\n", end_time - start_time);

    return 0;
}
