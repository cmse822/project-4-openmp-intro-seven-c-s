#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 20 // Matrix size

void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
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
            A[i][j] = 1 + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (2 - 1)));
            B[i][j] = 1 + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (2 - 1)));
        }
    }

    double start_time = omp_get_wtime();
    matrix_multiply(A, B, C);
    double end_time = omp_get_wtime();

    printf("Time taken: %f seconds\n", end_time - start_time);
    FILE *file = fopen("result_matrix_MMM.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%lf ", C[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
        
    return 0;
}
