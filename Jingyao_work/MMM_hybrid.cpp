#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100 // Matrix size

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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int rows_per_rank = N / size;
    int start_row = rank * rows_per_rank;
    int end_row = (rank + 1) * rows_per_rank;
    double A[N][N], B[N][N], C[N][N];

    // Initialize matrices A and B
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    double start_time = omp_get_wtime();
    matrix_multiply(A, B, C);
    double end_time = omp_get_wtime();

    printf("Time taken: %f seconds\n", end_time - start_time);

    double final_result[N][N];
    MPI_Gather(C[start_row], rows_per_rank * N, MPI_DOUBLE,
            final_result, rows_per_rank * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Print or save the final_result matrix
    }
    MPI_Finalize();
    return 0;
}
