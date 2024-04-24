#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>
#include <fstream>

using namespace std;

vector<vector<double>> matrix_multiply(vector<vector<double>> matrixA, vector<vector<double>> matrixB, int m) {
    vector<vector<double>> C(m, vector<double>(m, 0.0));
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                C[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return C;
}

vector<vector<double>> generate_matrix(int m) {
    vector<vector<double>> M(m, vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            M[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return M;
}

int main() {
    int totalSteps = 3; // number of different matrix sizes
    int maxThreads = omp_get_max_threads(); // max number of threads 

    vector<int> N = {20, 100, 1000};

    vector<vector<vector<double>>> A_matrices(totalSteps);
    vector<vector<vector<double>>> B_matrices(totalSteps);
    for (int n = 0; n < totalSteps; ++n) {
        A_matrices[n] = generate_matrix(N[n]);
        B_matrices[n] = generate_matrix(N[n]);
    }

    // Output file
    ofstream myfile("data.csv");
    myfile << "Matrix_Size,Threads,Time\n";

    // Loop through matrix sizes
    for (int n = 0; n < totalSteps; ++n) {
        // Loop through thread counts from 1 to maxThreads
        for (int threads = 1; threads <= maxThreads; threads *= 2) {
            cout << "Matrix Size: " << N[n] << ", Threads: " << threads << endl; // Debug message
            omp_set_num_threads(threads);
            auto start = chrono::high_resolution_clock::now();
            matrix_multiply(A_matrices[n], B_matrices[n], N[n]);
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

            myfile << N[n] << "," << threads << "," << duration.count() / 1e6 << "\n";
        }
    }

    myfile.close();

    return 0;
}
