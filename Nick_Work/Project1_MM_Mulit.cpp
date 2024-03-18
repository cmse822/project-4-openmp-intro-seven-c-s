#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <vector>

// Part 1 - Multiplying Two Matrices
// Ian's Processor information:
/*
11th Gen Intel i7 (2.80 GHz)

Sockets:    1
Cores:      4
L1 cache:   320  kB
L2 cache:   5.0  MB
L3 cache:   12.0 MB
*/

using namespace std;

vector<vector<double>> matrix_muliply(vector<vector<double>> matrixA, vector<vector<double>> matrixB, int m) {
    vector<vector<double>> C(m, vector<double>(m, 0.0));
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < m; ++k) 
            {
                C[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return C;
}

// function that returns a vector of vectors (matrix)
vector<vector<double>> generate_matrix(int m) {
    // Initialize M, to be a vector of vectors with entries of 0.0
    /*
    vector<vector<double>> M is creating the empty vector of vectors called M
    M(m, vector<double>(m, 0.0)) initializes the parent vector with entries of vectors
    vector<double>(m,0.0) initializes those child vectors with 0.0
    */
    vector<vector<double>> M(m, vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            M[i][j] = static_cast <double> (rand()) / RAND_MAX;
        }
    }
    return M;
}

int main() {
    // make the dimensions of each matrix: A, B, C
    int totalSteps = 47;

    vector<int> N {2, 4, 8, 10, 15,
	22, 30, 40, 50, 60,
	70, 75, 80, 85, 90,
	95, 100, 105, 110, 115,
	120, 130, 140, 160, 180,
	200, 230, 260, 300, 350,
	400, 500, 600, 700, 800,
	900, 1000, 1200, 1400, 1600,
	1800, 2000, 2300, 2600, 3000,
	3500, 4000,};

    vector<int> timeTaken(totalSteps,0);

    // Seeding C++'s random numbers
    srand(0);

    ofstream myfile ("data_not100.txt");

    myfile << "# Matrix Size";
    myfile << ", ";
    myfile << "Matrix Multiplication Time (s)";
    myfile << ", ";
    myfile << "Gflops";
    myfile << "\n";

    for(int n=0; n < totalSteps; n++) {
        auto A = generate_matrix(N[n]);
        auto B = generate_matrix(N[n]);    

        // Starting high_resolution_clock before multiplication
        // Implementation from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
        auto start = chrono::high_resolution_clock::now();
        // multiply A and B to get matrix C
        auto C = matrix_muliply(A, B, N[n]);
        // Stopping timer
        auto stop = chrono::high_resolution_clock::now();
        long long duration = chrono::duration_cast<chrono::microseconds>(stop - start);

        cout << "Time taken by matrix of size " << N[n] << " = " 
                    << duration.count() << " us" << endl;
        timeTaken[n] = duration.count();
    }

    // Writing to file:
    for(int n=0; n<totalSteps; n++) {
        long long operations; float time_taken;
        operations = (2*(N[n]*N[n]*N[n]) - N[n]*N[n]);
        time_taken = timeTaken[n]/1e6;
        myfile << N[n];
        myfile << ", ";
        myfile << time_taken;
        myfile << ", ";
        myfile << operations/time_taken * 1/1e9;
        myfile << "\n";
    }

    // display table of time taken vs size of matrix:
    cout << "N \t\t | t (micro s)";
    for(int i=0; i<totalSteps; i++) {
        cout << N[i] << '\t' << timeTaken[i] << endl;
    }

    return 0;
}
