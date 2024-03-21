#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <omp.h> // Include OpenMP header

using namespace std;

vector<vector<double>> matrix_muliply(vector<vector<double>> matrixA, vector<vector<double>> matrixB, int m) {
    vector<vector<double>> C(m, vector<double>(m, 0.0));
    #pragma omp parallel for
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

vector<vector<double>> generate_matrix(int m) {
    vector<vector<double>> M(m, vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            M[i][j] = static_cast <double> (rand()) / RAND_MAX;
        }
    }
    return M;
}

int main() {
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

    srand(0);

   // ofstream myfile ("data_not100.txt");

    myfile << "# Matrix Size";
    myfile << ", ";
    myfile << "Matrix Multiplication Time (s)";
    myfile << ", ";
    myfile << "Gflops";
    myfile << "\n";

    for(int n=0; n < totalSteps; n++) {
        auto A = generate_matrix(N[n]);
        auto B = generate_matrix(N[n]);    

        auto start = chrono::high_resolution_clock::now();
        auto C = matrix_muliply(A, B, N[n]);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

        cout << "Time taken by matrix multiplication of size " << N[n] << " = " 
                    << duration.count() << " us" << endl;

        timeTaken[n] = static_cast<long long>(duration.count());
    }

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

    cout << "N \t\t | t (micro s)";
    for(int i=0; i<totalSteps; i++) {
        cout << N[i] << '\t' << timeTaken[i] << endl;
    }

    return 0;
}
