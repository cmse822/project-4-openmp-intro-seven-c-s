# Group 7 Project 4 Report

## Part 1: OpenMP Matrix-Matrix Multiplication

**Consider the simple matrix-matrix multiplication,**

```C
for i = 1, N
  for j = 1, N
    for k = 1, N
      C[i,j] += A[i,k] * B[k,j]
```

**Each of these loops is parallelizable, and could include the OpenMP flags to speed up this code, although we decided to put the flag before the first main loop ('for i = 1, N'). This produced reasonable outputs that match what we predicted.**


**1. Modify your MMM code from Project 1 to implement OpenMP threading by adding appropriate compiler directives to the outer loop of the MMM kernel. When compiling the OpenMP version of your code be sure to include the appropriate compiler flag (`-fopenmp` for GCC).**



**2. Compute the time-to-solution of your MMM code for 1 thread (e.g., `export OMP_NUM_THREADS=1`) to the non-OpenMP version (i.e., compiled without the `-fopenmp` flag). Any matrix size `N` will do here. Does it perform as you expect? If not, consider the OpenMP directives you are using.**



**3. Perform a thread-to-thread speedup study of your MMM code either on your laptop or HPCC. Compute the total time to solution for a few thread counts (in powers of 2): `1,2,4,...T`, where T is the maximum number of threads available on the machine you are using. Do this for matrix sizes of `N=20,100,1000`.**



**4. Plot the times-to-solution for the MMM for each value of `N` separately as functions of the the thread count `T`. Compare the scaling of the MMM for different matrix dimensions.**

**When N = 20**

![image](https://github.com/cmse822/project-4-openmp-intro-seven-c-s/assets/85922017/665ae583-03c2-480a-bc86-acd40508e7ac)



**When N = 100**

![image](https://github.com/cmse822/project-4-openmp-intro-seven-c-s/assets/85922017/2bee961d-6299-450e-acd8-ea09a24086c2)



**When N = 1000**

![image](https://github.com/cmse822/project-4-openmp-intro-seven-c-s/assets/85922017/7a4654e6-aa27-4c75-80b2-33d46a7e2b3c)


**5. Verify that for the same input matrices that the solution does not depend on the number of threads.**



## Part 2: Adding OpenMP threading to a simple MPI application

**1. Wrap the print statements in an `omp parallel` region.**

Review the code labeled Hello_OMP.cpp in the Part 2 Folder to view these changes.

**2. Make sure to modify the `MPI_Init` call accordingly to allow for threads! What level of thread support do you need?**

At least MPI_THREAD_FUNNELED is needed for support. Please see the below screenshot to understand why. Note that the following are monotonically increasing. SERIALIZED and MULTIPLE would also work. These notes are on page 301 of the textbook. 

<img width="892" alt="Screenshot 2024-03-25 at 12 31 31 PM" src="https://github.com/cmse822/project-4-openmp-intro-seven-c-s/assets/143351616/8e3ff691-d616-4ab0-9b92-f415428481e6">


**3. Compile the code including the appropriate flag for OpenMP support. For a GCC-based MPI installation, this would be, e.g., `mpic++ -fopenmp hello.cpp`.**

Compiled using `mpic++ -fopenmp -o Hello_OMP_exec Hello_OMP.cpp`

**4. Run the code using 2 MPI ranks and 4 OpenMP threads per rank. To do this, prior to executing the run command, set the number of threads environment variable as `> export OMP_NUM_THREADS=4`. Then you can simply execute the application with the `mpiexec` command: `> mpiexec -n 2 ./a.out`.**

Ran using `mpiexec -n 2 ./Hello_OMP_exec`

**5. Explain the output.**

The output is showing "Hello World" statements 8 different times, one per thread per rank. Since there are four threads per rank, there are eight statements. 
```
Hello world from processor dev-intel16, rank 0 out of 2 processors, thread 0 out of 4 threads
Hello world from processor dev-intel16, rank 1 out of 2 processors, thread 0 out of 4 threads
Hello world from processor dev-intel16, rank 1 out of 2 processors, thread 2 out of 4 threads
Hello world from processor dev-intel16, rank 0 out of 2 processors, thread 3 out of 4 threads
Hello world from processor dev-intel16, rank 0 out of 2 processors, thread 2 out of 4 threads
Hello world from processor dev-intel16, rank 1 out of 2 processors, thread 1 out of 4 threads
Hello world from processor dev-intel16, rank 1 out of 2 processors, thread 3 out of 4 threads
Hello world from processor dev-intel16, rank 0 out of 2 processors, thread 1 out of 4 threads
```

## Part 3: Hybrid Parallel Matrix Multiplication

**1. Add MPI to  you OpenMP MMM code by distributing the rows of one of the input matrices across MPI ranks. Have each MPI rank perform its portion of the MMM using OpenMP threading. Think very carefully about the structure of the main MMM loops! Once done, gather the resulting matrix on rank 0 and output the result. Verify that for the same input matrices the result does not depend on either the number of MPI ranks or the number of OpenMP threads per rank.**


**2. On HPCC, carry out a performance study in which you vary the number of MPI ranks, the number of OpenMP threads per rank, and the matrix size. Make plots showing the times to solution for the various cases. Explain your results.**

