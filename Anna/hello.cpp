#include <iostream>
#include <omp.h>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]) 
{ 
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    #pragma omp parallel
    {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        printf("Hello world from processor %s, rank %d out of %d processors, thread %d out of %d threads\n",
               processor_name, world_rank, world_size, thread_id, num_threads);
    }

    MPI_Finalize();

    return 0;
}
