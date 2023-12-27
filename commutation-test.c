#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#define PI 3.14159265358979323846
#define PIx2 PI * 2.0
#define MPI_FFT_SEND_TASK 1212

int main(int argc, char **argv)
{
    int p_rank, ranksize;
    double t1, t2;
    char results[19][100];
    char result_string[1600];
    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &ranksize);
    MPI_Barrier(MPI_COMM_WORLD);

    m_printf("Programm started\n");

    int ri = 0;

    for(long host_size = 16; host_size < 5000000; host_size = host_size * 2) 
    {
        double avg_time = 0.;
        
        for (int iteration = 0; iteration < 256; iteration++) 
        {
            
            if (p_rank == 0) 
            {
                double* data = (double*)malloc(host_size * 2 * sizeof(double));
                
                t1 = MPI_Wtime();

                MPI_Send(data,  host_size * 2, MPI_DOUBLE, 1, MPI_FFT_SEND_TASK, MPI_COMM_WORLD);              
                
                t2 = MPI_Wtime() - t1;

                free(data);
                
                avg_time = avg_time + t2;
                printf("\tSize = %ld,\tIteration = %d,\tRuntime = %.8lf\n", host_size, iteration, t2);
            }
            else 
                {
                    double* data = (double*)malloc(host_size * 2 * sizeof(double));

                    MPI_Status status_p;

                    MPI_Recv(data,  host_size * 2, MPI_DOUBLE, 0, MPI_FFT_SEND_TASK, MPI_COMM_WORLD, &status_p);

                    free(data);
                }
            
            
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (p_rank == 0) 
        {
            avg_time = avg_time / 256;
            sprintf(results[ri], "Size = %ld,\tAverage runtime = %.8lf\n", host_size, avg_time);
            printf(results[ri]);
            ri++;
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (p_rank == 0) 
    {
        snprintf(result_string, 1600, "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s", 
                    results[0], results[1], results[2], results[3], 
                    results[4], results[5], results[6], results[7],
                    results[8], results[9], results[10], results[11], 
                    results[12], results[13], results[14], results[15], results[16], results[17], results[18]);
        
        char file_name[50];

        sprintf(file_name, "results-comm-test.txt");

        FILE* fp = fopen(file_name, "w");

        if(fp)
        {
            fputs(result_string, fp);
            fclose(fp);
            printf("DONE : results-comm-test.txt");
        }
    }

    MPI_Finalize ();
    return 0;
}