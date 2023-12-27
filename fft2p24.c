#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#define PI 3.14159265358979323846
#define PIx2 PI * 2.0
#define MPI_FFT_SEND_TASK 1212
#define MPI_FFT_GET_RESULT 1213

void fastFourierTransform(double* polynom, long pLength)
{
    double* A = (double*)malloc(sizeof(double) * pLength);
    double* B = (double*)malloc(sizeof(double) * pLength);

    for (long groupLength = 2; groupLength <= pLength; groupLength = groupLength * 2) //Внешний цикл определяющий размер группы (размеры маленьких массивов 2, 4, 8 и т.д.)
    {
        long distBtw = pLength / groupLength;
        long subLength = groupLength / 2;

        for (long groupNumber = 0; groupNumber < distBtw; groupNumber++) //Цикл перебирающий группы одной длины
        {

            for (long groupElement = 0; groupElement < subLength; groupElement++) //Разбиение на два независимых полинома
            {
                A[groupElement * 2] = polynom[(groupNumber + distBtw * groupElement * 2) * 2];
                A[groupElement * 2 + 1] = polynom[(groupNumber + distBtw * groupElement * 2) * 2 + 1];

                B[groupElement * 2] = polynom[(groupNumber + distBtw * (groupElement * 2 + 1)) * 2];
                B[groupElement * 2 + 1] = polynom[(groupNumber + distBtw * (groupElement * 2 + 1)) * 2 + 1];
            }

            for (long i = 0; i < groupLength; i++) //P(Wi)=A(Wi)+Wi*B(Wi)
            {
                double w, iw, tmp, itmp, alpha = PIx2 * (double)i / (double)groupLength;
                w = cos(alpha);
                iw = sin(alpha);

                tmp = w * B[(i % subLength) * 2] - iw * B[(i % subLength) * 2 + 1];
                itmp = w * B[(i % subLength) * 2 + 1] + iw * B[(i % subLength) * 2];

                polynom[(groupNumber + distBtw * i) * 2] = A[(i % subLength) * 2] + tmp;
                polynom[(groupNumber + distBtw * i) * 2 + 1] = A[(i % subLength) * 2 + 1] + itmp;
            }
        }
    }

    free(A);
    free(B);
}

void fastFourierTransformMaster(double* polynom, long pLength, int p_rank, int current_fraction) 
{
    
    long subLength = pLength / 2;
    double * A, * B;
    A = (double*) malloc(sizeof(double) * pLength);
    B = (double*) malloc(sizeof(double) * pLength);

    for (long groupElement = 0; groupElement < subLength; groupElement++) //Разбиение на два независимых полинома
    {
        A[groupElement * 2] = polynom[groupElement * 2 * 2];
        A[groupElement * 2 + 1] = polynom[groupElement * 2 * 2 + 1];

        B[groupElement * 2] = polynom[(groupElement * 2 + 1) * 2];
        B[groupElement * 2 + 1] = polynom[(groupElement * 2 + 1) * 2 + 1];
    }

    int nextFraction = current_fraction / 2;

    MPI_Request send_p;

    //pLenght = subLenght * 2
    MPI_Isend(B, pLength, MPI_DOUBLE, p_rank + nextFraction, MPI_FFT_SEND_TASK, MPI_COMM_WORLD, &send_p);

    if (nextFraction == 1)
    {
        fastFourierTransform(A, subLength);
    } else 
            {
                fastFourierTransformMaster(A, subLength, p_rank, nextFraction);
            }
    
    MPI_Status status_p;

    MPI_Wait(&send_p, &status_p);

    MPI_Recv(B, pLength, MPI_DOUBLE, p_rank + nextFraction, MPI_FFT_GET_RESULT, MPI_COMM_WORLD, &status_p);

    for (long i = 0; i < pLength; i++) //P(Wi)=A(Wi)+Wi*B(Wi)
    {
        double w, iw, tmp, itmp, alpha = PIx2 * (double)i / (double)pLength;
        w = cos(alpha);
        iw = sin(alpha);

        tmp = w * B[(i % subLength) * 2] - iw * B[(i % subLength) * 2 + 1];
        itmp = w * B[(i % subLength) * 2 + 1] + iw * B[(i % subLength) * 2];

        polynom[i * 2] = A[(i % subLength) * 2] + tmp;
        polynom[i * 2 + 1] = A[(i % subLength) * 2 + 1] + itmp;
    }

    free(A);
    free(B);
}

void fastFourierTransformSlave(long pLength, int p_rank, int current_fraction) 
{

    double* polynom = (double*)malloc(sizeof(double) * pLength * 2);

    MPI_Status status_p;

    MPI_Recv(polynom, pLength * 2, MPI_DOUBLE, p_rank - current_fraction, MPI_FFT_SEND_TASK, MPI_COMM_WORLD, &status_p);

    if (current_fraction == 1)
    {
        fastFourierTransform(polynom, pLength);
    } else 
            {
                fastFourierTransformMaster(polynom, pLength, p_rank, current_fraction);
            }

    MPI_Send(polynom, pLength * 2, MPI_DOUBLE, p_rank - current_fraction, MPI_FFT_GET_RESULT, MPI_COMM_WORLD);

    free(polynom);
}

#define m_printf if (p_rank==0)printf

int main(int argc, char **argv)
{
    int p_rank, ranksize;
    double t1, t2;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &ranksize);
    MPI_Barrier(MPI_COMM_WORLD);

    m_printf("Programm started\n");

    for(long host_size = 65536; host_size < 33554432; host_size = host_size * 2) 
    {
        for (int iteration = 0; iteration < 100; iteration++) 
        {
            t1 = MPI_Wtime();
    
            if (p_rank == 0) 
            {
                double* data = (double*)malloc(host_size * 2 * sizeof(double));
                fastFourierTransformMaster(data, host_size, p_rank, ranksize);
                free(data);
            }
            else 
                {
                    long size = host_size / ranksize;
        
                    int j = 1;
                    for (int i = 2; i < ranksize; i = i * 2) 
                    {
                        if ((p_rank % i) == j) break;
                        size = size * 2;
                        j = j * 2;
                    }
        
                    fastFourierTransformSlave(size, p_rank, j);
                }
            
            t2 = MPI_Wtime() - t1;
            MPI_Barrier(MPI_COMM_WORLD);
            m_printf("Runtime = %lf\n", t2);
            
        }
    }

    MPI_Finalize ();
    return 0;
}
