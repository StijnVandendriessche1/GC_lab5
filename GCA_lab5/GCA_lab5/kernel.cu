#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>

#include <chrono>

#define N 2048

//writeToCSV
//help function to write stuff to csv
void writeRecordToFile(std::string filename, std::string fieldOne, std::string fieldTwo, int fieldThree)
{
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    file << fieldOne << "," << fieldTwo << "," << fieldThree << std::endl;
    file.close();
}

__global__ void getMaxReduction(int* A, int* max)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = N;
    for (int n = 0; n < (log2f(N)); n++)
    {
        j = j / 2;
        if (i < j)
        {
            if (A[i] < A[i + j])
            {
                A[i] = A[i + j];
            }
        }
        __syncthreads();
    }
    if (i == 0)
    {
        *max = A[0];
    }
}

__global__ void getMinReduction(int* A, int* min)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = N;
    for (int n = 0; n < (log2f(N)); n++)
    {
        j = j / 2;
        if (i < j)
        {
            if (A[i] > A[i + j])
            {
                A[i] = A[i + j];
            }
        }
        __syncthreads();
    }
    if (i == 0)
    {
        *min = A[0];
    }
}

__global__ void getSumReduction(int* A, int* sum)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = N;
    for (int n = 0; n < (log2f(N)); n++)
    {
        j = j / 2;
        if (i < j)
        {
            A[i] = A[i] + A[i + j];
        }
        __syncthreads();
    }
    if (i == 0)
    {
        *sum = A[0];
    }
}

__global__ void getProdReduction(int* A, int* prod)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = N;
    for (int n = 0; n < (log2f(N)); n++)
    {
        j = j / 2;
        if (i < j)
        {
            A[i] *= A[i + j];
        }
        __syncthreads();
    }
    if (i == 0)
    {
        *prod = A[0];
    }
}

int* getRandomArray(int n)
{
    size_t size = n * sizeof(int);
    int* A = (int*)malloc(size);
    for (int i = 0; i < n; i++)
    {
        A[i] = rand();
    }
    return A;
}

void executeSync(int* res, bool print = true)
{
    //start chrono
    auto startTimeGPU = std::chrono::steady_clock::now();

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    //workflow for maximum
    int max = 0;
    int* A = getRandomArray(N);
    int* gpuA = NULL;
    cudaMalloc((void**)&gpuA, N * sizeof(int));
    int* gpuMax = NULL;
    cudaMalloc((void**)&gpuMax, sizeof(int));
    cudaMemcpy(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuMax, &max, sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t startMax, stopMax;
    cudaEventCreate(&startMax);
    cudaEventCreate(&stopMax);
    cudaEventRecord(startMax);
    getMaxReduction << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuMax);
    cudaEventRecord(stopMax);
    cudaEventSynchronize(stopMax);
    cudaMemcpy(&max, gpuMax, sizeof(int), cudaMemcpyDeviceToHost);

    //workflow for minimum
    int min = 0;
    int* B = getRandomArray(N);
    int* gpuB = NULL;
    cudaMalloc((void**)&gpuB, N * sizeof(int));
    int* gpuMin = NULL;
    cudaMalloc((void**)&gpuMin, sizeof(int));
    cudaMemcpy(gpuB, B, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuMin, &min, sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t startMin, stopMin;
    cudaEventCreate(&startMin);
    cudaEventCreate(&stopMin);
    cudaEventRecord(startMin);
    getMinReduction << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuMin);
    cudaEventRecord(stopMin);
    cudaEventSynchronize(stopMin);
    cudaMemcpy(&min, gpuMin, sizeof(int), cudaMemcpyDeviceToHost);

    //workflow for sum
    int sum = 0;
    int* C = getRandomArray(N);
    int* gpuC = NULL;
    cudaMalloc((void**)&gpuC, N * sizeof(int));
    int* gpuSum = NULL;
    cudaMalloc((void**)&gpuSum, sizeof(int));
    cudaMemcpy(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuSum, &sum, sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t startSum, stopSum;
    cudaEventCreate(&startSum);
    cudaEventCreate(&stopSum);
    cudaEventRecord(startSum);
    getSumReduction << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuSum);
    cudaEventRecord(stopSum);
    cudaEventSynchronize(stopSum);   
    cudaMemcpy(&sum, gpuSum, sizeof(int), cudaMemcpyDeviceToHost);

    //workflow for prod
    int prod = 0;
    int* D = getRandomArray(N);
    int* gpuD = NULL;
    cudaMalloc((void**)&gpuD, N * sizeof(int));
    int* gpuProd = NULL;
    cudaMalloc((void**)&gpuProd, sizeof(int));
    cudaMemcpy(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuProd, &prod, sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t startProd, stopProd;
    cudaEventCreate(&startProd);
    cudaEventCreate(&stopProd);
    cudaEventRecord(startProd);
    getProdReduction << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuProd);
    cudaEventRecord(stopProd);
    cudaEventSynchronize(stopProd);
    cudaMemcpy(&prod, gpuProd, sizeof(int), cudaMemcpyDeviceToHost);

    auto durationGPU = std::chrono::steady_clock::now() - startTimeGPU;
    
    if (print)
    {
        writeRecordToFile("output4.csv", "sync", std::to_string(N), durationGPU.count());
    }

    //write results to result
    res[0] = max;
    res[1] = min;
    res[2] = sum;
    res[3] = prod;

    //free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
    cudaFree(gpuD);
    cudaFree(gpuMax);
    cudaFree(gpuMin);
    cudaFree(gpuSum);
    cudaFree(gpuProd);
}

void executeAsync(int* res, bool print = true)
{
    auto startTimeGPU = std::chrono::steady_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    //workflow for maximum
    int max = 0;
    int* A = getRandomArray(N);
    int* gpuA = NULL;
    cudaMalloc((void**)&gpuA, N * sizeof(int));
    int* gpuMax = NULL;
    cudaMalloc((void**)&gpuMax, sizeof(int));
    cudaMemcpyAsync(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(gpuMax, &max, sizeof(int), cudaMemcpyHostToDevice, stream1);
    getMaxReduction << <blocksPerGrid, threadsPerBlock, 0, stream1 >> > (gpuA, gpuMax);
    cudaMemcpyAsync(&max, gpuMax, sizeof(int), cudaMemcpyDeviceToHost, stream1);

    //workflow for minimum
    int min = 0;
    int* B = getRandomArray(N);
    int* gpuB = NULL;
    cudaMalloc((void**)&gpuB, N * sizeof(int));
    int* gpuMin = NULL;
    cudaMalloc((void**)&gpuMin, sizeof(int));
    cudaMemcpyAsync(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(gpuMin, &min, sizeof(int), cudaMemcpyHostToDevice, stream2);
    getMinReduction << <blocksPerGrid, threadsPerBlock, 0, stream2 >> > (gpuA, gpuMin);
    cudaMemcpyAsync(&min, gpuMin, sizeof(int), cudaMemcpyDeviceToHost, stream2);

    //workflow for sum
    int sum = 0;
    int* C = getRandomArray(N);
    int* gpuC = NULL;
    cudaMalloc((void**)&gpuC, N * sizeof(int));
    int* gpuSum = NULL;
    cudaMalloc((void**)&gpuSum, sizeof(int));
    cudaMemcpyAsync(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(gpuSum, &sum, sizeof(int), cudaMemcpyHostToDevice, stream3);
    getSumReduction << <blocksPerGrid, threadsPerBlock, 0, stream3 >> > (gpuA, gpuSum);
    cudaMemcpyAsync(&sum, gpuSum, sizeof(int), cudaMemcpyDeviceToHost, stream3);

    //workflow for prod
    int prod = 0;
    int* D = getRandomArray(N);
    int* gpuD = NULL;
    cudaMalloc((void**)&gpuD, N * sizeof(int));
    int* gpuProd = NULL;
    cudaMalloc((void**)&gpuProd, sizeof(int));
    cudaMemcpyAsync(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice, stream4);
    cudaMemcpyAsync(gpuProd, &prod, sizeof(int), cudaMemcpyHostToDevice, stream4);
    getProdReduction << <blocksPerGrid, threadsPerBlock, 0, stream4 >> > (gpuA, gpuProd);
    cudaMemcpyAsync(&prod, gpuProd, sizeof(int), cudaMemcpyDeviceToHost, stream4);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    auto durationGPU = std::chrono::steady_clock::now() - startTimeGPU;

    if (print)
    {
        writeRecordToFile("output4.csv", "async", std::to_string(N), durationGPU.count());
    }

    //write results to result
    res[0] = max;
    res[1] = min;
    res[2] = sum;
    res[3] = prod;

    //free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
    cudaFree(gpuD);
    cudaFree(gpuMax);
    cudaFree(gpuMin);
    cudaFree(gpuSum);
    cudaFree(gpuProd);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
}


int main()
{
    size_t resSize = 4 * sizeof(int);
    int* res = (int*)malloc(resSize);

    executeSync(res, false);
    executeAsync(res, false);

    for (int i = 0; i < 1000; i++)
    {
        executeSync(res);
        printf("SYNC:\nmax: %d\nmin: %d\nsum: %d\nprod: %d\n", res[0], res[1], res[2], res[3]);
        executeAsync(res);
        printf("ASYNC:\nmax: %d\nmin: %d\nsum: %d\nprod: %d\n", res[0], res[1], res[2], res[3]);
    }
}
