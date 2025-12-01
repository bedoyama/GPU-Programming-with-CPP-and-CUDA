#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void scalarMultKernel(float *A, float scalar, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] * scalar;
    }
}

void scalarMultCpu(float *A, float scalar, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * scalar;
    }
}

int main() {
    int N = 300'000'000;
    float scalar = 3.5f;  // Scalar value for multiplication
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    float *d_A;
    float *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuCopyTime = 0;
    cudaEventElapsedTime(&gpuCopyTime, startEvent, stopEvent);

    std::cout<< std::fixed << "Time to copy data to GPU: " << gpuCopyTime << " ms" << std::endl;

    cudaEventRecord(startEvent, 0);

    scalarMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, scalar, d_C, N);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuExecutionTime = 0;
    cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent);

    std::cout<< std::fixed << "Time to execute on GPU: " << gpuExecutionTime << " ms" << std::endl;

    cudaEventRecord(startEvent, 0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuRetrieveTime = 0;
    cudaEventElapsedTime(&gpuRetrieveTime, startEvent, stopEvent);

    std::cout<< std::fixed << "Time taken to copy results back GPU: " << gpuRetrieveTime << " ms" << std::endl << std::endl;

    float gpuDuration = (gpuCopyTime + gpuExecutionTime + gpuRetrieveTime);
    std::cout << "Time taken by GPU: " << gpuDuration << " ms" << std::endl;


    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // h_C now contains GPU results, create separate array for CPU results
    float *h_C_cpu = (float *)malloc(size);

    auto start = std::chrono::high_resolution_clock::now();

    scalarMultCpu(h_A, scalar, h_C_cpu, N);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = (stop - start);

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;
    std::cout << "========================================== " << std::endl;

    std::cout << "speed up (execution time only): " << cpuDuration.count() / gpuExecutionTime << std::endl;
    std::cout << "speed up (GPU total time): " << cpuDuration.count() / gpuDuration << std::endl;
    std::cout << "========================================== " << std::endl;

    // Verify results
    std::cout << "Verifying GPU results against CPU..." << std::endl;

    bool resultsMatch = true;
    int mismatchCount = 0;
    const int maxMismatchesToShow = 10;

    for (int i = 0; i < N; ++i) {
        float diff = fabs(h_C[i] - h_C_cpu[i]);
        if (diff > 1e-5) {
            resultsMatch = false;
            if (mismatchCount < maxMismatchesToShow) {
                std::cout << "Mismatch at index " << i << ": GPU = " << h_C[i]
                         << ", CPU = " << h_C_cpu[i] << ", diff = " << diff << std::endl;
            }
            mismatchCount++;
        }
    }

    if (resultsMatch) {
        std::cout << "Verification passed! All results match." << std::endl;
    } else {
        std::cout << "Verification failed! Found " << mismatchCount << " mismatches." << std::endl;
    }
    std::cout << "Verification finished!" << std::endl;

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
    free(h_C_cpu);

    return 0;
}