#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// Kernel for array power of 2
__global__ void arrayPower2Kernel(float *A, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] * A[i];  // or powf(A[i], 2.0f)
    }
}

// Kernel for array square root
__global__ void arraySqrtKernel(float *A, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = sqrtf(A[i]);
    }
}

// CPU implementations for verification
void arrayPower2Cpu(float *A, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * A[i];
    }
}

void arraySqrtCpu(float *A, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = sqrtf(A[i]);
    }
}

void verifyResults(float *gpu, float *cpu, int N, const char *opName) {
    bool resultsMatch = true;
    int mismatchCount = 0;
    const int maxMismatchesToShow = 5;

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(gpu[i] - cpu[i]);
        // Use relative tolerance for floating point comparison
        float relTol = 1e-5f * fmaxf(fabsf(gpu[i]), fabsf(cpu[i]));
        if (diff > fmaxf(relTol, 1e-5f)) {
            resultsMatch = false;
            if (mismatchCount < maxMismatchesToShow) {
                std::cout << "  Mismatch at index " << i << ": GPU = " << gpu[i]
                         << ", CPU = " << cpu[i] << ", diff = " << diff << std::endl;
            }
            mismatchCount++;
        }
    }

    if (resultsMatch) {
        std::cout << "  ✓ " << opName << " verification PASSED!" << std::endl;
    } else {
        std::cout << "  ✗ " << opName << " verification FAILED! Found " << mismatchCount << " mismatches." << std::endl;
    }
}

int main() {
    int N = 50'000'000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);

    // Initialize input array with positive values (needed for square root)
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i + 1);  // Values from 1 to N
    }

    // Allocate device memory
    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    std::cout << "========================================== " << std::endl;
    std::cout << "Testing Array Operations with " << N << " elements" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;

    // ===================== Array Power of 2 =====================
    std::cout << "1. Array Power of 2 (x²):" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    arrayPower2Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    auto start = std::chrono::high_resolution_clock::now();
    arrayPower2Cpu(h_A, h_C_cpu, N);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Power of 2");
    
    // Show sample results
    std::cout << "  Sample results:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "    A[" << i << "] = " << h_A[i] << " -> C[" << i << "] = " << h_C[i] << std::endl;
    }
    std::cout << std::endl;

    // ===================== Array Square Root =====================
    std::cout << "2. Array Square Root (√x):" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    arraySqrtKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    start = std::chrono::high_resolution_clock::now();
    arraySqrtCpu(h_A, h_C_cpu, N);
    stop = std::chrono::high_resolution_clock::now();
    cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Square Root");
    
    // Show sample results
    std::cout << "  Sample results:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "    A[" << i << "] = " << h_A[i] << " -> C[" << i << "] = " << h_C[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "========================================== " << std::endl;
    std::cout << "All operations completed!" << std::endl;
    std::cout << "========================================== " << std::endl;

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
    free(h_C_cpu);

    return 0;
}
