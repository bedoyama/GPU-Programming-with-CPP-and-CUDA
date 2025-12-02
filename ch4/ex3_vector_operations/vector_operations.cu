#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// Kernel for vector multiplication
__global__ void vectorMultKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] * B[i];
    }
}

// Kernel for vector division
__global__ void vectorDivKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] / B[i];
    }
}

// Kernel for vector absolute difference
__global__ void vectorAbsDiffKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = fabsf(A[i] - B[i]);
    }
}

// Kernel for vector maximum
__global__ void vectorMaxKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = fmaxf(A[i], B[i]);
    }
}

// Kernel for vector minimum
__global__ void vectorMinKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = fminf(A[i], B[i]);
    }
}

// Kernel for vector modulus (A[i] % B[i])
__global__ void vectorModulusKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = fmodf(A[i], B[i]);
    }
}

// CPU implementations for verification
void vectorMultCpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * B[i];
    }
}

void vectorDivCpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] / B[i];
    }
}

void vectorAbsDiffCpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = fabsf(A[i] - B[i]);
    }
}

void vectorMaxCpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = fmaxf(A[i], B[i]);
    }
}

void vectorMinCpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = fminf(A[i], B[i]);
    }
}

void vectorModulusCpu(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = fmodf(A[i], B[i]);
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
    int N = 10'000'000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i + 1);  // Avoid zeros for division/modulus
        h_B[i] = static_cast<float>((i % 100) + 1);  // Values from 1 to 100
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    std::cout << "========================================== " << std::endl;
    std::cout << "Testing Vector Operations with " << N << " elements" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;

    // ===================== Vector Multiplication =====================
    std::cout << "1. Vector Multiplication:" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    vectorMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    auto start = std::chrono::high_resolution_clock::now();
    vectorMultCpu(h_A, h_B, h_C_cpu, N);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Multiplication");
    std::cout << std::endl;

    // ===================== Vector Division =====================
    std::cout << "2. Vector Division:" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    vectorDivKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    start = std::chrono::high_resolution_clock::now();
    vectorDivCpu(h_A, h_B, h_C_cpu, N);
    stop = std::chrono::high_resolution_clock::now();
    cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Division");
    std::cout << std::endl;

    // ===================== Vector Absolute Difference =====================
    std::cout << "3. Vector Absolute Difference:" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    vectorAbsDiffKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    start = std::chrono::high_resolution_clock::now();
    vectorAbsDiffCpu(h_A, h_B, h_C_cpu, N);
    stop = std::chrono::high_resolution_clock::now();
    cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Absolute Difference");
    std::cout << std::endl;

    // ===================== Vector Maximum =====================
    std::cout << "4. Vector Maximum:" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    vectorMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    start = std::chrono::high_resolution_clock::now();
    vectorMaxCpu(h_A, h_B, h_C_cpu, N);
    stop = std::chrono::high_resolution_clock::now();
    cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Maximum");
    std::cout << std::endl;

    // ===================== Vector Minimum =====================
    std::cout << "5. Vector Minimum:" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    vectorMinKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    start = std::chrono::high_resolution_clock::now();
    vectorMinCpu(h_A, h_B, h_C_cpu, N);
    stop = std::chrono::high_resolution_clock::now();
    cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Minimum");
    std::cout << std::endl;

    // ===================== Vector Modulus =====================
    std::cout << "6. Vector Modulus:" << std::endl;
    
    cudaEventRecord(startEvent, 0);
    vectorModulusKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    start = std::chrono::high_resolution_clock::now();
    vectorModulusCpu(h_A, h_B, h_C_cpu, N);
    stop = std::chrono::high_resolution_clock::now();
    cpuTime = (stop - start);
    
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    verifyResults(h_C, h_C_cpu, N, "Modulus");
    std::cout << std::endl;

    std::cout << "========================================== " << std::endl;
    std::cout << "All operations completed!" << std::endl;
    std::cout << "========================================== " << std::endl;

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    return 0;
}
