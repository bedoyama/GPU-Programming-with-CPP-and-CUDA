#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__device__ bool isPrimeCheck(long long num) {
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;

    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}

__global__ void checkPrimeKernel(long long *d_primes, bool *d_isPrime, long long start, long long end) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long num = start + (tid * 2);

    if (num > end) return;

    bool isPrime = isPrimeCheck(num);

    d_primes[tid] = num;
    d_isPrime[tid] = isPrime;

    /*
    * for study purposes we can print the verification of each number
    */
    //printf("tid=%d %lld is prime? %d\n", tid, num, isPrime);
}

bool checkPrimeCpu(long long num) {
    
    if (num <= 1) return false;
    if (num == 2) return true;
    if (num % 2 == 0) return false;
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}


int main() {
    long long start =  100'001LL; // must start with odd
    long long end   =  900'001LL;

    int threadsPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    long long * h_primes = (long long *)malloc(totalNumbers * sizeof(long long));
    bool * h_isPrime = (bool *)malloc(totalNumbers * sizeof(bool));

    long long * d_primes;
    cudaMalloc((void**)&d_primes, totalNumbers * sizeof(long long));

    bool * d_isPrime;
    cudaMalloc((void**)&d_isPrime, totalNumbers * sizeof(bool));

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    checkPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, d_isPrime, start, end);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaMemcpy(h_primes, d_primes, totalNumbers * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_isPrime, d_isPrime, totalNumbers * sizeof(bool), cudaMemcpyDeviceToHost);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (long long num = start; num <= end; num += 2) {
        checkPrimeCpu(num);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;

    // Print first 42 primes found
    std::cout << "\nFirst 42 primes found:" << std::endl;
    int primesFound = 0;
    for (int i = 0; i < totalNumbers && primesFound < 42; ++i) {
        if (h_isPrime[i]) {
            std::cout << h_primes[i];
            primesFound++;
            if (primesFound < 42 && primesFound % 7 == 0) {
                std::cout << std::endl;
            } else if (primesFound < 42) {
                std::cout << ", ";
            }
        }
    }
    std::cout << std::endl << std::endl;

    std::cout << "Verifying GPU results against CPU..." << std::endl;
    for (int i = 0; i < totalNumbers; ++i) {
        long long num = h_primes[i];
        bool isPrimeGpu = h_isPrime[i];
        bool isPrimeCpu = checkPrimeCpu(num);

        if (isPrimeGpu != isPrimeCpu) {
            std::cout << "Mismatch for number " << num << ": GPU says " << isPrimeGpu << ", CPU says " << isPrimeCpu << std::endl;
        }
    }
    std::cout << "Verification finished!" << std::endl << std::endl;

    // Cleanup memory
    cudaFree(d_primes);
    cudaFree(d_isPrime);
    free(h_primes);
    free(h_isPrime);

    return 0;
}
