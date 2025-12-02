#include <iostream>
#include <cuda_runtime.h>

// Kernel WITHOUT shared memory (slow)
__global__ void matrixMultiplyNoShared(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel WITH shared memory (fast)
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int N) {
    __shared__ float As[16][16];  // Shared memory tile for A
    __shared__ float Bs[16][16];  // Shared memory tile for B
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + 15) / 16; t++) {
        // Load tile into shared memory
        if (row < N && (t * 16 + tx) < N)
            As[ty][tx] = A[row * N + t * 16 + tx];
        else
            As[ty][tx] = 0.0f;
            
        if ((t * 16 + ty) < N && col < N)
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();  // Wait for all threads to load
        
        // Compute partial sum using shared memory
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();  // Wait before loading next tile
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Simple reduction without shared memory
__global__ void reduceNoShared(float *input, float *output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        atomicAdd(output, input[idx]);  // Very slow!
    }
}

// Reduction with shared memory
__global__ void reduceShared(float *input, float *output, int N) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Demonstrate shared memory banks
__global__ void sharedMemoryBanks(float *output) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    
    // No bank conflicts - consecutive threads access consecutive addresses
    shared[tid] = tid;
    __syncthreads();
    
    // Bank conflict example - all threads access same bank
    // (For demonstration - don't do this!)
    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < 256; i += 32) {  // Stride of 32 = bank conflicts
            sum += shared[i];
        }
        output[0] = sum;
    }
}

int main() {
    std::cout << "========================================== " << std::endl;
    std::cout << "Shared Memory Acceleration" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    // Get device shared memory info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Shared memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "Number of banks: 32" << std::endl;
    std::cout << "Bank width: 4 bytes" << std::endl << std::endl;
    
    // ========== Example 1: Matrix Multiplication ==========
    std::cout << "=== Example 1: Matrix Multiplication ===" << std::endl << std::endl;
    
    int N = 512;
    size_t size = N * N * sizeof(float);
    
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    
    // Initialize
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Without shared memory
    cudaEventRecord(start);
    matrixMultiplyNoShared<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_no_shared;
    cudaEventElapsedTime(&time_no_shared, start, stop);
    std::cout << "Without shared memory: " << time_no_shared << " ms" << std::endl;
    std::cout << "  - Each thread reads from global memory " << N << " times" << std::endl;
    std::cout << "  - Total global memory reads: " << (long long)N * N * N << std::endl << std::endl;
    
    // With shared memory
    cudaEventRecord(start);
    matrixMultiplyShared<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);
    std::cout << "With shared memory:    " << time_shared << " ms" << std::endl;
    std::cout << "  - Tiles loaded once per block to shared memory" << std::endl;
    std::cout << "  - Reused " << threads.x << "x" << threads.y << " times per tile" << std::endl;
    std::cout << "  - Speedup: " << time_no_shared / time_shared << "x" << std::endl << std::endl;
    
    // ========== Example 2: Reduction ==========
    std::cout << "=== Example 2: Array Reduction (Sum) ===" << std::endl << std::endl;
    
    int M = 1000000;
    float *h_input = new float[M];
    for (int i = 0; i < M; i++) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, M * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, M * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    
    // Without shared memory
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduceNoShared<<<gridSize, blockSize>>>(d_input, d_output, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_reduce_no_shared;
    cudaEventElapsedTime(&time_reduce_no_shared, start, stop);
    std::cout << "Without shared memory: " << time_reduce_no_shared << " ms" << std::endl;
    std::cout << "  - Uses atomic operations on global memory (very slow)" << std::endl << std::endl;
    
    // With shared memory
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduceShared<<<gridSize, blockSize>>>(d_input, d_output, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_reduce_shared;
    cudaEventElapsedTime(&time_reduce_shared, start, stop);
    std::cout << "With shared memory:    " << time_reduce_shared << " ms" << std::endl;
    std::cout << "  - Reduces within block using shared memory" << std::endl;
    std::cout << "  - Only one atomic per block to global memory" << std::endl;
    std::cout << "  - Speedup: " << time_reduce_no_shared / time_reduce_shared << "x" << std::endl << std::endl;
    
    // ========== Key Concepts ==========
    std::cout << "========================================== " << std::endl;
    std::cout << "Key Concepts:" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "1. SHARED MEMORY:" << std::endl;
    std::cout << "   - On-chip memory, much faster than global" << std::endl;
    std::cout << "   - Shared by all threads in a block" << std::endl;
    std::cout << "   - Limited size (typically 48-96 KB per block)" << std::endl;
    std::cout << "   - Declare: __shared__ type name[size];" << std::endl << std::endl;
    
    std::cout << "2. MEMORY HIERARCHY (Speed):" << std::endl;
    std::cout << "   Registers      > 1 cycle    (private per thread)" << std::endl;
    std::cout << "   Shared Memory  ~ 1-32 cycles (shared per block)" << std::endl;
    std::cout << "   L1/L2 Cache    ~ 30-200 cycles" << std::endl;
    std::cout << "   Global Memory  ~ 400-800 cycles" << std::endl << std::endl;
    
    std::cout << "3. SYNCHRONIZATION:" << std::endl;
    std::cout << "   - __syncthreads(): Barrier within block" << std::endl;
    std::cout << "   - All threads must reach before any continue" << std::endl;
    std::cout << "   - Required after writing and before reading shared memory" << std::endl << std::endl;
    
    std::cout << "4. BANK CONFLICTS:" << std::endl;
    std::cout << "   - Shared memory divided into 32 banks" << std::endl;
    std::cout << "   - Consecutive 4-byte words in consecutive banks" << std::endl;
    std::cout << "   - Conflict when multiple threads access same bank" << std::endl;
    std::cout << "   - Best: consecutive threads -> consecutive addresses" << std::endl << std::endl;
    
    std::cout << "5. USE CASES:" << std::endl;
    std::cout << "   - Data reuse (tiles in matrix multiply)" << std::endl;
    std::cout << "   - Intra-block communication" << std::endl;
    std::cout << "   - Reductions and scans" << std::endl;
    std::cout << "   - Staging area for coalesced global memory access" << std::endl << std::endl;
    
    std::cout << "6. BEST PRACTICES:" << std::endl;
    std::cout << "   - Size: multiples of warp size (32) to avoid waste" << std::endl;
    std::cout << "   - Access: consecutive threads -> consecutive addresses" << std::endl;
    std::cout << "   - Minimize bank conflicts" << std::endl;
    std::cout << "   - Balance: shared mem per block affects occupancy" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_input;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
