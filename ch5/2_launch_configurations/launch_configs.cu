#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Simple kernel for testing different configurations
__global__ void vectorAdd(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// 2D kernel for matrix operations
__global__ void matrixAdd2D(float *a, float *b, float *c, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

void testConfiguration1D(int N, int threadsPerBlock) {
    size_t size = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "  Threads/Block: " << threadsPerBlock 
              << " | Blocks: " << blocksPerGrid 
              << " | Time: " << milliseconds / 100.0 << " ms" 
              << " | Occupancy: " << (threadsPerBlock * blocksPerGrid) << " threads" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void testConfiguration2D(int width, int height, dim3 threadsPerBlock) {
    size_t size = width * height * sizeof(float);
    float *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    matrixAdd2D<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height);
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        matrixAdd2D<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    int threadsPerBlockTotal = threadsPerBlock.x * threadsPerBlock.y;
    int totalBlocks = blocksPerGrid.x * blocksPerGrid.y;
    
    std::cout << "  Block: " << threadsPerBlock.x << "x" << threadsPerBlock.y 
              << " (" << threadsPerBlockTotal << " threads)"
              << " | Grid: " << blocksPerGrid.x << "x" << blocksPerGrid.y 
              << " (" << totalBlocks << " blocks)"
              << " | Time: " << milliseconds / 100.0 << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "========================================== " << std::endl;
    std::cout << "Launch Configuration Examples" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: [" << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "Max grid dimensions: [" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl << std::endl;
    
    // Example 1: 1D Configurations
    std::cout << "=== Example 1: 1D Vector Addition (N = 10,000,000) ===" << std::endl;
    std::cout << "Testing different threads per block configurations:" << std::endl;
    int N = 10000000;
    
    int configs[] = {32, 64, 128, 256, 512, 1024};
    for (int threads : configs) {
        testConfiguration1D(N, threads);
    }
    
    std::cout << "\nObservations:" << std::endl;
    std::cout << "- Multiples of warp size (32) perform better" << std::endl;
    std::cout << "- 256 or 512 threads/block often optimal" << std::endl;
    std::cout << "- Too few threads: underutilization" << std::endl;
    std::cout << "- Too many threads: resource constraints" << std::endl << std::endl;
    
    // Example 2: 2D Configurations
    std::cout << "=== Example 2: 2D Matrix Addition (2048x2048) ===" << std::endl;
    std::cout << "Testing different 2D block configurations:" << std::endl;
    int width = 2048, height = 2048;
    
    dim3 configs2D[] = {
        dim3(8, 8),    // 64 threads
        dim3(16, 16),  // 256 threads (common choice)
        dim3(32, 32),  // 1024 threads (max for many GPUs)
        dim3(16, 32),  // 512 threads (rectangular)
        dim3(32, 16),  // 512 threads (rectangular, opposite)
        dim3(8, 32),   // 256 threads (very rectangular)
    };
    
    for (const auto& config : configs2D) {
        testConfiguration2D(width, height, config);
    }
    
    std::cout << "\nObservations:" << std::endl;
    std::cout << "- 16x16 (256 threads) is commonly optimal for 2D" << std::endl;
    std::cout << "- Square blocks often work well for matrices" << std::endl;
    std::cout << "- Consider memory access patterns when choosing shape" << std::endl << std::endl;
    
    // Example 3: Occupancy considerations
    std::cout << "=== Example 3: Understanding Occupancy ===" << std::endl;
    
    int blockSize = 256;
    int minGridSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vectorAdd, 0, N);
    
    std::cout << "Occupancy API suggestions for vectorAdd kernel:" << std::endl;
    std::cout << "  Suggested block size: " << blockSize << std::endl;
    std::cout << "  Minimum grid size for max occupancy: " << minGridSize << std::endl;
    std::cout << "  Total threads: " << blockSize * minGridSize << std::endl << std::endl;
    
    // Example 4: Common patterns
    std::cout << "=== Example 4: Common Launch Patterns ===" << std::endl << std::endl;
    
    std::cout << "Pattern 1: Simple 1D kernel" << std::endl;
    std::cout << "  int idx = threadIdx.x + blockIdx.x * blockDim.x;" << std::endl;
    std::cout << "  Launch: kernel<<<(N + 255) / 256, 256>>>(...);" << std::endl << std::endl;
    
    std::cout << "Pattern 2: 2D kernel (images, matrices)" << std::endl;
    std::cout << "  dim3 threads(16, 16);" << std::endl;
    std::cout << "  dim3 blocks((width + 15) / 16, (height + 15) / 16);" << std::endl;
    std::cout << "  Launch: kernel<<<blocks, threads>>>(...);" << std::endl << std::endl;
    
    std::cout << "Pattern 3: Grid-stride loop (flexible N)" << std::endl;
    std::cout << "  int idx = threadIdx.x + blockIdx.x * blockDim.x;" << std::endl;
    std::cout << "  int stride = blockDim.x * gridDim.x;" << std::endl;
    std::cout << "  for (int i = idx; i < N; i += stride) { ... }" << std::endl;
    std::cout << "  Launch: kernel<<<numBlocks, 256>>>(...);" << std::endl << std::endl;
    
    // Best practices
    std::cout << "========================================== " << std::endl;
    std::cout << "Launch Configuration Best Practices:" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "1. Use multiples of warp size (32)" << std::endl;
    std::cout << "2. Typical range: 128-512 threads per block" << std::endl;
    std::cout << "3. For 2D: 16x16 (256) or 32x32 (1024) common" << std::endl;
    std::cout << "4. Ensure enough blocks to saturate GPU" << std::endl;
    std::cout << "5. Always check: if (idx < N) to handle boundaries" << std::endl;
    std::cout << "6. Profile different configs for your specific case" << std::endl;
    std::cout << "7. Consider shared memory usage (affects occupancy)" << std::endl;
    std::cout << "8. Balance: more threads = more parallelism" << std::endl;
    std::cout << "           BUT also = more resource usage" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    return 0;
}
