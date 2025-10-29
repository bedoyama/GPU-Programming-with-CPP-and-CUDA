#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

struct Point {
    float x;
    float z;
    float y;
};

__global__ void calculateEuclideanDistanceKernel(Point *lineA, Point *lineB, float *distances, int numPoints) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numPoints) {
        float dx = lineA[idx].x - lineB[idx].x;
        float dy = lineA[idx].y - lineB[idx].y;
        float dz = lineA[idx].z - lineB[idx].z;
        
        distances[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
        
    }
}

int main() {
    int numPoints = 10'000'000;
    size_t sizePoints = numPoints * sizeof(Point);
    size_t sizeDistances = numPoints * sizeof(float);

    auto start_total = std::chrono::high_resolution_clock::now();

    // Memory allocation timing
    auto start_alloc = std::chrono::high_resolution_clock::now();
    Point *h_lineA = (Point *)malloc(sizePoints);
    Point *h_lineB = (Point *)malloc(sizePoints);
    float *h_distances = (float *)malloc(sizeDistances);
    auto end_alloc = std::chrono::high_resolution_clock::now();

    // Data initialization timing
    auto start_init = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numPoints; i++) {
        h_lineA[i].x = i * 1.0f;
        h_lineA[i].y = i * 2.0f;
        h_lineA[i].z = i * 3.0f;
        h_lineB[i].x = i * 0.5f;
        h_lineB[i].y = i * 1.5f;
        h_lineB[i].z = i * 2.5f;
    }
    auto end_init = std::chrono::high_resolution_clock::now();

    // GPU memory allocation timing
    auto start_gpu_alloc = std::chrono::high_resolution_clock::now();
    Point *d_lineA;
    Point *d_lineB;
    float *d_distances;
    cudaMalloc((void **)&d_lineA, sizePoints);
    cudaMalloc((void **)&d_lineB, sizePoints);
    cudaMalloc((void **)&d_distances, sizeDistances);
    auto end_gpu_alloc = std::chrono::high_resolution_clock::now();

    // Host to device memory transfer timing
    auto start_h2d = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_lineA, h_lineA, sizePoints, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lineB, h_lineB, sizePoints, cudaMemcpyHostToDevice);
    auto end_h2d = std::chrono::high_resolution_clock::now();

    // Kernel execution setup
    int blockSize = 1024;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel execution timing
    cudaEventRecord(start);
    calculateEuclideanDistanceKernel<<<gridSize, blockSize>>>(d_lineA, d_lineB, d_distances, numPoints);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, start, stop);

    // Device to host memory transfer timing
    auto start_d2h = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_distances, d_distances, sizeDistances, cudaMemcpyDeviceToHost);
    auto end_d2h = std::chrono::high_resolution_clock::now();

    // Cleanup timing
    auto start_cleanup = std::chrono::high_resolution_clock::now();
    cudaFree(d_lineA);
    cudaFree(d_lineB);
    cudaFree(d_distances);
    free(h_lineA);
    free(h_lineB);
    free(h_distances);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    auto end_cleanup = std::chrono::high_resolution_clock::now();

    auto end_total = std::chrono::high_resolution_clock::now();

    // Calculate durations
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_alloc - start_alloc);
    auto init_time = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start_init);
    auto gpu_alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu_alloc - start_gpu_alloc);
    auto h2d_time = std::chrono::duration_cast<std::chrono::microseconds>(end_h2d - start_h2d);
    auto d2h_time = std::chrono::duration_cast<std::chrono::microseconds>(end_d2h - start_d2h);
    auto cleanup_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cleanup - start_cleanup);
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);

    // Performance analysis
    std::cout << "=== EUCLIDEAN DISTANCE PROFILING RESULTS ===" << std::endl;
    std::cout << std::fixed;
    std::cout << "Data size: " << numPoints << " points" << std::endl;
    std::cout << "Memory per array: " << sizePoints / (1024*1024) << " MB" << std::endl;
    std::cout << "Total GPU memory: " << (2 * sizePoints + sizeDistances) / (1024*1024) << " MB" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== TIMING BREAKDOWN ===" << std::endl;
    std::cout << "Host memory allocation:  " << alloc_time.count() << " µs" << std::endl;
    std::cout << "Data initialization:     " << init_time.count() << " µs" << std::endl;
    std::cout << "GPU memory allocation:   " << gpu_alloc_time.count() << " µs" << std::endl;
    std::cout << "Host->Device transfer:   " << h2d_time.count() << " µs" << std::endl;
    std::cout << "Kernel execution:        " << gpuDuration * 1000 << " µs" << std::endl;
    std::cout << "Device->Host transfer:   " << d2h_time.count() << " µs" << std::endl;
    std::cout << "Cleanup:                 " << cleanup_time.count() << " µs" << std::endl;
    std::cout << "Total execution:         " << total_time.count() << " µs" << std::endl;
    std::cout << std::endl;

    std::cout << "=== PERFORMANCE METRICS ===" << std::endl;
    std::cout << "Grid size: " << gridSize << " blocks" << std::endl;
    std::cout << "Block size: " << blockSize << " threads" << std::endl;
    std::cout << "Total threads: " << gridSize * blockSize << std::endl;
    std::cout << "Points per second: " << (numPoints / (gpuDuration / 1000.0)) / 1e6 << " M points/sec" << std::endl;
    
    // Calculate memory bandwidth
    size_t total_bytes = 2 * sizePoints + sizeDistances; // Read 2 arrays, write 1 array
    float bandwidth_gb_s = (total_bytes / (1024.0*1024.0*1024.0)) / (gpuDuration / 1000.0);
    std::cout << "Memory bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Calculate arithmetic intensity
    int ops_per_point = 8; // 3 subtractions, 3 multiplications, 1 addition, 1 sqrt
    long long total_ops = (long long)numPoints * ops_per_point;
    float gflops = (total_ops / (gpuDuration / 1000.0)) / 1e9;
    std::cout << "Compute performance: " << gflops << " GFLOPS" << std::endl;
    
    float arithmetic_intensity = (float)total_ops / total_bytes;
    std::cout << "Arithmetic intensity: " << arithmetic_intensity << " FLOPs/byte" << std::endl;

    return 0;
}