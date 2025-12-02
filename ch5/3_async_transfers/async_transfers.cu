#include <iostream>
#include <cuda_runtime.h>

__global__ void processData(float *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Simulate some work
        float result = data[idx];
        for (int i = 0; i < 100; i++) {
            result = result * 0.99f + 0.01f;
        }
        data[idx] = result;
    }
}

int main() {
    std::cout << "========================================== " << std::endl;
    std::cout << "Asynchronous Data Transfers" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    int N = 10000000;
    size_t size = N * sizeof(float);
    
    // Regular (pageable) memory
    float *h_data_regular = new float[N];
    
    // Pinned (page-locked) memory - required for async transfers
    float *h_data_pinned;
    cudaMallocHost(&h_data_pinned, size);  // Allocate pinned memory
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data_regular[i] = i * 0.001f;
        h_data_pinned[i] = i * 0.001f;
    }
    
    float *d_data;
    cudaMalloc(&d_data, size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    // ========== Test 1: Synchronous transfer (blocking) ==========
    std::cout << "=== Test 1: Synchronous Transfer (cudaMemcpy) ===" << std::endl;
    
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data_regular, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "  H2D transfer time: " << milliseconds << " ms" << std::endl;
    
    cudaEventRecord(start);
    processData<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "  Kernel time: " << milliseconds << " ms" << std::endl;
    
    cudaEventRecord(start);
    cudaMemcpy(h_data_regular, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "  D2H transfer time: " << milliseconds << " ms" << std::endl;
    
    float total_sync;
    cudaEventElapsedTime(&total_sync, start, stop);
    std::cout << "  TOTAL SYNCHRONOUS TIME: " << total_sync << " ms" << std::endl << std::endl;
    
    // ========== Test 2: Asynchronous transfer (non-blocking) ==========
    std::cout << "=== Test 2: Asynchronous Transfer (cudaMemcpyAsync) ===" << std::endl;
    std::cout << "Note: Requires pinned memory" << std::endl << std::endl;
    
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    
    cudaEventRecord(start_total);
    
    // Async H2D
    cudaEventRecord(start);
    cudaMemcpyAsync(d_data, h_data_pinned, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    
    // Kernel can start as soon as data arrives (due to stream ordering)
    processData<<<(N + 255) / 256, 256>>>(d_data, N);
    
    // Async D2H
    cudaMemcpyAsync(h_data_pinned, d_data, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);  // Wait for everything to complete
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "  H2D transfer time: " << milliseconds << " ms" << std::endl;
    
    float total_async;
    cudaEventElapsedTime(&total_async, start_total, stop_total);
    std::cout << "  TOTAL ASYNCHRONOUS TIME: " << total_async << " ms" << std::endl << std::endl;
    
    // ========== Test 3: Overlapping transfers and computation ==========
    std::cout << "=== Test 3: Overlapping with Streams ===" << std::endl;
    
    const int nStreams = 4;
    int streamSize = N / nStreams;
    size_t streamBytes = streamSize * sizeof(float);
    
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    cudaEventRecord(start_total);
    
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        
        // Each stream processes its chunk: transfer H2D -> compute -> transfer D2H
        cudaMemcpyAsync(&d_data[offset], &h_data_pinned[offset], 
                       streamBytes, cudaMemcpyHostToDevice, streams[i]);
        
        processData<<<(streamSize + 255) / 256, 256, 0, streams[i]>>>(&d_data[offset], streamSize);
        
        cudaMemcpyAsync(&h_data_pinned[offset], &d_data[offset], 
                       streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Wait for all streams
    for (int i = 0; i < nStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    
    float total_overlap;
    cudaEventElapsedTime(&total_overlap, start_total, stop_total);
    std::cout << "  TOTAL OVERLAPPED TIME: " << total_overlap << " ms" << std::endl;
    std::cout << "  Used " << nStreams << " streams to overlap operations" << std::endl << std::endl;
    
    // Cleanup streams
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    // ========== Comparison ==========
    std::cout << "========================================== " << std::endl;
    std::cout << "Performance Comparison:" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "Synchronous:  " << total_sync << " ms (baseline)" << std::endl;
    std::cout << "Asynchronous: " << total_async << " ms (" 
              << (total_sync / total_async) << "x speedup)" << std::endl;
    std::cout << "Overlapped:   " << total_overlap << " ms (" 
              << (total_sync / total_overlap) << "x speedup)" << std::endl << std::endl;
    
    // ========== Key Concepts ==========
    std::cout << "========================================== " << std::endl;
    std::cout << "Key Concepts:" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "1. SYNCHRONOUS (cudaMemcpy):" << std::endl;
    std::cout << "   - Blocks CPU until transfer completes" << std::endl;
    std::cout << "   - Sequential: copy -> compute -> copy" << std::endl;
    std::cout << "   - Works with regular host memory" << std::endl << std::endl;
    
    std::cout << "2. ASYNCHRONOUS (cudaMemcpyAsync):" << std::endl;
    std::cout << "   - Returns immediately to CPU" << std::endl;
    std::cout << "   - REQUIRES pinned memory (cudaMallocHost)" << std::endl;
    std::cout << "   - GPU operations still sequential in default stream" << std::endl << std::endl;
    
    std::cout << "3. OVERLAPPING (Multiple Streams):" << std::endl;
    std::cout << "   - Different streams can execute concurrently" << std::endl;
    std::cout << "   - Copy H2D (stream 1) + Compute (stream 2) + Copy D2H (stream 3)" << std::endl;
    std::cout << "   - Best performance when GPU has copy engines" << std::endl << std::endl;
    
    std::cout << "4. PINNED MEMORY:" << std::endl;
    std::cout << "   - Allocated with cudaMallocHost()" << std::endl;
    std::cout << "   - Page-locked, cannot be swapped out" << std::endl;
    std::cout << "   - Faster transfers via DMA" << std::endl;
    std::cout << "   - Limited resource - use sparingly" << std::endl;
    std::cout << "   - Free with cudaFreeHost()" << std::endl << std::endl;
    
    std::cout << "Best Practice:" << std::endl;
    std::cout << "- Use pinned memory for large, repeated transfers" << std::endl;
    std::cout << "- Use async + streams to hide transfer latency" << std::endl;
    std::cout << "- Profile to ensure actual overlap occurs" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    // Cleanup
    delete[] h_data_regular;
    cudaFreeHost(h_data_pinned);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    
    return 0;
}
