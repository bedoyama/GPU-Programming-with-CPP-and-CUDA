#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel(float *data, int N, int work_factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float result = data[idx];
        // Simulate work
        for (int i = 0; i < work_factor; i++) {
            result = sqrtf(result * result + 0.01f);
        }
        data[idx] = result;
    }
}

int main() {
    std::cout << "========================================== " << std::endl;
    std::cout << "CUDA Streams for Parallelization" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    // Check device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "Async engine count: " << prop.asyncEngineCount << std::endl << std::endl;
    
    int N = 5000000;
    int nStreams = 4;
    int streamSize = N / nStreams;
    size_t streamBytes = streamSize * sizeof(float);
    
    // Allocate pinned host memory
    float *h_data;
    cudaMallocHost(&h_data, N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.001f;
    }
    
    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========== Test 1: Single Stream (Sequential) ==========
    std::cout << "=== Test 1: Single Stream (Default) ===" << std::endl;
    std::cout << "All operations execute sequentially" << std::endl << std::endl;
    
    cudaEventRecord(start);
    
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        
        // H2D transfer
        cudaMemcpy(&d_data[offset], &h_data[offset], streamBytes, cudaMemcpyHostToDevice);
        
        // Kernel
        kernel<<<(streamSize + 255) / 256, 256>>>(&d_data[offset], streamSize, 1000);
        
        // D2H transfer
        cudaMemcpy(&h_data[offset], &d_data[offset], streamBytes, cudaMemcpyDeviceToHost);
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_sequential;
    cudaEventElapsedTime(&time_sequential, start, stop);
    std::cout << "Time: " << time_sequential << " ms" << std::endl;
    std::cout << "Timeline: [H2D1][K1][D2H1][H2D2][K2][D2H2][H2D3][K3][D2H3][H2D4][K4][D2H4]" << std::endl << std::endl;
    
    // ========== Test 2: Multiple Streams (Concurrent) ==========
    std::cout << "=== Test 2: Multiple Streams (Concurrent) ===" << std::endl;
    std::cout << "Operations from different streams can overlap" << std::endl << std::endl;
    
    // Create streams
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    cudaEventRecord(start);
    
    // Launch all operations across streams
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        
        // Each stream: H2D -> Kernel -> D2H
        cudaMemcpyAsync(&d_data[offset], &h_data[offset], 
                       streamBytes, cudaMemcpyHostToDevice, streams[i]);
        
        kernel<<<(streamSize + 255) / 256, 256, 0, streams[i]>>>(&d_data[offset], streamSize, 1000);
        
        cudaMemcpyAsync(&h_data[offset], &d_data[offset], 
                       streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Wait for all streams
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_concurrent;
    cudaEventElapsedTime(&time_concurrent, start, stop);
    std::cout << "Time: " << time_concurrent << " ms" << std::endl;
    std::cout << "Timeline (idealized overlap):" << std::endl;
    std::cout << "Stream 0: [H2D1]     [K1]          [D2H1]" << std::endl;
    std::cout << "Stream 1:      [H2D2]     [K2]          [D2H2]" << std::endl;
    std::cout << "Stream 2:           [H2D3]     [K3]          [D2H3]" << std::endl;
    std::cout << "Stream 3:                [H2D4]     [K4]          [D2H4]" << std::endl << std::endl;
    
    std::cout << "Speedup: " << time_sequential / time_concurrent << "x" << std::endl << std::endl;
    
    // ========== Test 3: Stream Dependencies ==========
    std::cout << "=== Test 3: Stream Dependencies with Events ===" << std::endl;
    std::cout << "Use events to coordinate between streams" << std::endl << std::endl;
    
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    
    // Stream 0: Process first half
    cudaMemcpyAsync(d_data, h_data, streamBytes * 2, cudaMemcpyHostToDevice, streams[0]);
    kernel<<<(streamSize * 2 + 255) / 256, 256, 0, streams[0]>>>(d_data, streamSize * 2, 500);
    cudaEventRecord(event1, streams[0]);  // Mark completion
    
    // Stream 1: Wait for stream 0, then process second half
    cudaStreamWaitEvent(streams[1], event1, 0);  // Wait for event1
    cudaMemcpyAsync(&d_data[streamSize * 2], &h_data[streamSize * 2], 
                   streamBytes * 2, cudaMemcpyHostToDevice, streams[1]);
    kernel<<<(streamSize * 2 + 255) / 256, 256, 0, streams[1]>>>(&d_data[streamSize * 2], streamSize * 2, 500);
    cudaEventRecord(event2, streams[1]);
    
    // Stream 2: Wait for both, then combine results
    cudaStreamWaitEvent(streams[2], event1, 0);
    cudaStreamWaitEvent(streams[2], event2, 0);
    cudaMemcpyAsync(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost, streams[2]);
    
    cudaDeviceSynchronize();
    
    std::cout << "Dependency chain:" << std::endl;
    std::cout << "  Stream 0 -> event1 -> Stream 1 -> event2 -> Stream 2" << std::endl << std::endl;
    
    // ========== Test 4: Stream Priorities ==========
    std::cout << "=== Test 4: Stream Priorities ===" << std::endl;
    
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    
    std::cout << "Priority range: " << greatestPriority << " (highest) to " 
              << leastPriority << " (lowest)" << std::endl;
    
    if (greatestPriority < leastPriority) {
        cudaStream_t highPriorityStream, lowPriorityStream;
        cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority);
        cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamNonBlocking, leastPriority);
        
        std::cout << "Created high and low priority streams" << std::endl;
        std::cout << "High priority work executes first when resources are contended" << std::endl;
        
        cudaStreamDestroy(highPriorityStream);
        cudaStreamDestroy(lowPriorityStream);
    } else {
        std::cout << "Stream priorities not supported on this device" << std::endl;
    }
    std::cout << std::endl;
    
    // ========== Key Concepts ==========
    std::cout << "========================================== " << std::endl;
    std::cout << "Key Concepts:" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "1. STREAMS:" << std::endl;
    std::cout << "   - Sequence of operations that execute in order" << std::endl;
    std::cout << "   - Operations in DIFFERENT streams can run concurrently" << std::endl;
    std::cout << "   - Default stream (0) blocks other streams" << std::endl << std::endl;
    
    std::cout << "2. BENEFITS:" << std::endl;
    std::cout << "   - Overlap: Copy + Compute + Copy (different streams)" << std::endl;
    std::cout << "   - Better GPU utilization" << std::endl;
    std::cout << "   - Hide latency of data transfers" << std::endl << std::endl;
    
    std::cout << "3. USAGE:" << std::endl;
    std::cout << "   - Create: cudaStreamCreate(&stream)" << std::endl;
    std::cout << "   - Use: kernel<<<grid, block, sharedMem, stream>>>(...)" << std::endl;
    std::cout << "   - Sync: cudaStreamSynchronize(stream)" << std::endl;
    std::cout << "   - Destroy: cudaStreamDestroy(stream)" << std::endl << std::endl;
    
    std::cout << "4. COORDINATION:" << std::endl;
    std::cout << "   - cudaEventRecord(event, stream): Mark point in stream" << std::endl;
    std::cout << "   - cudaStreamWaitEvent(stream, event): Wait in stream for event" << std::endl;
    std::cout << "   - cudaEventSynchronize(event): CPU waits for event" << std::endl << std::endl;
    
    std::cout << "5. BEST PRACTICES:" << std::endl;
    std::cout << "   - Use non-default streams for concurrency" << std::endl;
    std::cout << "   - Ensure enough work per stream (not too fine-grained)" << std::endl;
    std::cout << "   - Requires pinned memory for async copies" << std::endl;
    std::cout << "   - Profile to verify actual concurrency" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    // Cleanup
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    
    return 0;
}
