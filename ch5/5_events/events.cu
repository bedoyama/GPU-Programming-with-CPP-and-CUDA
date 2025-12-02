#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel1(float *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        for (int i = 0; i < 500; i++) {
            data[idx] = sqrtf(data[idx] * data[idx] + 0.01f);
        }
    }
}

__global__ void kernel2(float *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    std::cout << "========================================== " << std::endl;
    std::cout << "CUDA Events for Timing and Synchronization" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    int N = 5000000;
    size_t size = N * sizeof(float);
    
    float *h_data, *d_data;
    cudaMallocHost(&h_data, size);
    cudaMalloc(&d_data, size);
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.001f;
    }
    
    // ========== Example 1: Basic Timing with Events ==========
    std::cout << "=== Example 1: Basic Timing ===" << std::endl << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Time a memory copy
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Memory copy H2D: " << milliseconds << " ms" << std::endl;
    
    // Time a kernel
    cudaEventRecord(start);
    kernel1<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution: " << milliseconds << " ms" << std::endl << std::endl;
    
    // ========== Example 2: Timing Multiple Operations ==========
    std::cout << "=== Example 2: Timing Pipeline ===" << std::endl << std::endl;
    
    cudaEvent_t e1, e2, e3, e4;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventCreate(&e4);
    
    cudaEventRecord(e1);  // Start
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    cudaEventRecord(e2);  // After H2D
    kernel1<<<(N + 255) / 256, 256>>>(d_data, N);
    
    cudaEventRecord(e3);  // After kernel1
    kernel2<<<(N + 255) / 256, 256>>>(d_data, N);
    
    cudaEventRecord(e4);  // After kernel2
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(e4);  // Wait for everything
    
    // Calculate individual times
    float time_h2d, time_k1, time_k2, time_d2h, time_total;
    cudaEventElapsedTime(&time_h2d, e1, e2);
    cudaEventElapsedTime(&time_k1, e2, e3);
    cudaEventElapsedTime(&time_k2, e3, e4);
    cudaEventElapsedTime(&time_d2h, e4, start);  // Need another event for D2H
    
    // Actually, let's redo this properly
    cudaEventRecord(e1);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEventRecord(e2);
    
    kernel1<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaEventRecord(e3);
    
    kernel2<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaEventRecord(e4);
    
    cudaEvent_t e5;
    cudaEventCreate(&e5);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(e5);
    
    cudaEventSynchronize(e5);
    
    cudaEventElapsedTime(&time_h2d, e1, e2);
    cudaEventElapsedTime(&time_k1, e2, e3);
    cudaEventElapsedTime(&time_k2, e3, e4);
    cudaEventElapsedTime(&time_d2h, e4, e5);
    cudaEventElapsedTime(&time_total, e1, e5);
    
    std::cout << "H2D transfer:  " << time_h2d << " ms" << std::endl;
    std::cout << "Kernel 1:      " << time_k1 << " ms" << std::endl;
    std::cout << "Kernel 2:      " << time_k2 << " ms" << std::endl;
    std::cout << "D2H transfer:  " << time_d2h << " ms" << std::endl;
    std::cout << "Total:         " << time_total << " ms" << std::endl << std::endl;
    
    // ========== Example 3: Events for Synchronization ==========
    std::cout << "=== Example 3: Inter-Stream Synchronization ===" << std::endl << std::endl;
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    cudaEvent_t event_milestone;
    cudaEventCreate(&event_milestone);
    
    // Stream 1: Process data
    cudaMemcpyAsync(d_data, h_data, size / 2, cudaMemcpyHostToDevice, stream1);
    kernel1<<<(N / 2 + 255) / 256, 256, 0, stream1>>>(d_data, N / 2);
    cudaEventRecord(event_milestone, stream1);  // Record when stream1 finishes
    
    // Stream 2: Wait for stream1, then continue
    cudaStreamWaitEvent(stream2, event_milestone, 0);  // Wait for milestone
    kernel2<<<(N / 2 + 255) / 256, 256, 0, stream2>>>(d_data, N / 2);
    
    cudaDeviceSynchronize();
    
    std::cout << "Stream 1 executes kernel1" << std::endl;
    std::cout << "Stream 1 records event_milestone" << std::endl;
    std::cout << "Stream 2 waits for event_milestone" << std::endl;
    std::cout << "Stream 2 executes kernel2 (only after stream1 completes)" << std::endl << std::endl;
    
    // ========== Example 4: Event Flags ==========
    std::cout << "=== Example 4: Event Creation Flags ===" << std::endl << std::endl;
    
    // Default event
    cudaEvent_t event_default;
    cudaEventCreate(&event_default);
    std::cout << "Default event (cudaEventDefault):" << std::endl;
    std::cout << "  - Synchronizes with default stream" << std::endl;
    std::cout << "  - Can be used for timing" << std::endl << std::endl;
    
    // Blocking sync event
    cudaEvent_t event_blocking;
    cudaEventCreateWithFlags(&event_blocking, cudaEventBlockingSync);
    std::cout << "Blocking sync event (cudaEventBlockingSync):" << std::endl;
    std::cout << "  - cudaEventSynchronize() uses less CPU" << std::endl;
    std::cout << "  - Better for CPU-heavy applications" << std::endl << std::endl;
    
    // Disable timing event
    cudaEvent_t event_no_timing;
    cudaEventCreateWithFlags(&event_no_timing, cudaEventDisableTiming);
    std::cout << "No timing event (cudaEventDisableTiming):" << std::endl;
    std::cout << "  - Cannot be used for cudaEventElapsedTime()" << std::endl;
    std::cout << "  - Lower overhead for synchronization only" << std::endl << std::endl;
    
    // Interprocess event
    cudaEvent_t event_interprocess;
    cudaEventCreateWithFlags(&event_interprocess, cudaEventInterprocess);
    std::cout << "Interprocess event (cudaEventInterprocess):" << std::endl;
    std::cout << "  - Can be shared between processes" << std::endl;
    std::cout << "  - For multi-process CUDA applications" << std::endl << std::endl;
    
    // ========== Example 5: Query Event Status ==========
    std::cout << "=== Example 5: Event Queries ===" << std::endl << std::endl;
    
    cudaEvent_t query_event;
    cudaEventCreate(&query_event);
    
    // Launch long kernel
    cudaEventRecord(query_event);
    kernel1<<<(N + 255) / 256, 256>>>(d_data, N);
    
    // Query if complete (non-blocking)
    cudaError_t result = cudaEventQuery(query_event);
    if (result == cudaErrorNotReady) {
        std::cout << "Event not ready (kernel still running)" << std::endl;
    } else if (result == cudaSuccess) {
        std::cout << "Event complete" << std::endl;
    }
    
    // Wait for completion
    cudaEventSynchronize(query_event);
    std::cout << "After synchronize: Event complete" << std::endl << std::endl;
    
    // ========== Key Concepts ==========
    std::cout << "========================================== " << std::endl;
    std::cout << "Key Concepts:" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "1. EVENT TIMING:" << std::endl;
    std::cout << "   - cudaEventRecord(): Insert event into stream" << std::endl;
    std::cout << "   - cudaEventElapsedTime(): Get time between events" << std::endl;
    std::cout << "   - Higher precision than CPU timing" << std::endl << std::endl;
    
    std::cout << "2. EVENT SYNCHRONIZATION:" << std::endl;
    std::cout << "   - cudaEventSynchronize(): CPU waits for event" << std::endl;
    std::cout << "   - cudaStreamWaitEvent(): Stream waits for event" << std::endl;
    std::cout << "   - cudaEventQuery(): Non-blocking status check" << std::endl << std::endl;
    
    std::cout << "3. USE CASES:" << std::endl;
    std::cout << "   - Performance profiling" << std::endl;
    std::cout << "   - Pipeline synchronization" << std::endl;
    std::cout << "   - Inter-stream dependencies" << std::endl;
    std::cout << "   - Overlap detection" << std::endl << std::endl;
    
    std::cout << "4. BEST PRACTICES:" << std::endl;
    std::cout << "   - Use events instead of cudaDeviceSynchronize() for profiling" << std::endl;
    std::cout << "   - Reuse event objects when possible" << std::endl;
    std::cout << "   - Always destroy events (cudaEventDestroy)" << std::endl;
    std::cout << "   - Use appropriate flags for your use case" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaEventDestroy(e4);
    cudaEventDestroy(e5);
    cudaEventDestroy(event_milestone);
    cudaEventDestroy(event_default);
    cudaEventDestroy(event_blocking);
    cudaEventDestroy(event_no_timing);
    cudaEventDestroy(event_interprocess);
    cudaEventDestroy(query_event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    
    return 0;
}
