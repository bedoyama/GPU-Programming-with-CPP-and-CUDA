# GPU Profiling Guide: CUDA Programs with NVIDIA Nsight Tools

## Overview

This guide demonstrates how to compile and profile CUDA programs using NVIDIA profiling tools, specifically using the Euclidean Distance calculation program as an example. The guide covers both basic compilation and advanced profiling techniques.

## Prerequisites

### Development Environment Setup

This guide assumes you're working in a dev container with the following configuration:

**`.devcontainer/devcontainer.json`** should include:
```json
{
    "build": {
        "dockerfile": "DockerfileUbuntu"
    },
    "runArgs": [
        "--gpus", "all",
        "--cap-add=SYS_ADMIN",
        "--privileged"
    ],
    "customizations": {
        "vscode": {
            "extensions": [
                "nvidia.nsight-vscode-edition",
                "ms-vscode.cmake-tools",
                "ms-vscode.cpptools-extension-pack"
            ]
        }
    }
}
```

### Required Tools
- **CUDA Toolkit 12.0+** with nvcc compiler
- **NVIDIA Nsight Systems** (timeline profiler)
- **NVIDIA Nsight Compute** (kernel profiler - limited in containers)
- **CMAKE** (optional, for build management)

## Sample Program: Euclidean Distance Calculator

### Basic Program Structure

The euclidean distance program (`euclidean_distance.cu`) calculates distances between corresponding points in two arrays:

```cpp
__global__ void calculateEuclideanDistanceKernel(Point *lineA, Point *lineB, 
                                                 float *distances, int numPoints) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numPoints) {
        float dx = lineA[idx].x - lineB[idx].x;
        float dy = lineA[idx].y - lineB[idx].y;
        float dz = lineA[idx].z - lineB[idx].z;
        
        distances[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}
```

**Key characteristics:**
- Processes 10 million 3D points
- Memory-bound workload (low arithmetic intensity)
- Uses 267 MB of GPU memory
- Simple parallelizable computation

## Compilation Methods

### Method 1: Direct nvcc Compilation

```bash
# Basic compilation
cd /workspaces/GPU-Programming-with-CPP-and-CUDA-bedoyama/ch4/3_euclidean_distance
nvcc -o euclidean_distance euclidean_distance.cu

# Compilation with debug info for profiling
nvcc -g -G -o euclidean_distance euclidean_distance.cu

# Optimized compilation with profiling symbols
nvcc -O3 -lineinfo -o euclidean_distance euclidean_distance.cu
```

**Compilation Flags Explained:**
- `-g`: Generate host debug information
- `-G`: Generate device debug information  
- `-O3`: Maximum optimization level
- `-lineinfo`: Generate line number information for profiling
- `--ptxas-options=-v`: Verbose PTX assembler output

### Method 2: CMake Build System

```bash
cd /workspaces/GPU-Programming-with-CPP-and-CUDA-bedoyama/ch4/3_euclidean_distance/build
cmake ..
make
```

**CMakeLists.txt** should contain:
```cmake
cmake_minimum_required(VERSION 3.18)
project(euclidean_distance LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(euclidean_distance ../euclidean_distance.cu)

set_target_properties(euclidean_distance PROPERTIES
    CUDA_ARCHITECTURES "50;60;70;75;80;86"
)
```

## Profiling Methods

### 1. NVIDIA Nsight Systems (Timeline Profiling) ✅ Working

Nsight Systems provides system-wide timeline analysis and is the primary profiling tool available in containerized environments.

#### Basic Profiling
```bash
cd /workspaces/GPU-Programming-with-CPP-and-CUDA-bedoyama/ch4/3_euclidean_distance/build

# Basic timeline profiling
nsys profile ./euclidean_distance

# Enhanced CUDA tracing
nsys profile --trace=cuda,cudnn,cublas ./euclidean_distance

# Profile with specific duration
nsys profile --duration=10 ./euclidean_distance
```

#### Advanced Profiling Options
```bash
# Generate specific output file name
nsys profile -o euclidean_profile ./euclidean_distance

# Capture CPU sampling data
nsys profile --sample=cpu ./euclidean_distance

# Profile with environment variables
nsys profile --env-var CUDA_LAUNCH_BLOCKING=1 ./euclidean_distance
```

#### Analyzing Results
```bash
# Generate text-based statistics
nsys stats report1.nsys-rep

# Generate CSV format statistics  
nsys stats --format csv report1.nsys-rep

# Export specific analysis
nsys stats --report gputrace report1.nsys-rep
```

**Expected Output Sections:**
- **OS Runtime Summary**: System calls and CPU operations
- **CUDA API Summary**: CUDA function call timings
- **GPU Kernel Summary**: Kernel execution details (if permissions allow)

### 2. NVIDIA Nsight Compute (Kernel Profiling) ❌ Limited

Nsight Compute provides detailed kernel-level analysis but requires specific GPU permissions not available in most container environments.

#### Attempting Kernel Profiling
```bash
# Basic kernel profiling (may fail with permission errors)
ncu --set basic ./euclidean_distance

# Try with administrative privileges
sudo /usr/local/cuda/bin/ncu --set basic ./euclidean_distance
```

**Common Permission Error:**
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access 
NVIDIA GPU Performance Counters on the target device 0.
```

**Workaround Solutions:**
- Run on bare metal systems (not containers)
- Use cloud instances with proper GPU counter access
- Focus on Nsight Systems for container-based development

### 3. Enhanced Manual Timing Analysis ✅ Recommended

Create a detailed timing version of your program for comprehensive analysis.

#### Enhanced Profiling Program
```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// ... (kernel code remains the same)

int main() {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Detailed timing for each phase
    auto start_alloc = std::chrono::high_resolution_clock::now();
    // ... memory allocation code
    auto end_alloc = std::chrono::high_resolution_clock::now();
    
    auto start_gpu_alloc = std::chrono::high_resolution_clock::now();
    // ... GPU allocation code  
    auto end_gpu_alloc = std::chrono::high_resolution_clock::now();
    
    // CUDA Events for kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    calculateEuclideanDistanceKernel<<<gridSize, blockSize>>>(d_lineA, d_lineB, d_distances, numPoints);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, start, stop);
    
    // Calculate and display comprehensive metrics
    // ... (see euclidean_distance_profiled.cu for full implementation)
}
```

#### Compile and Run Enhanced Version
```bash
# Compile the profiling version
nvcc -o euclidean_distance_profiled euclidean_distance_profiled.cu

# Run with detailed timing output
./euclidean_distance_profiled
```

## Profiling Results Analysis

### Example Nsight Systems Output

**CUDA API Summary:**
```
Operation          Time (%)  Duration (ms)  Calls  Avg (ms)
cudaMalloc         71.1%     193.56         3      64.52    ← Major bottleneck
cudaMemcpy         27.1%     73.88          3      24.63    ← Memory transfer cost
cudaFree           0.9%      2.36           3      0.79
cudaLaunchKernel   0.3%      0.87           1      0.87     ← Fast kernel execution
```

### Example Enhanced Timing Output

```
=== EUCLIDEAN DISTANCE PROFILING RESULTS ===
Data size: 10000000 points
Memory per array: 114 MB
Total GPU memory: 267 MB

=== TIMING BREAKDOWN ===
Host memory allocation:  21 µs
Data initialization:     221505 µs      ← CPU bottleneck
GPU memory allocation:   1546939 µs     ← Major bottleneck (71% of time)
Host->Device transfer:   33906 µs
Kernel execution:        2174.560059 µs ← Optimized (only 0.1% of time)
Device->Host transfer:   37408 µs
Cleanup:                 37219 µs
Total execution:         1879680 µs

=== PERFORMANCE METRICS ===
Grid size: 9766 blocks
Block size: 1024 threads
Total threads: 10000384
Points per second: 4598.631299 M points/sec
Memory bandwidth: 119.918655 GB/s
Compute performance: 36.789051 GFLOPS
Arithmetic intensity: 0.285714 FLOPs/byte ← Memory-bound workload
```

## Performance Analysis Guidelines

### Key Metrics to Monitor

1. **Kernel Execution Time**
   - Should be the dominant time for compute-bound kernels
   - In this example: only 2.17ms (very efficient)

2. **Memory Transfer Times**
   - Host↔Device transfers often bottlenecks
   - Consider async transfers or unified memory

3. **Memory Allocation Times**
   - GPU memory allocation can be expensive
   - Consider memory pools or pre-allocation

4. **Memory Bandwidth Utilization**
   - Compare achieved vs theoretical peak bandwidth
   - RTX 3050: ~119 GB/s achieved is reasonable

5. **Arithmetic Intensity**
   - FLOPs per byte transferred
   - Low values (< 1.0) indicate memory-bound kernels

### Optimization Strategies Based on Profiling

**For Memory-Bound Kernels (like Euclidean Distance):**
- Optimize memory access patterns (coalescing)
- Use shared memory for data reuse
- Consider vectorized loads/stores
- Minimize memory allocations

**For Compute-Bound Kernels:**
- Optimize algorithm complexity
- Use specialized instructions (e.g., `__fmaf_rn`)
- Increase occupancy
- Balance work per thread

**For Transfer-Bound Applications:**
- Use asynchronous transfers with streams
- Consider unified memory (cudaMallocManaged)
- Overlap computation with communication
- Minimize host-device synchronization

## Common Profiling Issues and Solutions

### Issue 1: GPU Performance Counter Permissions
**Problem:** `ERR_NVGPUCTRPERM` error with Nsight Compute
**Solution:** 
- Use Nsight Systems instead for container environments
- For bare metal: `sudo modprobe nvidia` and proper driver setup
- Use enhanced manual timing as fallback

### Issue 2: Missing Kernel Data in Profiles
**Problem:** "SKIPPED: does not contain CUDA kernel data"
**Solution:**
- Ensure CUDA kernels actually execute (check for errors)
- Use `--trace=cuda` flag with nsys
- Verify GPU is accessible with `nvidia-smi`

### Issue 3: Large Profile Files
**Problem:** Nsight profile files become very large
**Solution:**
- Use `--duration=N` to limit capture time
- Profile representative workloads only
- Use `--sample=none` to disable CPU sampling if not needed

## Best Practices

### Development Workflow
1. **Start Simple:** Basic nvcc compilation and timing
2. **Add Profiling:** Compile with debug symbols
3. **Profile Timeline:** Use Nsight Systems for overview
4. **Detailed Analysis:** Create enhanced timing version
5. **Optimize:** Based on profiling insights
6. **Verify:** Re-profile to confirm improvements

### Code Instrumentation
```cpp
// Add timing instrumentation points
#define PROFILE_SECTION(name, code) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
        std::cout << name << ": " << duration.count() << " µs" << std::endl; \
    } while(0)

// Usage example
PROFILE_SECTION("GPU Memory Allocation", {
    cudaMalloc((void**)&d_data, size);
});
```

### Error Checking
Always include CUDA error checking in profiled code:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc((void**)&d_data, size));
```

## Conclusion

This profiling approach successfully identified that:
- **GPU memory allocation** was the primary bottleneck (71% of execution time)
- **Kernel execution** was highly optimized (only 0.1% of time)
- The workload is **memory-bound** (low arithmetic intensity)
- **Memory bandwidth utilization** is reasonable for the hardware

The combination of Nsight Systems timeline profiling and enhanced manual timing provides comprehensive performance insights even in containerized environments where advanced GPU profiling tools may have limited functionality.

## Quick Reference Commands

```bash
# Basic compilation and run
nvcc -o program program.cu && ./program

# Timeline profiling
nsys profile --trace=cuda ./program
nsys stats report1.nsys-rep

# Enhanced compilation with debug info
nvcc -g -lineinfo -O3 -o program program.cu

# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version
which nsys ncu
```