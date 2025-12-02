# Chapter 5: A Closer Look into the World of GPUs

This chapter provides comprehensive examples illustrating fundamental GPU programming concepts.

## Examples Overview

### 1. Thread, Block, and Grid Hierarchy (`1_thread_block_grid/`)

Demonstrates the CUDA execution model:

- Thread indexing within blocks
- Block organization within grids
- 1D, 2D, and 3D configurations
- Global thread ID calculation
- Visual hierarchy representation

**Build & Run:**

```bash
cd 1_thread_block_grid && mkdir build && cd build
cmake .. && make
./thread_hierarchy
```

### 2. Launch Configurations (`2_launch_configurations/`)

Explores different kernel launch configurations:

- Impact of threads per block on performance
- 1D vs 2D configurations
- Occupancy optimization
- Common launch patterns
- Best practices for configuration selection

**Build & Run:**

```bash
cd 2_launch_configurations && mkdir build && cd build
cmake .. && make
./launch_configs
```

### 3. Asynchronous Data Transfers (`3_async_transfers/`)

Shows memory transfer optimization:

- Synchronous vs asynchronous transfers
- Pinned (page-locked) memory
- Overlapping transfers with computation
- Performance comparison
- cudaMemcpyAsync usage

**Build & Run:**

```bash
cd 3_async_transfers && mkdir build && cd build
cmake .. && make
./async_transfers
```

### 4. Parallelizing with Streams (`4_streams/`)

Demonstrates concurrent execution:

- Multiple CUDA streams
- Overlapping kernel execution
- Stream dependencies with events
- Stream priorities
- Timeline visualization

**Build & Run:**

```bash
cd 4_streams && mkdir build && cd build
cmake .. && make
./streams
```

### 5. Following the Events (`5_events/`)

Illustrates event-based synchronization:

- Precise GPU timing
- Event recording and synchronization
- Inter-stream coordination
- Event flags and options
- Query event status

**Build & Run:**

```bash
cd 5_events && mkdir build && cd build
cmake .. && make
./events
```

### 6. Shared Memory Acceleration (`6_shared_memory/`)

Shows on-chip memory optimization:

- Matrix multiplication with/without shared memory
- Reduction operations
- Memory hierarchy comparison
- Bank conflict avoidance
- Performance improvements

**Build & Run:**

```bash
cd 6_shared_memory && mkdir build && cd build
cmake .. && make
./shared_memory
```

### 7. Device Capabilities Query (`7_device_query/`)

Queries and displays GPU hardware information:

- Compute capability
- Memory specifications
- Multiprocessor details
- Execution limits
- Feature support
- Occupancy calculations

**Build & Run:**

```bash
cd 7_device_query && mkdir build && cd build
cmake .. && make
./device_query
```

## Key Concepts Covered

### Thread Hierarchy

- **Thread**: Smallest execution unit, identified by threadIdx
- **Block**: Group of threads, identified by blockIdx, can share data via shared memory
- **Grid**: Collection of blocks executing the same kernel
- **Warps**: Groups of 32 threads that execute in lockstep

### Memory Hierarchy

- **Registers**: Fastest, private to each thread
- **Shared Memory**: Fast, shared within block (~1-32 cycles)
- **L1/L2 Cache**: Automatic caching
- **Global Memory**: Largest but slowest (~400-800 cycles)
- **Constant Memory**: Read-only, cached
- **Texture Memory**: Optimized for spatial locality

### Execution Model

- **Asynchronous Execution**: GPU operations don't block CPU
- **Streams**: Independent execution queues
- **Events**: Markers for timing and synchronization
- **Occupancy**: Active warps / max possible warps

### Optimization Strategies

1. Choose appropriate launch configuration
2. Maximize occupancy (but not at all costs)
3. Use shared memory for data reuse
4. Overlap transfers and computation with streams
5. Minimize global memory access
6. Ensure coalesced memory access
7. Avoid bank conflicts in shared memory

## Building All Examples

From the ch5 directory:

```bash
for dir in */; do
    cd "$dir"
    mkdir -p build && cd build
    cmake .. && make
    cd ../..
done
```

## Running All Examples

```bash
for dir in */; do
    cd "$dir/build"
    echo "=== Running ${dir} ==="
    ./*
    cd ../..
done
```

## Learning Path

1. **Start with**: `1_thread_block_grid` to understand the execution model
2. **Then**: `2_launch_configurations` to learn optimal configurations
3. **Next**: `7_device_query` to know your hardware
4. **Advanced**: `3_async_transfers` and `4_streams` for concurrency
5. **Timing**: `5_events` for precise measurements
6. **Optimization**: `6_shared_memory` for maximum performance

## Further Reading

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Nsight Compute: https://developer.nvidia.com/nsight-compute
