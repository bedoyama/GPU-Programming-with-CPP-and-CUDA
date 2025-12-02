#include <iostream>
#include <cuda_runtime.h>

// Kernel to demonstrate thread, block, and grid indexing
__global__ void threadBlockGridDemo(int *data, int width, int height) {
    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Block indices within grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    // Block dimensions
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int bdz = blockDim.z;
    
    // Grid dimensions
    int gdx = gridDim.x;
    int gdy = gridDim.y;
    int gdz = gridDim.z;
    
    // Calculate global thread ID (2D example)
    int col = tx + bx * bdx;  // Global X coordinate
    int row = ty + by * bdy;  // Global Y coordinate
    
    // Linear index in 2D array
    int idx = row * width + col;
    
    // Only process if within bounds
    if (col < width && row < height) {
        // Store global thread ID
        data[idx] = idx;
        
        // Print info for first few threads to demonstrate hierarchy
        if (bx == 0 && by == 0 && tx < 4 && ty < 4) {
            printf("Thread(%d,%d,%d) in Block(%d,%d,%d) | BlockDim(%d,%d,%d) | GridDim(%d,%d,%d) | Global(%d,%d) -> idx=%d\n",
                   tx, ty, tz, bx, by, bz, bdx, bdy, bdz, gdx, gdy, gdz, col, row, idx);
        }
    }
}

// Kernel to show 1D indexing pattern
__global__ void oneDimensionalPattern(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < N) {
        data[idx] = idx * idx;  // Store square of index
        
        if (idx < 10) {
            printf("1D: Thread %d in Block %d (BlockDim=%d) -> Global ID=%d, Value=%d\n",
                   threadIdx.x, blockIdx.x, blockDim.x, idx, idx * idx);
        }
    }
}

// Kernel to show 2D indexing pattern
__global__ void twoDimensionalPattern(int *data, int width, int height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        data[idx] = row * 1000 + col;  // Encode position
        
        if (row < 3 && col < 3) {
            printf("2D: Thread(%d,%d) in Block(%d,%d) -> Global(%d,%d) -> idx=%d\n",
                   threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, col, row, idx);
        }
    }
}

void printArray2D(int *data, int width, int height, const char *title) {
    std::cout << "\n" << title << ":" << std::endl;
    for (int i = 0; i < std::min(height, 8); i++) {
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < std::min(width, 8); j++) {
            std::cout << data[i * width + j] << " ";
        }
        if (width > 8) std::cout << "...";
        std::cout << std::endl;
    }
    if (height > 8) std::cout << "  ..." << std::endl;
}

int main() {
    std::cout << "========================================== " << std::endl;
    std::cout << "Thread, Block, and Grid Hierarchy Demo" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    // Example 1: 1D Grid and Blocks
    std::cout << "=== Example 1: 1D Configuration ===" << std::endl;
    int N = 32;
    int *d_data1, *h_data1;
    h_data1 = new int[N];
    cudaMalloc(&d_data1, N * sizeof(int));
    
    int threadsPerBlock1D = 8;
    int blocksPerGrid1D = (N + threadsPerBlock1D - 1) / threadsPerBlock1D;
    
    std::cout << "Array size: " << N << std::endl;
    std::cout << "Threads per block: " << threadsPerBlock1D << std::endl;
    std::cout << "Blocks in grid: " << blocksPerGrid1D << std::endl;
    std::cout << "Total threads: " << threadsPerBlock1D * blocksPerGrid1D << std::endl << std::endl;
    
    std::cout << "Kernel output:" << std::endl;
    oneDimensionalPattern<<<blocksPerGrid1D, threadsPerBlock1D>>>(d_data1, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data1, d_data1, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "\nResults (first 16 values):" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << "  data[" << i << "] = " << h_data1[i] << std::endl;
    }
    
    // Example 2: 2D Grid and Blocks
    std::cout << "\n\n=== Example 2: 2D Configuration ===" << std::endl;
    int width = 16, height = 12;
    int *d_data2, *h_data2;
    h_data2 = new int[width * height];
    cudaMalloc(&d_data2, width * height * sizeof(int));
    
    dim3 threadsPerBlock2D(4, 4);  // 16 threads per block in 4x4 arrangement
    dim3 blocksPerGrid2D(
        (width + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
        (height + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y
    );
    
    std::cout << "Array dimensions: " << width << "x" << height << std::endl;
    std::cout << "Threads per block: " << threadsPerBlock2D.x << "x" << threadsPerBlock2D.y 
              << " = " << threadsPerBlock2D.x * threadsPerBlock2D.y << " threads" << std::endl;
    std::cout << "Blocks in grid: " << blocksPerGrid2D.x << "x" << blocksPerGrid2D.y 
              << " = " << blocksPerGrid2D.x * blocksPerGrid2D.y << " blocks" << std::endl;
    std::cout << "Total threads: " << threadsPerBlock2D.x * threadsPerBlock2D.y * 
                                      blocksPerGrid2D.x * blocksPerGrid2D.y << std::endl << std::endl;
    
    std::cout << "Kernel output:" << std::endl;
    twoDimensionalPattern<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_data2, width, height);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data2, d_data2, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    printArray2D(h_data2, width, height, "2D Array Results");
    
    // Example 3: Detailed hierarchy visualization
    std::cout << "\n\n=== Example 3: Thread Hierarchy Visualization ===" << std::endl;
    int w = 8, h = 6;
    int *d_data3, *h_data3;
    h_data3 = new int[w * h];
    cudaMalloc(&d_data3, w * h * sizeof(int));
    
    dim3 threads(2, 2);  // Small blocks to see hierarchy clearly
    dim3 blocks(
        (w + threads.x - 1) / threads.x,
        (h + threads.y - 1) / threads.y
    );
    
    std::cout << "Grid Layout:" << std::endl;
    std::cout << "  Block dimensions: " << threads.x << "x" << threads.y << std::endl;
    std::cout << "  Grid dimensions: " << blocks.x << "x" << blocks.y << std::endl;
    std::cout << "  This creates a " << blocks.x << "x" << blocks.y << " grid of blocks," << std::endl;
    std::cout << "  each containing " << threads.x << "x" << threads.y << " threads" << std::endl << std::endl;
    
    std::cout << "Detailed thread information:" << std::endl;
    threadBlockGridDemo<<<blocks, threads>>>(d_data3, w, h);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data3, d_data3, w * h * sizeof(int), cudaMemcpyDeviceToHost);
    printArray2D(h_data3, w, h, "\nGlobal Thread IDs");
    
    // Visualize the block structure
    std::cout << "\n\nBlock Structure Visualization:" << std::endl;
    std::cout << "Each cell shows [block_x, block_y] containing thread IDs:" << std::endl;
    for (int by = 0; by < blocks.y; by++) {
        for (int ty = 0; ty < threads.y; ty++) {
            for (int bx = 0; bx < blocks.x; bx++) {
                std::cout << "[" << bx << "," << by << "]:{";
                for (int tx = 0; tx < threads.x; tx++) {
                    int global_x = bx * threads.x + tx;
                    int global_y = by * threads.y + ty;
                    if (global_x < w && global_y < h) {
                        int idx = global_y * w + global_x;
                        std::cout << h_data3[idx];
                        if (tx < threads.x - 1) std::cout << ",";
                    }
                }
                std::cout << "} ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Key takeaways
    std::cout << "\n========================================== " << std::endl;
    std::cout << "Key Concepts Summary:" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "1. THREAD: Smallest execution unit" << std::endl;
    std::cout << "   - Has threadIdx (x,y,z) within its block" << std::endl;
    std::cout << "   - Executes the kernel code" << std::endl << std::endl;
    
    std::cout << "2. BLOCK: Group of threads" << std::endl;
    std::cout << "   - Has blockIdx (x,y,z) within the grid" << std::endl;
    std::cout << "   - Has blockDim (x,y,z) - number of threads" << std::endl;
    std::cout << "   - Threads in a block can cooperate via shared memory" << std::endl;
    std::cout << "   - Max threads per block: typically 1024" << std::endl << std::endl;
    
    std::cout << "3. GRID: Collection of blocks" << std::endl;
    std::cout << "   - Has gridDim (x,y,z) - number of blocks" << std::endl;
    std::cout << "   - All blocks execute the same kernel" << std::endl;
    std::cout << "   - Blocks execute independently" << std::endl << std::endl;
    
    std::cout << "4. GLOBAL INDEX CALCULATION:" << std::endl;
    std::cout << "   1D: idx = threadIdx.x + blockIdx.x * blockDim.x" << std::endl;
    std::cout << "   2D: col = threadIdx.x + blockIdx.x * blockDim.x" << std::endl;
    std::cout << "       row = threadIdx.y + blockIdx.y * blockDim.y" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    // Cleanup
    delete[] h_data1;
    delete[] h_data2;
    delete[] h_data3;
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    
    return 0;
}
