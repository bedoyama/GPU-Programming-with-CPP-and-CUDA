#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "\n========================================== " << std::endl;
    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    // Compute Capability
    std::cout << "=== Compute Capability ===" << std::endl;
    std::cout << "  Version: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Unified addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
    std::cout << "  Managed memory: " << (prop.managedMemory ? "Yes" : "No") << std::endl << std::endl;
    
    // Memory
    std::cout << "=== Memory ===" << std::endl;
    std::cout << "  Total global memory: " << std::fixed << std::setprecision(2)
              << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "  Total constant memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "  L2 cache size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    float bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    std::cout << "  Peak memory bandwidth: " << std::fixed << std::setprecision(1)
              << bandwidth << " GB/s" << std::endl << std::endl;
    
    // Multiprocessors
    std::cout << "=== Multiprocessors ===" << std::endl;
    std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    
    // Calculate theoretical CUDA cores (varies by architecture)
    int coresPerSM = 0;
    if (prop.major == 3) coresPerSM = 192;
    else if (prop.major == 5) coresPerSM = 128;
    else if (prop.major == 6) coresPerSM = (prop.minor == 0) ? 64 : 128;
    else if (prop.major == 7) coresPerSM = 64;
    else if (prop.major == 8) coresPerSM = (prop.minor == 0) ? 64 : 128;
    else if (prop.major == 9) coresPerSM = 128;
    
    if (coresPerSM > 0) {
        std::cout << "  CUDA cores per SM: " << coresPerSM << " (estimated)" << std::endl;
        std::cout << "  Total CUDA cores: " << coresPerSM * prop.multiProcessorCount 
                  << " (estimated)" << std::endl;
    }
    
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Max warps per SM: " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl << std::endl;
    
    // Execution Configuration
    std::cout << "=== Execution Configuration Limits ===" << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max thread dimensions: [" << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "  Max grid dimensions: [" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl << std::endl;
    
    // Registers
    std::cout << "=== Registers ===" << std::endl;
    std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor << std::endl << std::endl;
    
    // Features
    std::cout << "=== Features ===" << std::endl;
    std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  Async engine count: " << prop.asyncEngineCount << std::endl;
    std::cout << "  ECC enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    std::cout << "  Integrated GPU: " << (prop.integrated ? "Yes" : "No") << std::endl;
    std::cout << "  Can map host memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
    std::cout << "  Compute mode: ";
    switch (prop.computeMode) {
        case cudaComputeModeDefault:
            std::cout << "Default (multiple threads can use)" << std::endl;
            break;
        case cudaComputeModeExclusive:
            std::cout << "Exclusive (only one thread can use)" << std::endl;
            break;
        case cudaComputeModeProhibited:
            std::cout << "Prohibited (no CUDA allowed)" << std::endl;
            break;
        case cudaComputeModeExclusiveProcess:
            std::cout << "Exclusive Process" << std::endl;
            break;
    }
    
    std::cout << "  Kernel timeout enabled: " << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
    std::cout << "  Cooperative launch: " << (prop.cooperativeLaunch ? "Yes" : "No") << std::endl;
    std::cout << "  Multi-device cooperative launch: " << (prop.cooperativeMultiDeviceLaunch ? "Yes" : "No") << std::endl << std::endl;
    
    // Texture and Surface
    std::cout << "=== Texture and Surface ===" << std::endl;
    std::cout << "  Max 1D texture size: " << prop.maxTexture1D << std::endl;
    std::cout << "  Max 2D texture dimensions: [" << prop.maxTexture2D[0] << ", " 
              << prop.maxTexture2D[1] << "]" << std::endl;
    std::cout << "  Max 3D texture dimensions: [" << prop.maxTexture3D[0] << ", " 
              << prop.maxTexture3D[1] << ", " << prop.maxTexture3D[2] << "]" << std::endl;
    std::cout << "  Texture alignment: " << prop.textureAlignment << " bytes" << std::endl;
    std::cout << "  Surface alignment: " << prop.surfaceAlignment << " bytes" << std::endl << std::endl;
    
    // PCI
    std::cout << "=== PCI ===" << std::endl;
    std::cout << "  PCI bus ID: " << prop.pciBusID << std::endl;
    std::cout << "  PCI device ID: " << prop.pciDeviceID << std::endl;
    std::cout << "  PCI domain ID: " << prop.pciDomainID << std::endl << std::endl;
}

void printMemoryInfo() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    std::cout << "=== Current Memory Usage ===" << std::endl;
    std::cout << "  Total memory: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Free memory:  " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Used memory:  " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl << std::endl;
}

void printOccupancyExample() {
    std::cout << "=== Occupancy Calculator Example ===" << std::endl << std::endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int blockSizes[] = {64, 128, 256, 512, 1024};
    
    std::cout << "Example: Kernel with no shared memory, 32 registers per thread" << std::endl;
    std::cout << std::setw(15) << "Block Size" 
              << std::setw(15) << "Blocks/SM" 
              << std::setw(15) << "Warps/SM"
              << std::setw(15) << "Occupancy" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int blockSize : blockSizes) {
        if (blockSize > prop.maxThreadsPerBlock) continue;
        
        int blocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;
        if (blocksPerSM > prop.maxBlocksPerMultiProcessor) {
            blocksPerSM = prop.maxBlocksPerMultiProcessor;
        }
        
        int warpsPerSM = (blocksPerSM * blockSize) / prop.warpSize;
        int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        float occupancy = (float)warpsPerSM / maxWarpsPerSM * 100.0f;
        
        std::cout << std::setw(15) << blockSize
                  << std::setw(15) << blocksPerSM
                  << std::setw(15) << warpsPerSM
                  << std::setw(14) << std::fixed << std::setprecision(1) << occupancy << "%"
                  << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================== " << std::endl;
    std::cout << "CUDA Device Capabilities Query" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    // Get number of devices
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found!" << std::endl;
        return 0;
    }
    
    std::cout << "\nFound " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get CUDA runtime version
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." 
              << (runtimeVersion % 100) / 10 << std::endl;
    
    // Get CUDA driver version
    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." 
              << (driverVersion % 100) / 10 << std::endl;
    
    // Print properties for each device
    for (int i = 0; i < deviceCount; i++) {
        printDeviceProperties(i);
    }
    
    // Current device memory info
    printMemoryInfo();
    
    // Occupancy example
    printOccupancyExample();
    
    // Key takeaways
    std::cout << "========================================== " << std::endl;
    std::cout << "Understanding Hardware Capabilities" << std::endl;
    std::cout << "========================================== " << std::endl;
    std::cout << "1. COMPUTE CAPABILITY:" << std::endl;
    std::cout << "   - Defines feature set (X.Y version)" << std::endl;
    std::cout << "   - Higher = newer features, better performance" << std::endl << std::endl;
    
    std::cout << "2. MEMORY HIERARCHY:" << std::endl;
    std::cout << "   - Know your limits: global, shared, constant" << std::endl;
    std::cout << "   - Bandwidth is often the bottleneck" << std::endl << std::endl;
    
    std::cout << "3. EXECUTION LIMITS:" << std::endl;
    std::cout << "   - Max threads/block: typically 1024" << std::endl;
    std::cout << "   - Max blocks/SM: affects occupancy" << std::endl;
    std::cout << "   - Max registers: affects occupancy" << std::endl << std::endl;
    
    std::cout << "4. OCCUPANCY:" << std::endl;
    std::cout << "   - % of max warps that can be active" << std::endl;
    std::cout << "   - Limited by: threads/block, shared mem, registers" << std::endl;
    std::cout << "   - Higher occupancy -> better latency hiding" << std::endl;
    std::cout << "   - But: more isn't always better!" << std::endl << std::endl;
    
    std::cout << "5. FEATURES TO CHECK:" << std::endl;
    std::cout << "   - Concurrent kernels: multiple kernels at once" << std::endl;
    std::cout << "   - Async engines: copy + compute overlap" << std::endl;
    std::cout << "   - Managed memory: unified memory support" << std::endl << std::endl;
    
    std::cout << "6. OPTIMIZATION STRATEGY:" << std::endl;
    std::cout << "   - Query capabilities at runtime" << std::endl;
    std::cout << "   - Adapt launch configuration to device" << std::endl;
    std::cout << "   - Use cudaOccupancyMaxPotentialBlockSize()" << std::endl;
    std::cout << "   - Profile to find actual bottlenecks" << std::endl;
    std::cout << "========================================== " << std::endl;
    
    return 0;
}
