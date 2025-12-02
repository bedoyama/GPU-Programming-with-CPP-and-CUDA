#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <filesystem>
#include <sys/stat.h>

// CUDA kernel to swap color channels: R->G, G->B, B->R
__global__ void swapColorChannelsKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        
        if (channels == 3 || channels == 4) {
            // R -> G, G -> B, B -> R
            output[idx + 0] = input[idx + 2];  // New R = Old B
            output[idx + 1] = input[idx + 0];  // New G = Old R
            output[idx + 2] = input[idx + 1];  // New B = Old G
            
            // Keep alpha channel unchanged if present
            if (channels == 4) {
                output[idx + 3] = input[idx + 3];
            }
        }
    }
}

// CPU implementation for verification
void swapColorChannelsCpu(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            
            if (channels == 3 || channels == 4) {
                output[idx + 0] = input[idx + 2];  // New R = Old B
                output[idx + 1] = input[idx + 0];  // New G = Old R
                output[idx + 2] = input[idx + 1];  // New B = Old G
                
                if (channels == 4) {
                    output[idx + 3] = input[idx + 3];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Check if input path is provided as parameter
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../input_data/image01.jpg" << std::endl;
        std::cerr << "         " << argv[0] << " /path/to/your/image.jpg" << std::endl;
        return 1;
    }
    
    // Get input path from command line parameter
    const char *input_path = argv[1];
    
    // Generate output paths
    std::string input_str(input_path);
    std::string output_dir;
    std::string filename;
    std::string extension;
    
    // Extract directory and filename
    size_t last_slash = input_str.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        std::string input_dir = input_str.substr(0, last_slash + 1);
        // Replace input_data with output_data
        size_t input_data_pos = input_dir.find("input_data");
        if (input_data_pos != std::string::npos) {
            output_dir = input_dir.substr(0, input_data_pos) + "output_data/";
        } else {
            output_dir = input_dir + "output_data/";
        }
        filename = input_str.substr(last_slash + 1);
    } else {
        output_dir = "output_data/";
        filename = input_str;
    }
    
    // Extract name and extension
    size_t last_dot = filename.find_last_of(".");
    std::string base_name = (last_dot != std::string::npos) ? filename.substr(0, last_dot) : filename;
    extension = (last_dot != std::string::npos) ? filename.substr(last_dot) : ".jpg";
    
    // Create output directory if it doesn't exist
    mkdir(output_dir.c_str(), 0755);
    
    // Generate output file paths
    std::string output_path_gpu = output_dir + base_name + "_modified_gpu" + extension;
    std::string output_path_cpu = output_dir + base_name + "_modified_cpu" + extension;
    
    std::cout << "========================================== " << std::endl;
    std::cout << "CUDA Image Color Channel Swap" << std::endl;
    std::cout << "Operation: R->G, G->B, B->R" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    // Load image
    int width, height, channels;
    unsigned char *h_input = stbi_load(input_path, &width, &height, &channels, 0);
    
    if (!h_input) {
        std::cerr << "Error: Could not load image from " << input_path << std::endl;
        std::cerr << "Please ensure the image file exists in the input_data folder." << std::endl;
        return 1;
    }
    
    std::cout << "Image loaded successfully!" << std::endl;
    std::cout << "  Width: " << width << " pixels" << std::endl;
    std::cout << "  Height: " << height << " pixels" << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Total pixels: " << width * height << std::endl;
    std::cout << "  Image size: " << (width * height * channels) / 1024.0 << " KB" << std::endl << std::endl;
    
    if (channels < 3) {
        std::cerr << "Error: Image must have at least 3 channels (RGB)" << std::endl;
        stbi_image_free(h_input);
        return 1;
    }
    
    size_t image_size = width * height * channels * sizeof(unsigned char);
    
    // Allocate host memory for output
    unsigned char *h_output_gpu = (unsigned char *)malloc(image_size);
    unsigned char *h_output_cpu = (unsigned char *)malloc(image_size);
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, image_size);
    cudaMalloc((void **)&d_output, image_size);
    
    // Copy input image to device
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    std::cout << "Kernel configuration:" << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock.x << "x" << threadsPerBlock.y << std::endl;
    std::cout << "  Blocks per grid: " << blocksPerGrid.x << "x" << blocksPerGrid.y << std::endl << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    // ===================== GPU Processing =====================
    std::cout << "Processing on GPU..." << std::endl;
    
    cudaEventRecord(startEvent, 0);
    swapColorChannelsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, channels);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
    
    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, image_size, cudaMemcpyDeviceToHost);
    
    std::cout << "  GPU processing time: " << gpuTime << " ms" << std::endl;
    
    // Save GPU output
    if (stbi_write_jpg(output_path_gpu.c_str(), width, height, channels, h_output_gpu, 95)) {
        std::cout << "  GPU output saved to: " << output_path_gpu << std::endl;
    } else {
        std::cerr << "  Error: Could not save GPU output image" << std::endl;
    }
    std::cout << std::endl;
    
    // ===================== CPU Processing =====================
    std::cout << "Processing on CPU..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    swapColorChannelsCpu(h_input, h_output_cpu, width, height, channels);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = (stop - start);
    
    std::cout << "  CPU processing time: " << cpuTime.count() << " ms" << std::endl;
    
    // Save CPU output
    if (stbi_write_jpg(output_path_cpu.c_str(), width, height, channels, h_output_cpu, 95)) {
        std::cout << "  CPU output saved to: " << output_path_cpu << std::endl;
    } else {
        std::cerr << "  Error: Could not save CPU output image" << std::endl;
    }
    std::cout << std::endl;
    
    // ===================== Performance Comparison =====================
    std::cout << "========================================== " << std::endl;
    std::cout << "Performance Summary:" << std::endl;
    std::cout << "  GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "  CPU time: " << cpuTime.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << cpuTime.count() / gpuTime << "x" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    // ===================== Verification =====================
    std::cout << "Verifying GPU results against CPU..." << std::endl;
    
    bool resultsMatch = true;
    int mismatchCount = 0;
    const int maxMismatchesToShow = 10;
    
    for (size_t i = 0; i < width * height * channels; i++) {
        if (h_output_gpu[i] != h_output_cpu[i]) {
            resultsMatch = false;
            if (mismatchCount < maxMismatchesToShow) {
                std::cout << "  Mismatch at byte " << i << ": GPU = " << (int)h_output_gpu[i]
                         << ", CPU = " << (int)h_output_cpu[i] << std::endl;
            }
            mismatchCount++;
        }
    }
    
    if (resultsMatch) {
        std::cout << "  ✓ Verification PASSED! GPU and CPU outputs match perfectly." << std::endl;
    } else {
        std::cout << "  ✗ Verification FAILED! Found " << mismatchCount << " mismatches." << std::endl;
    }
    
    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output_gpu);
    free(h_output_cpu);
    stbi_image_free(h_input);
    
    std::cout << "\nProcessing complete!" << std::endl;
    
    return 0;
}
