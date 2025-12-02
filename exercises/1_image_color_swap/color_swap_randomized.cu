#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <filesystem>
#include <sys/stat.h>

// CUDA kernel for randomized color channel mixing
// Each output channel is a weighted sum of all input channels
// Weights are random but sum to 1.0
__global__ void randomizedChannelMixKernel(unsigned char *input, unsigned char *output, 
                                           float *weights, int width, int height, int channels) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        
        if (channels == 3 || channels == 4) {
            // For each output channel, compute weighted sum of input channels
            for (int out_ch = 0; out_ch < 3; out_ch++) {
                float sum = 0.0f;
                for (int in_ch = 0; in_ch < 3; in_ch++) {
                    // weights[out_ch * 3 + in_ch] gives weight for input channel in_ch to output channel out_ch
                    sum += input[idx + in_ch] * weights[out_ch * 3 + in_ch];
                }
                // Clamp to valid range [0, 255]
                output[idx + out_ch] = (unsigned char)(fminf(fmaxf(sum, 0.0f), 255.0f));
            }
            
            // Keep alpha channel unchanged if present
            if (channels == 4) {
                output[idx + 3] = input[idx + 3];
            }
        }
    }
}

// CPU implementation for verification
void randomizedChannelMixCpu(unsigned char *input, unsigned char *output, 
                             float *weights, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            
            if (channels == 3 || channels == 4) {
                for (int out_ch = 0; out_ch < 3; out_ch++) {
                    float sum = 0.0f;
                    for (int in_ch = 0; in_ch < 3; in_ch++) {
                        sum += input[idx + in_ch] * weights[out_ch * 3 + in_ch];
                    }
                    output[idx + out_ch] = (unsigned char)(fminf(fmaxf(sum, 0.0f), 255.0f));
                }
                
                if (channels == 4) {
                    output[idx + 3] = input[idx + 3];
                }
            }
        }
    }
}

// Generate random coefficients that are applied cyclically to each output channel
// newR = c1*R + c2*G + c3*B
// newG = c1*G + c2*B + c3*R  (coefficients shift)
// newB = c1*B + c2*R + c3*G  (coefficients shift again)
void generateRandomWeights(float *weights, bool use_fixed_seed = false) {
    std::mt19937 gen;
    if (use_fixed_seed) {
        gen.seed(42);  // Fixed seed for consistent results
    } else {
        std::random_device rd;
        gen.seed(rd()); // True randomness
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Generate one set of random coefficients [c1, c2, c3]
    float coeffs[3];
    float sum = 0.0f;
    
    // Generate random coefficients
    for (int i = 0; i < 3; i++) {
        coeffs[i] = dist(gen);
        sum += coeffs[i];
    }
    
    // Normalize so they sum to 1.0
    for (int i = 0; i < 3; i++) {
        coeffs[i] = coeffs[i] / sum;
    }
    
    // Apply coefficients cyclically to each output channel
    for (int out_ch = 0; out_ch < 3; out_ch++) {
        for (int in_ch = 0; in_ch < 3; in_ch++) {
            // Cyclic mapping: output channel shifts which coefficient applies to which input
            int coeff_idx = (in_ch + out_ch) % 3;
            weights[out_ch * 3 + in_ch] = coeffs[coeff_idx];
        }
    }
}

void printWeights(float *weights, bool use_fixed_seed = false) {
    const char* channel_names[] = {"R", "G", "B"};
    
    // First show the base coefficients
    std::cout << "Cyclic coefficient pattern (coefficients shift for each output channel):" << std::endl;
    if (use_fixed_seed) {
        std::cout << "  Using FIXED SEED (seed=42) for consistent results" << std::endl;
    } else {
        std::cout << "  Using RANDOM SEED for unique results each run" << std::endl;
    }
    std::cout << "  Base coefficients: [c1=" << std::fixed << std::setprecision(3) 
              << weights[0] << ", c2=" << weights[1] << ", c3=" << weights[2] << "]" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Applied cyclically to output channels:" << std::endl;
    
    for (int out_ch = 0; out_ch < 3; out_ch++) {
        std::cout << "  " << channel_names[out_ch] << "_out = ";
        float sum = 0.0f;
        for (int in_ch = 0; in_ch < 3; in_ch++) {
            float w = weights[out_ch * 3 + in_ch];
            sum += w;
            if (in_ch > 0) std::cout << " + ";
            std::cout << w << "*" << channel_names[in_ch] << "_in";
        }
        std::cout << " (sum=" << sum << ")" << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    bool use_fixed_seed = false;
    const char *input_path = nullptr;
    
    // Check if input path is provided as parameter
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path> [--fixed-seed]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --fixed-seed    Use fixed seed for consistent results (for testing)" << std::endl;
        std::cerr << "Examples:" << std::endl;
        std::cerr << "  " << argv[0] << " ../input_data/image01.jpg" << std::endl;
        std::cerr << "  " << argv[0] << " ../input_data/image01.jpg --fixed-seed" << std::endl;
        return 1;
    }
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--fixed-seed") {
            use_fixed_seed = true;
        } else if (input_path == nullptr) {
            input_path = argv[i];
        }
    }
    
    if (input_path == nullptr) {
        std::cerr << "Error: No input image path provided" << std::endl;
        return 1;
    }
    
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
    
    // Create output directory structure if it doesn't exist
    mkdir(output_dir.c_str(), 0755);
    std::string gpu_out_dir = output_dir + "gpu_out/";
    std::string cpu_out_dir = output_dir + "cpu_out/";
    mkdir(gpu_out_dir.c_str(), 0755);
    mkdir(cpu_out_dir.c_str(), 0755);
    
    // Generate output file paths with organized folder structure
    std::string output_path_gpu = gpu_out_dir + base_name + "_randomized_gpu" + extension;
    std::string output_path_cpu = cpu_out_dir + base_name + "_randomized_cpu" + extension;
    
    std::cout << "========================================== " << std::endl;
    std::cout << "CUDA Randomized Color Channel Mixing" << std::endl;
    std::cout << "Operation: Weighted random mixing" << std::endl;
    std::cout << "========================================== " << std::endl << std::endl;
    
    // Generate random weights (3x3 matrix: 3 output channels, each from 3 input channels)
    float h_weights[9];
    generateRandomWeights(h_weights, use_fixed_seed);
    printWeights(h_weights, use_fixed_seed);
    
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
    float *d_weights;
    cudaMalloc((void **)&d_input, image_size);
    cudaMalloc((void **)&d_output, image_size);
    cudaMalloc((void **)&d_weights, 9 * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, 9 * sizeof(float), cudaMemcpyHostToDevice);
    
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
    randomizedChannelMixKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_weights, width, height, channels);
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
    randomizedChannelMixCpu(h_input, h_output_cpu, h_weights, width, height, channels);
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
        // Allow small differences due to floating point rounding
        int diff = abs((int)h_output_gpu[i] - (int)h_output_cpu[i]);
        if (diff > 1) {
            resultsMatch = false;
            if (mismatchCount < maxMismatchesToShow) {
                std::cout << "  Mismatch at byte " << i << ": GPU = " << (int)h_output_gpu[i]
                         << ", CPU = " << (int)h_output_cpu[i] << ", diff = " << diff << std::endl;
            }
            mismatchCount++;
        }
    }
    
    if (resultsMatch) {
        std::cout << "  ✓ Verification PASSED! GPU and CPU outputs match (within tolerance)." << std::endl;
    } else {
        std::cout << "  ✗ Verification FAILED! Found " << mismatchCount << " significant mismatches." << std::endl;
    }
    
    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);
    free(h_output_gpu);
    free(h_output_cpu);
    stbi_image_free(h_input);
    
    std::cout << "\nProcessing complete!" << std::endl;
    
    return 0;
}
