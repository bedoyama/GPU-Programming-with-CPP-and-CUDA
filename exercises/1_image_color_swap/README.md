# Image Color Processing Exercise

This exercise demonstrates CUDA-based image processing with two different color transformation algorithms.

## Programs

### 1. `color_swap` - Simple Channel Swapping
Performs deterministic color channel swapping:
- **R → G** (Red channel becomes Green)
- **G → B** (Green channel becomes Blue)  
- **B → R** (Blue channel becomes Red)

### 2. `color_swap_randomized` - Cyclic Coefficient Mixing
Performs weighted color mixing using a cyclic coefficient pattern:
- Generates random coefficients `[c1, c2, c3]` that sum to 1.0
- Applies them cyclically to each output channel:
  - **R_out = c1×R + c2×G + c3×B**
  - **G_out = c2×R + c3×G + c1×B** (coefficients shift)
  - **B_out = c3×R + c1×G + c2×B** (coefficients shift again)

## Prerequisites

1. **Download the stb libraries** (required for image I/O):

   ```bash
   chmod +x download_stb.sh
   ./download_stb.sh
   ```

   Alternatively, download manually:

   - [stb_image.h](https://raw.githubusercontent.com/nothings/stb/master/stb_image.h)
   - [stb_image_write.h](https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h)

2. **Input images** are available in `../input_data/` folder (21 sample images provided)

## Building

```bash
mkdir build
cd build
cmake ..
make
```

This builds both programs: `color_swap` and `color_swap_randomized`

## Running

### Simple Channel Swapping
```bash
./color_swap <input_image_path>
```

### Randomized Coefficient Mixing
```bash
# With random coefficients (different each run)
./color_swap_randomized <input_image_path>

# With fixed seed for consistent results (testing)
./color_swap_randomized <input_image_path> --fixed-seed
```

### Examples
```bash
# Process single image with simple swap
./color_swap ../input_data/image01.jpg

# Process with random coefficients
./color_swap_randomized ../input_data/image01.jpg

# Process with consistent coefficients (for testing)
./color_swap_randomized ../input_data/image01.jpg --fixed-seed
```

## Output Structure

Both programs generate organized output files in:
- `../output_data/gpu_out/` - GPU processing results
- `../output_data/cpu_out/` - CPU processing results (for verification)

### Output Files
- **color_swap**: `<filename>_swapped_gpu.jpg` and `<filename>_swapped_cpu.jpg`
- **color_swap_randomized**: `<filename>_randomized_gpu.jpg` and `<filename>_randomized_cpu.jpg`

### Console Output
Both programs display:
- Image dimensions and properties
- Coefficient information (randomized version shows the cyclic pattern)
- GPU and CPU processing times
- Performance speedup analysis
- Verification results comparing GPU and CPU outputs

## Example Output

### Simple Color Swap (`color_swap`)
```
==========================================
CUDA Image Color Channel Swap
Operation: R->G, G->B, B->R
==========================================

Image loaded successfully!
  Width: 1920 pixels
  Height: 1080 pixels
  Channels: 3
  Total pixels: 2073600
  Image size: 6220.8 KB

Kernel configuration:
  Threads per block: 16x16
  Blocks per grid: 120x68

Processing on GPU...
  GPU processing time: 0.245 ms
  GPU output saved to: ../output_data/gpu_out/image01_swapped_gpu.jpg

Processing on CPU...
  CPU processing time: 15.432 ms
  CPU output saved to: ../output_data/cpu_out/image01_swapped_cpu.jpg

==========================================
Performance Summary:
  GPU time: 0.245 ms
  CPU time: 15.432 ms
  Speedup: 63.0x
==========================================

Verifying GPU results against CPU...
  ✓ Verification PASSED! GPU and CPU outputs match perfectly.
```

### Randomized Coefficient Mixing (`color_swap_randomized`)
```
==========================================
CUDA Randomized Color Channel Mixing
Operation: Weighted random mixing
==========================================

Cyclic coefficient pattern (coefficients shift for each output channel):
  Using RANDOM SEED for unique results each run
  Base coefficients: [c1=0.364, c2=0.264, c3=0.372]

Applied cyclically to output channels:
  R_out = 0.364*R_in + 0.264*G_in + 0.372*B_in (sum=1.000)
  G_out = 0.264*R_in + 0.372*G_in + 0.364*B_in (sum=1.000)
  B_out = 0.372*R_in + 0.364*G_in + 0.264*B_in (sum=1.000)

Image loaded successfully!
  Width: 1440 pixels
  Height: 1863 pixels
  Channels: 3
  Total pixels: 2682720
  Image size: 7859.531 KB

Kernel configuration:
  Threads per block: 16x16
  Blocks per grid: 90x117

Processing on GPU...
  GPU processing time: 1.093 ms
  GPU output saved to: ../output_data/gpu_out/image01_randomized_gpu.jpg

Processing on CPU...
  CPU processing time: 81.303 ms
  CPU output saved to: ../output_data/cpu_out/image01_randomized_cpu.jpg

==========================================
Performance Summary:
  GPU time: 1.093 ms
  CPU time: 81.303 ms
  Speedup: 74.412x
==========================================

Verifying GPU results against CPU...
  ✓ Verification PASSED! GPU and CPU outputs match (within tolerance).
```

## Algorithm Details

### Cyclic Coefficient Pattern
The `color_swap_randomized` program uses a mathematically elegant cyclic pattern:
1. **Generate**: Three random coefficients `[c1, c2, c3]` normalized to sum = 1.0
2. **Apply Cyclically**: 
   - Output Red uses coefficients in order: `c1×R + c2×G + c3×B`
   - Output Green shifts by 1: `c2×R + c3×G + c1×B` 
   - Output Blue shifts by 2: `c3×R + c1×G + c2×B`
3. **Result**: Same transformation strength applied to all channels, just rotated

### Command Line Options
- **Default behavior**: Fully random coefficients (different each run)
- **`--fixed-seed`**: Uses seed=42 for consistent results (testing/debugging)

## Batch Processing

Process all sample images:
```bash
# Simple swapping on all images
for img in ../input_data/*.jpg; do ./color_swap "$img"; done

# Randomized mixing on all images  
for img in ../input_data/*.jpg; do ./color_swap_randomized "$img"; done

# Fixed seed for consistent testing
for img in ../input_data/*.jpg; do ./color_swap_randomized "$img" --fixed-seed; done
```

## Performance Characteristics

- **GPU Speedup**: Typically 15-80× faster than CPU
- **Memory Efficient**: Processes large images (up to 8MB+) efficiently
- **Scalable**: Performance scales with image resolution
- **Verified**: All GPU results validated against CPU implementations

## Notes

- Both programs support RGB and RGBA images (3 or 4 channels)
- Images with less than 3 channels will produce an error
- Alpha channel (if present) is preserved unchanged  
- JPEG quality is set to 95 for output images
- Output directories are created automatically if they don't exist
- Floating-point precision ensures accurate color transformations
