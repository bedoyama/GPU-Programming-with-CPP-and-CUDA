# Image Color Channel Swap Exercise

This exercise demonstrates CUDA-based image processing by swapping color channels in a JPEG image.

## Operation

The program performs the following color channel swap:

- **R → G** (Red channel becomes Green)
- **G → B** (Green channel becomes Blue)
- **B → R** (Blue channel becomes Red)

## Prerequisites

1. **Download the stb libraries** (required for image I/O):

   ```bash
   chmod +x download_stb.sh
   ./download_stb.sh
   ```

   Alternatively, download manually:

   - [stb_image.h](https://raw.githubusercontent.com/nothings/stb/master/stb_image.h)
   - [stb_image_write.h](https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h)

2. **Place an input image** in the `../input_data/` folder named `image01.jpg`

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

From the build directory:

```bash
./color_swap
```

Or specify a custom input image:

```bash
./color_swap path/to/your/image.jpg
```

## Output

The program generates two output files in the build directory:

- `output_gpu.jpg` - Result from GPU processing
- `output_cpu.jpg` - Result from CPU processing (for verification)

The program also displays:

- Image dimensions and properties
- GPU and CPU processing times
- Performance speedup
- Verification results comparing GPU and CPU outputs

## Example Output

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
  GPU output saved to: output_gpu.jpg

Processing on CPU...
  CPU processing time: 15.432 ms
  CPU output saved to: output_cpu.jpg

==========================================
Performance Summary:
  GPU time: 0.245 ms
  CPU time: 15.432 ms
  Speedup: 63.0x
==========================================

Verifying GPU results against CPU...
  ✓ Verification PASSED! GPU and CPU outputs match perfectly.

Processing complete!
```

## Notes

- The program supports RGB and RGBA images (3 or 4 channels)
- Images with less than 3 channels will produce an error
- Alpha channel (if present) is preserved unchanged
- JPEG quality is set to 95 for output images
