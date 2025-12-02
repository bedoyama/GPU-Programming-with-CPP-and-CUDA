# Color Swap CUDA Program Usage

## Description
This program swaps color channels in an image using CUDA parallel processing:
- Red → Green  
- Green → Blue
- Blue → Red

## Updated Usage

The program now **requires** an input image path as a command line parameter.

### Command Syntax
```bash
./color_swap <input_image_path>
```

### Examples

```bash
# Using an image from the input_data folder
./color_swap ../input_data/570049770_1342796557394278_76461342431256016_n.jpg

# Using an absolute path
./color_swap /path/to/your/image.jpg

# Using a relative path
./color_swap ../my_image.png
```

### Error Handling
If no input path is provided, the program will show:
```
Usage: ./color_swap <input_image_path>
Example: ./color_swap ../input_data/image01.jpg
         ./color_swap /path/to/your/image.jpg
```

### Output Files
The program automatically creates an `output_data` folder at the same level as `input_data` and generates two output files with the original filename plus `_modified` suffix:

- `originalname_modified_gpu.jpg` - Result from GPU processing
- `originalname_modified_cpu.jpg` - Result from CPU processing (for verification)

**Example:**
- Input: `../../input_data/my_photo.jpg`
- Output: `../../output_data/my_photo_modified_gpu.jpg` and `../../output_data/my_photo_modified_cpu.jpg`

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png) 
- BMP (.bmp)
- TGA (.tga)
- And other formats supported by stb_image

### Performance Information
The program displays:
- Image dimensions and file size
- GPU vs CPU processing times
- Performance speedup comparison
- Verification results (GPU vs CPU output matching)

## Changes Made
- Removed default hardcoded input path
- Made input path a required command line parameter
- Added clear usage instructions and error messages
- Improved user experience with better parameter validation