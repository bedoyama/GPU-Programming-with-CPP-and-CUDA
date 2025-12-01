# GPU Profiling Quick Reference

## Compilation Commands

```bash
# Basic compilation
nvcc -o program program.cu

# With profiling symbols
nvcc -g -lineinfo -O3 -o program program.cu

# CMake build
mkdir build && cd build
cmake .. && make
```

## Profiling Commands

```bash
# Timeline profiling (recommended for containers)
nsys profile --trace=cuda ./program

# Generate statistics
nsys stats report.nsys-rep
nsys stats --format csv report.nsys-rep

# Kernel profiling (limited in containers)
ncu --set basic ./program
sudo ncu --set basic ./program  # try with sudo
```

## Key Files

- **Original:** `euclidean_distance.cu` - Basic program
- **Enhanced:** `euclidean_distance_profiled.cu` - Detailed timing analysis
- **Documentation:** `PROFILING_GUIDE.md` - Complete profiling guide

## Expected Results (Euclidean Distance)

- **Data Size:** 10M points, 267 MB GPU memory
- **Performance:** ~4.5 billion points/second
- **Bottleneck:** GPU memory allocation (71% of time)
- **Kernel Time:** Only ~2.2ms (highly optimized)
- **Memory Bandwidth:** ~119 GB/s

## Troubleshooting

```bash
# Check GPU status
nvidia-smi

# Verify tools
which nvcc nsys ncu

# Permission issues - use Nsight Systems instead of Nsight Compute
# Missing kernel data - ensure CUDA kernels execute without errors
```