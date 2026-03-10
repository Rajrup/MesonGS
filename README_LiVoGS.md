## LiVoGS

Setup and running instructions for LiVoGS on VideoGS trained models.

## Installation

### Dependencies - NvComp

- Tested on Ubuntu 24.04 with CUDA 12.1
- NVComp 5.1.0.21 (optional, for GPU-accelerated lossless compression):
  - Download from (NvComp 5.1.0.21)[https://developer.nvidia.com/nvcomp-downloads]
  - Copy `include/*` to `$CUDA_ROOT/include/`.
  Example:

  ```bash
  cp -r nvcomp-linux-x86_64-5.1.0.21_cuda12-archive/include/* cuda-12.1/include/
  ```

  - Copy `lib/*` to `$CUDA_ROOT/lib64/`.
  Example:

  ```bash
  cp -r nvcomp-linux-x86_64-5.1.0.21_cuda12-archive/lib/* cuda-12.1/lib64/
  ```

## Setup LiVoGS

```bash
conda activate videogs


# Clone LiVoGS (no submodules)
git submodule add https://github.com/haodongw101/LiVoGS.git
cd LiVoGS

# Selectively clone only the PyRLGR pybind11 submodule
git submodule update --init compression/PyRLGR/thirdparty/pybind11

# Install Octree Compression
cd compression/Octree_Compression_GPU
make pybind

# Install RAHT-3DGS-codec
cd ../RAHT-3DGS-codec/cuda
pip install . --no-build-isolation

# Install PyRLGR
cd ../../PyRLGR
pip install . --no-build-isolation

# We won't install gsplat as we are using the VideoGS trained models.
cd ../../../
```