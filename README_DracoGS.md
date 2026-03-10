## Build Draco for 3DGS

```bash
git clone --recursive https://github.com/Rajrup/DracoGS.git DracoGS
cd DracoGS

conda activate mesongs
pip install numpy==1.26.4
pip install pybind11 plyfile

mkdir build && cd build
cmake .. -DDRACOGS_BUILD_PYTHON=ON -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
make -j$(nproc)
```

Notes:
```
Higher QP = more precision = better quality = less compression (bigger files)
qp: Quantization parameter for position (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
qfd: Quantization parameter for SH0 (DC) (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
qfr1: Quantization parameter for SH1 (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
qfr2: Quantization parameter for SH2 (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
qfr3: Quantization parameter for SH3 (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
qo: Quantization parameter for opacity (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
qs: Quantization parameter for scale (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
qr: Quantization parameter for rotation (0 - 30, where 0 is lossless-noquantization and 30 means 30 bit precision, Default: 16)
cl: Compression level (0 - 10, 0 is fastest-least compression and 10 is slowest-most compression, Default: 7)
```

## Python API (In-Memory Pipeline)

The Python module eliminates intermediate disk I/O by compressing and decompressing entirely in memory using numpy arrays.

### Usage

```python
import sys
sys.path.insert(0, "build/compression")   # or wherever _dracogs.so lives
sys.path.insert(0, "compression")

from compression_decompression import read_gs_ply, encode_dracogs, decode_dracogs, save_gs_ply

# 1. Read a 3DGS PLY (binary or ASCII) -> dict of numpy arrays
gs = read_gs_ply("input.ply")

# 2. Compress in memory -> bytes
bitstream = encode_dracogs(gs, qp=16, qfd=8, qfr1=8, qfr2=8, qfr3=8, qo=8, qs=8, qr=8, cl=7)
print(f"Compressed: {len(bitstream)} bytes")

# 3. Decompress in memory -> dict of numpy arrays
gs_decoded = decode_dracogs(bitstream)

# 4. Save to PLY
save_gs_ply(gs_decoded, "output.ply")

# Optionally save/load the bitstream to/from disk
with open("compressed.drc", "wb") as f:
    f.write(bitstream)
```

### API Reference

| Function | Description |
|----------|-------------|
| `read_gs_ply(path)` | Read 3DGS PLY, return `dict` of numpy float32 arrays |
| `encode_dracogs(gs_data, qp=16, ...)` | Compress to Draco bitstream (`bytes`) |
| `decode_dracogs(bitstream)` | Decompress from Draco bitstream to `dict` |
| `save_gs_ply(gs_data, path)` | Write `dict` back to PLY |