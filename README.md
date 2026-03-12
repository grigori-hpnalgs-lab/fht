# FHT - Fast Hadamard Transform

High-performance Fast Hadamard Transform library with SIMD-optimized implementations for x86 (SSE/AVX) and ARM (NEON), and Python bindings via nanobind.

## Install

```bash
pip install fht
```

From source:

```bash
git clone https://github.com/yourorg/fht.git
cd fht
pip install .
```

## Python Usage

```python
import numpy as np
from fht import fht

# 1D transform (in-place)
x = np.random.randn(1024).astype(np.float32)
fht(x)

# Allocating mode (returns new array, original unchanged)
y = fht(x, inplace=False)

# Preallocated output
out = np.empty_like(x)
fht(x, out=out)

# 2D — each row transformed in parallel via OpenMP
X = np.random.randn(1000, 2**16).astype(np.float32)
fht(X, axis=-1)

# Complex arrays (decomposes into real/imag, transforms separately)
z = np.random.randn(512).astype(np.complex128)
fht(z)
```

Supported dtypes: `float32`, `float64`, `complex64`, `complex128`.

The transform axis must have a power-of-2 length. For 2D arrays, rows (or columns) are processed in parallel with OpenMP — set thread count via `OMP_NUM_THREADS=N`.

## C/C++ Usage

Header-only. Just include and compile:

```cpp
#include <fht/fht.h>

float buf[1024];
fht_float(buf, 10);  // log2(1024) = 10
```

### C API

```c
int fht_float(float *buf, int log_n);
int fht_double(double *buf, int log_n);
int fht_float_oop(float *in, float *out, int log_n);
int fht_double_oop(double *in, double *out, int log_n);
```

Returns 0 on success, 1 on invalid `log_n` (valid range: 0-30).

### CMake Integration

```cmake
# Via CPM (recommended)
CPMAddPackage("gh:yourorg/fht@1.0.0")
target_link_libraries(myapp PRIVATE fht::fht)

# Or as subdirectory
add_subdirectory(fht)
target_link_libraries(myapp PRIVATE fht::fht)
```

Compile with `-mavx` on x86 for best performance.

## Publishing Wheels

Prebuilt wheels for PyPI are built using [cibuildwheel](https://cibuildwheel.pypa.io/):

```bash
pip install cibuildwheel
cibuildwheel --output-dir dist/
```

A GitHub Actions workflow can automate this across platforms:

```yaml
# .github/workflows/wheels.yml
name: Build wheels
on:
  push:
    tags: ["v*"]
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-13, macos-14, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: pypa/cibuildwheel@v2.21
      - uses: actions/upload-artifact@v4
        with:
          path: wheelhouse/*.whl
  publish:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: artifact/
```

## Platform Support

| Platform | Float | Double | Notes |
|----------|-------|--------|-------|
| x86_64 + AVX | yes | yes | Best with `-mavx` |
| x86_64 + SSE | yes | yes | Fallback |
| ARM64 (NEON) | yes | yes | Pre-optimized for Apple M-series |

ARM double precision is ~1.6-2x slower than float (2 doubles vs 4 floats per 128-bit register).

## Re-optimizing for Your Hardware

The NEON kernels are pre-optimized for Apple M1. To tune for your specific ARM chip:

```bash
cd scripts
python optimize_v7_grid.py --quick --output ../include/fht/neon/fht_neon.h
```

Or via CMake:

```bash
cmake -B build -DFHT_OPTIMIZE_FOR_HOST=ON
cmake --build build
```

## Limitations

Known issues we plan to address:

- [ ] **`inplace=False` / `out=` does copy + in-place** — wastes one extra memory pass. A true out-of-place kernel (read from `in`, write to `out`) would avoid this.
- [ ] **Complex number support is slow** — currently ~4x slower than real. We deinterleave real/imag into separate buffers, transform each, then reinterleave. Fused complex kernels that handle the deinterleave inside the butterfly would close this gap.
- [ ] **No GPU support** — CUDA/Metal kernels are not yet available.

## Acknowledgments

The x86 AVX/SSE implementation is based on [FFHT](https://github.com/FALCONN-LIB/FFHT) from the [FALCONN](https://github.com/FALCONN-LIB/FALCONN) project by Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, and Ludwig Schmidt. The original code was copied and integrated with minor modifications.

The ARM NEON implementation was written from scratch with auto-tuned code generation.

## License

MIT. See [LICENSE](LICENSE).
