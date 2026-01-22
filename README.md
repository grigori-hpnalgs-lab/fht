# FHT - Fast Hadamard Transform Library

A portable, high-performance Fast Hadamard Transform (FHT) library with optimized implementations for x86 (SSE/AVX) and ARM (NEON).

## Features

- **Cross-platform**: Works on x86 (32/64-bit) and ARM64
- **Header-only**: No compilation required, just include
- **High performance**: SIMD-optimized for both x86 (SSE/AVX) and ARM (NEON)
- **Optional re-optimization**: Tune NEON kernels for your specific ARM hardware
- **CMake integration**: Easy to integrate via `find_package` or CPM
- **MIT licensed**: Free for commercial use

## Quick Start

### Header-only Usage

```cpp
#include <fht/fht.h>

int main() {
    float buf[1024];  // Must be power of 2
    // ... fill buf with data ...

    int result = fht_float(buf, 10);  // log2(1024) = 10
    if (result != 0) {
        // Handle error
    }

    return 0;
}
```

### CMake Integration

#### Option 1: Subdirectory

```cmake
add_subdirectory(fht)
target_link_libraries(myapp PRIVATE fht::fht)
```

#### Option 2: find_package

```cmake
find_package(fht REQUIRED)
target_link_libraries(myapp PRIVATE fht::fht)
```

#### Option 3: CPM (Recommended)

```cmake
include(cmake/CPM.cmake)

CPMAddPackage(
    NAME fht
    GITHUB_REPOSITORY yourorg/fht
    VERSION 1.0.0
)

target_link_libraries(myapp PRIVATE fht::fht)
```

## API Reference

### In-place Transform

```c
int fht_float(float *buf, int log_n);
int fht_double(double *buf, int log_n);
```

- `buf`: Array of 2^log_n elements
- `log_n`: Log base 2 of array size (0 to 30)
- Returns: 0 on success, 1 on invalid log_n

### Out-of-place Transform

```c
int fht_float_oop(float *in, float *out, int log_n);
int fht_double_oop(double *in, double *out, int log_n);
```

### C++ Overloads

```cpp
int fht(float *buf, int log_n);
int fht(double *buf, int log_n);
int fht(float *in, float *out, int log_n);
int fht(double *in, double *out, int log_n);
```

## Platform Support

| Platform | Single Precision | Double Precision | Notes |
|----------|------------------|------------------|-------|
| x86_64 + AVX | Yes | Yes | Best performance with `-mavx` |
| x86_64 + SSE | Yes | Yes | Fallback if AVX unavailable |
| x86 (32-bit) | Yes | Yes | SSE2 required |
| ARM64 (NEON) | Yes | Yes | Pre-optimized for Apple M1 |

**Note:** ARM double precision is ~1.6-2x slower than float due to half the SIMD width (2 doubles vs 4 floats per 128-bit register).

## Re-optimizing for Your Hardware (ARM)

The default NEON implementation is pre-optimized for Apple M1. To optimize for your specific ARM hardware:

### Option 1: CMake

```bash
mkdir build && cd build
cmake .. -DFHT_OPTIMIZE_FOR_HOST=ON
cmake --build . --target fht_optimize
```

### Option 2: Manual

```bash
cd scripts

# Optimize float only (default)
python optimize_v7_grid.py --quick --output ../include/fht/neon/fht_neon.h

# Optimize both float and double
python optimize_v7_grid.py --quick --precision both --output ../include/fht/neon/fht_neon.h

# Optimize double only
python optimize_v7_grid.py --quick --precision double --output ../include/fht/neon/fht_neon.h
```

The optimization process benchmarks different algorithm parameters on your hardware and generates a customized header file.

## Building Tests

Tests use GoogleTest (fetched automatically via CPM).

### Using CMake Presets (Recommended)

```bash
# Configure with Homebrew Clang + tests
cmake --preset dev

# Build
cmake --build build

# Run tests
ctest --test-dir build
```

Available presets:
- `default` - Release build with Homebrew Clang
- `dev` - Development build with tests
- `dev-optimize` - Development build with tests and host optimization
- `appleclang` - Use system AppleClang instead

### Manual Configuration

```bash
mkdir build && cd build
cmake .. -DFHT_BUILD_TESTS=ON
cmake --build .
ctest
```

### Test Options

```bash
# Limit max test size (default: 26, i.e., 64M elements)
cmake .. -DFHT_BUILD_TESTS=ON -DFHT_TEST_MAX_LOG_N=20

# Run tests with verbose output
ctest --output-on-failure

# Run specific test
ctest -R "log_n_10"

# Run only float tests
ctest -R "FHTFloatTest"

# Run only double tests
ctest -R "FHTDoubleTest"
```

### Test Coverage

The test suite includes:
- **Reference comparison**: Tests against a simple recursive implementation for all sizes
- **Inverse property**: Verifies FHT(FHT(x)) = N*x
- **Edge cases**: Invalid inputs, log_n=0, log_n=1, log_n=2
- **Large sizes**: Tests up to log_n=28 (256M elements) where memory permits
- **Out-of-place**: Verifies OOP matches in-place results

## Directory Structure

```
fht/
├── include/fht/           # Public headers
│   ├── fht.h              # Main header (include this)
│   ├── fht_config.h       # Platform detection
│   ├── x86/               # x86 SSE/AVX implementation
│   └── neon/              # ARM NEON implementation
├── tests/                 # Test suite
│   ├── test_fht.cpp       # GoogleTest comprehensive tests
│   ├── test_basic.cpp     # Quick sanity check
│   └── fht_reference.h    # Reference implementation for testing
├── scripts/               # Optimization tools
│   ├── gen_neon_v7.py     # Code generator
│   └── optimize_v7_grid.py # Grid search optimizer
├── data/                  # Pre-computed optimization data
├── cmake/                 # CMake helpers
│   ├── CPM.cmake          # Package manager
│   └── fhtConfig.cmake.in # Install config
├── CMakeLists.txt         # Build configuration
├── LICENSE                # MIT license
└── README.md              # This file
```

## Performance

The library achieves excellent performance across a wide range of transform sizes:

- **x86**: Uses SSE or AVX intrinsics with pre-generated optimized code
- **ARM**: Uses NEON intrinsics with per-size optimized parameters including:
  - Radix-2 or radix-4 butterflies
  - Loop unrolling (1-4x)
  - Prefetching for large sizes
  - Iterative or recursive strategies

## License

MIT License. See [LICENSE](LICENSE) for details.

The x86 implementation is based on [FFHT](https://github.com/FALCONN-LIB/FFHT) by Andoni, Indyk, Laarhoven, Razenshteyn, and Schmidt.

## References

- [FFHT: Fast Fast Hadamard Transform](https://github.com/FALCONN-LIB/FFHT)
- [Fast Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform)
