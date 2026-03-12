"""FHT throughput benchmarks.

Usage:
    python benchmarks.py                    # all cores
    OMP_NUM_THREADS=1 python benchmarks.py  # single-threaded
    OMP_NUM_THREADS=4 python benchmarks.py  # 4 threads
"""

import os
import time
import numpy as np
from fht import fht

BUFFER_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
WARMUP = 2
REPS = 5
LOG_N_RANGE = range(14, 25, 2)
DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


def bench(log_n: int, dtype: np.dtype) -> float:
    n = 1 << log_n
    elem_bytes = np.dtype(dtype).itemsize
    total_elems = BUFFER_BYTES // elem_bytes
    nrows = total_elems // n
    if nrows < 1:
        return float("nan")

    x = np.empty((nrows, n), dtype=dtype)
    # fill with real random data, cast
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.complexfloating):
        real_dtype = np.float32 if dtype == np.complex64 else np.float64
        x.real = rng.standard_normal((nrows, n)).astype(real_dtype)
        x.imag = rng.standard_normal((nrows, n)).astype(real_dtype)
    else:
        x[...] = rng.standard_normal((nrows, n)).astype(dtype)

    for _ in range(WARMUP):
        fht(x, axis=-1)

    elapsed = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fht(x, axis=-1)
        t1 = time.perf_counter()
        elapsed.append(t1 - t0)

    best = min(elapsed)
    actual_bytes = nrows * n * elem_bytes
    return (actual_bytes / 1e9) / best


def main():
    omp = os.environ.get("OMP_NUM_THREADS", "all")
    print(f"FHT throughput benchmark  |  buffer=1 GB  |  OMP_NUM_THREADS={omp}")
    print()

    # header
    dtype_names = [d.__name__ for d in DTYPES]
    hdr = f"{'log_n':>7} {'n':>12}"
    for name in dtype_names:
        hdr += f" {name + ' GB/s':>16}"
    print(hdr)
    print("-" * len(hdr))

    for log_n in LOG_N_RANGE:
        n = 1 << log_n
        row = f"{log_n:>7} {n:>12,}"
        for dtype in DTYPES:
            gb_s = bench(log_n, dtype)
            row += f" {gb_s:>16.2f}"
        print(row)


if __name__ == "__main__":
    main()
