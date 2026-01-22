#!/usr/bin/env python3
"""
Grid Search Optimizer for V7 FHT NEON Kernels.

Explores the extended parameter space:
- strategy: iterative, recursive
- threshold: 2..log_n (for recursive)
- radix: 2, 4
- unroll_factor: 1, 2, 4
- prefetch_distance: 0, 256, 512
- max_registers: 8, 16

Uses early stopping and smart pruning to reduce search space.

Usage:
    python optimize_v7_grid.py                    # Full optimization (log_n 4-26)
    python optimize_v7_grid.py --sizes 10 14 18  # Specific sizes
    python optimize_v7_grid.py --quick           # Quick mode (fewer configs)
    python optimize_v7_grid.py --max-logn 20     # Up to log_n=20
    python optimize_v7_grid.py --output path.h   # Specify output header path
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import V7's code generator
from gen_neon_v7 import (
    KernelParams,
    generate_kernel_v7,
    generate_kernel_double_v7,
    generate_full_header,
    VERSION,
)

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR.parent / 'data'
INCLUDE_DIR = SCRIPT_DIR.parent / 'include' / 'fht' / 'neon'
RESULTS_PATH = DATA_DIR / 'optimization_results_v7.json'
RESULTS_PATH_DOUBLE = DATA_DIR / 'optimization_results_v7_double.json'
DEFAULT_HEADER_PATH = INCLUDE_DIR / 'fht_neon.h'


# =============================================================================
# Parameter Grid Generation
# =============================================================================

def generate_param_grid(log_n: int, quick: bool = False) -> List[KernelParams]:
    """
    Generate all parameter combinations for a given size.

    Returns a list of KernelParams to evaluate.
    """
    configs = []

    # Define parameter ranges
    if quick:
        thresholds = [t for t in [4, 6, 8, 10, 12, 14] if t <= log_n]
        radixes = [2, 4]
        unrolls = [1, 2]
        prefetches = [0, 256] if log_n >= 16 else [0]
        max_regs = [16]
    else:
        # Full grid
        thresholds = list(range(2, min(log_n + 1, 17)))  # 2..16 or log_n
        radixes = [2, 4]
        unrolls = [1, 2, 4]
        prefetches = [0, 256, 512] if log_n >= 16 else [0]
        max_regs = [8, 16]

    # Iterative configurations
    for radix in radixes:
        for unroll in unrolls:
            for prefetch in prefetches:
                for max_reg in max_regs:
                    configs.append(KernelParams(
                        strategy='iterative',
                        radix=radix,
                        unroll_factor=unroll,
                        prefetch_distance=prefetch,
                        max_registers=max_reg,
                    ))

    # Recursive configurations
    for threshold in thresholds:
        for radix in radixes:
            for unroll in unrolls:
                for prefetch in prefetches:
                    for max_reg in max_regs:
                        configs.append(KernelParams(
                            strategy='recursive',
                            threshold=threshold,
                            radix=radix,
                            unroll_factor=unroll,
                            prefetch_distance=prefetch,
                            max_registers=max_reg,
                        ))

    return configs


def generate_param_grid_double(log_n: int, quick: bool = False) -> List[KernelParams]:
    """
    Generate parameter combinations for double precision.

    Double precision forces radix=2 (can't do radix-4 with 2 elements per register).
    """
    configs = []

    # Define parameter ranges
    if quick:
        thresholds = [t for t in [4, 6, 8, 10, 12, 14] if t <= log_n]
        unrolls = [1, 2]
        prefetches = [0, 256] if log_n >= 16 else [0]
        max_regs = [16]
    else:
        thresholds = list(range(2, min(log_n + 1, 17)))
        unrolls = [1, 2, 4]
        prefetches = [0, 256, 512] if log_n >= 16 else [0]
        max_regs = [8, 16]

    # Iterative configurations (radix always 2 for double)
    for unroll in unrolls:
        for prefetch in prefetches:
            for max_reg in max_regs:
                configs.append(KernelParams(
                    strategy='iterative',
                    radix=2,  # Forced for double
                    unroll_factor=unroll,
                    prefetch_distance=prefetch,
                    max_registers=max_reg,
                ))

    # Recursive configurations
    for threshold in thresholds:
        for unroll in unrolls:
            for prefetch in prefetches:
                for max_reg in max_regs:
                    configs.append(KernelParams(
                        strategy='recursive',
                        threshold=threshold,
                        radix=2,  # Forced for double
                        unroll_factor=unroll,
                        prefetch_distance=prefetch,
                        max_registers=max_reg,
                    ))

    return configs


def prune_grid(configs: List[KernelParams], log_n: int) -> List[KernelParams]:
    """
    Remove obviously suboptimal configurations.

    Pruning rules:
    - unroll > 4 for small sizes (log_n <= 10)
    - prefetch for very small sizes (log_n < 16)
    - max_registers=8 rarely helps (keep only for medium sizes)
    """
    pruned = []

    for cfg in configs:
        # Skip high unroll for very small sizes
        if log_n <= 6 and cfg.unroll_factor > 2:
            continue

        # Skip prefetch for small sizes that fit in L1
        if log_n < 14 and cfg.prefetch_distance > 0:
            continue

        # max_registers=8 mainly helps for specific size ranges
        if cfg.max_registers == 8 and (log_n < 8 or log_n > 18):
            continue

        # Skip recursive with threshold >= log_n (equivalent to iterative)
        if cfg.strategy == 'recursive' and cfg.threshold >= log_n:
            continue

        pruned.append(cfg)

    return pruned


# =============================================================================
# Benchmark Code Generation
# =============================================================================

def generate_benchmark_code(kernel_code: str, log_n: int, func_name: str,
                            precision: str = 'float') -> str:
    """Generate standalone benchmark code for a kernel."""
    # Determine batch size for timing accuracy
    if log_n <= 10:
        batch_size = 256
    elif log_n <= 12:
        batch_size = 64
    elif log_n <= 14:
        batch_size = 16
    elif log_n <= 16:
        batch_size = 4
    elif log_n <= 20:
        batch_size = 2
    else:
        batch_size = 1

    # Double precision uses more memory, reduce batch size
    if precision == 'double':
        batch_size = max(1, batch_size // 2)

    type_name = precision
    dist_type = 'float' if precision == 'float' else 'double'

    return f'''// Auto-generated benchmark for {func_name}
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <arm_neon.h>

// Kernel code
{kernel_code}

int main() {{
    const int log_n = {log_n};
    const size_t n = 1UL << log_n;
    const int batch_size = {batch_size};
    const int warmup = 5;
    const int iterations = 30;

    std::vector<{type_name}> buf(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<{dist_type}> dist(-1.0, 1.0);

    for (size_t i = 0; i < n; ++i) {{
        buf[i] = dist(rng);
    }}

    // Warmup
    for (int i = 0; i < warmup; ++i) {{
        {func_name}(buf.data());
    }}

    // Timed runs
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {{
        // Reinitialize periodically to prevent numerical issues
        if (i % 10 == 0) {{
            for (size_t j = 0; j < n; ++j) {{
                buf[j] = dist(rng);
            }}
        }}

        auto start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < batch_size; ++b) {{
            {func_name}(buf.data());
        }}
        auto end = std::chrono::high_resolution_clock::now();

        double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double us_per_call = total_ns / (1000.0 * batch_size);
        times.push_back(us_per_call);
    }}

    // Return median time
    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];

    std::cout << median << std::endl;
    return 0;
}}
'''


# =============================================================================
# Measurement
# =============================================================================

def measure_time(kernel_code: str, log_n: int, func_name: str,
                 verbose: bool = False, precision: str = 'float') -> Optional[float]:
    """Compile and measure execution time for a kernel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / 'bench.cpp'
        bin_path = Path(tmpdir) / 'bench'

        bench_code = generate_benchmark_code(kernel_code, log_n, func_name, precision)
        src_path.write_text(bench_code)

        # Compile - use portable ARM flags
        compile_cmd = [
            'clang++', '-O3', '-std=c++17', '-ffast-math',
            '-o', str(bin_path), str(src_path)
        ]

        # Add ARM NEON flag
        if platform.machine() in ('arm64', 'aarch64'):
            compile_cmd.insert(4, '-march=armv8-a+simd')

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                # Try g++ as fallback
                compile_cmd[0] = 'g++'
                result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            if result.returncode != 0:
                if verbose:
                    print(f"    Compile error: {result.stderr[:300]}")
                return None
        except subprocess.TimeoutExpired:
            if verbose:
                print("    Compile timeout")
            return None
        except FileNotFoundError as e:
            if verbose:
                print(f"    Compiler not found: {e}")
            return None

        # Run benchmark
        try:
            result = subprocess.run(
                [str(bin_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                if verbose:
                    print(f"    Runtime error: {result.stderr[:200]}")
                return None

            time_us = float(result.stdout.strip())
            return time_us

        except (subprocess.TimeoutExpired, ValueError) as e:
            if verbose:
                print(f"    Runtime error: {e}")
            return None


def try_config(log_n: int, params: KernelParams, verbose: bool = False,
               precision: str = 'float') -> Tuple[Optional[float], str, KernelParams]:
    """Try a single configuration and return timing."""
    func_name = f'fht_neon_v7_{precision}_{log_n}'

    try:
        if precision == 'float':
            kernel_code = generate_kernel_v7(log_n, params, func_name)
        else:
            kernel_code = generate_kernel_double_v7(log_n, params, func_name)
    except Exception as e:
        if verbose:
            print(f"    {params.short_desc()}: SKIP (codegen: {e})")
        return None, '', params

    time_us = measure_time(kernel_code, log_n, func_name, verbose, precision)

    if time_us is not None:
        if verbose:
            print(f"    {params.short_desc()}: {time_us:.2f} us")
    elif verbose:
        print(f"    {params.short_desc()}: FAIL (benchmark)")

    return time_us, kernel_code, params


# =============================================================================
# Single Size Optimization
# =============================================================================

def optimize_single_size(log_n: int, quick: bool = False,
                         verbose: bool = False, early_stop: bool = True,
                         precision: str = 'float') -> Dict[str, Any]:
    """
    Optimize for a single log_n using grid search.

    Uses early stopping: if we find a config within 2% of the current best
    after testing 20 configs, we can reduce remaining tests.
    """
    print(f"\nlog_n = {log_n} (n = {1 << log_n:,}) [{precision}]")

    # Generate and prune parameter grid
    if precision == 'double':
        all_configs = generate_param_grid_double(log_n, quick)
    else:
        all_configs = generate_param_grid(log_n, quick)
    configs = prune_grid(all_configs, log_n)

    print(f"  Testing {len(configs)} configurations (pruned from {len(all_configs)})")

    results = []  # List of (time, code, params)
    best_time = float('inf')
    configs_since_improvement = 0
    max_configs_without_improvement = 30 if early_stop else len(configs)

    for i, cfg in enumerate(configs):
        time_us, code, params = try_config(log_n, cfg, verbose, precision)

        if time_us is not None:
            results.append((time_us, code, params))

            if time_us < best_time:
                improvement = (best_time - time_us) / best_time * 100 if best_time < float('inf') else 100
                best_time = time_us
                configs_since_improvement = 0
                if verbose or improvement > 5:
                    print(f"    NEW BEST: {time_us:.2f} us ({params.short_desc()})")
            else:
                configs_since_improvement += 1

        # Early stopping check
        if early_stop and configs_since_improvement >= max_configs_without_improvement:
            print(f"  Early stop: no improvement in {max_configs_without_improvement} configs")
            break

        # Progress update
        if (i + 1) % 20 == 0 and not verbose:
            print(f"  ... tested {i+1}/{len(configs)} configs, best={best_time:.2f} us")

    if not results:
        print(f"  ERROR: No working configuration found!")
        return None

    # Sort by time and get best
    results.sort(key=lambda x: x[0])
    best_time, best_code, best_params = results[0]

    print(f"  BEST: {best_time:.2f} us - {best_params.short_desc()}")

    return {
        'log_n': log_n,
        'best_time': best_time,
        'best_params': best_params.to_dict(),
        'best_desc': best_params.short_desc(),
        'code_size': len(best_code),
        'num_configs_tested': len(results),
        'top_5': [(t, p.short_desc()) for t, _, p in results[:5]],
    }


# =============================================================================
# Full Optimization
# =============================================================================

def optimize_all_sizes(sizes: List[int], quick: bool = False,
                       verbose: bool = False, early_stop: bool = True,
                       precision: str = 'float') -> Dict[int, Dict[str, Any]]:
    """Optimize for all specified sizes."""
    print("=" * 70)
    print(f"V7 Grid Search Optimization [{precision}]")
    print("=" * 70)
    print(f"Sizes to optimize: {sizes}")
    print(f"Precision: {precision}")
    print(f"Mode: {'quick' if quick else 'full'}")
    print(f"Early stopping: {'enabled' if early_stop else 'disabled'}")

    results = {}
    for log_n in sizes:
        result = optimize_single_size(log_n, quick, verbose, early_stop, precision)
        if result:
            results[log_n] = result

    return results


# =============================================================================
# Results I/O
# =============================================================================

def save_results(results: Dict[int, Dict[str, Any]], path: Path = RESULTS_PATH,
                 precision: str = 'float'):
    """Save optimization results to JSON."""
    serializable = {}
    for k, v in results.items():
        serializable[str(k)] = {
            'log_n': v['log_n'],
            'best_time': v['best_time'],
            'best_desc': v['best_desc'],
            'best_params': v['best_params'],
            'code_size': v['code_size'],
            'num_configs_tested': v['num_configs_tested'],
        }

    serializable['_metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'generator': 'optimize_v7_grid.py',
        'version': VERSION,
        'precision': precision,
    }

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {path}")


def load_results(path: Path = RESULTS_PATH) -> Dict[int, Dict[str, Any]]:
    """Load previous optimization results."""
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    results = {}
    for k, v in data.items():
        if k.startswith('_'):
            continue
        results[int(k)] = v

    return results


def print_summary(results: Dict[int, Dict[str, Any]]):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("Optimization Summary")
    print("=" * 70)
    print(f"{'log_n':>6} {'N':>14} {'Time (us)':>12} {'Strategy':>35}")
    print("-" * 70)

    for log_n in sorted(results.keys()):
        data = results[log_n]
        n = 1 << log_n
        print(f"{log_n:>6} {n:>14,} {data['best_time']:>12.2f} {data['best_desc']:>35}")


# =============================================================================
# Header Generation
# =============================================================================

def generate_header(results: Dict[int, Dict[str, Any]], max_log_n: int = 30,
                    output_path: Optional[Path] = None,
                    double_results: Optional[Dict[int, Dict[str, Any]]] = None):
    """Generate optimized fht_neon header file."""
    if output_path is None:
        output_path = DEFAULT_HEADER_PATH
    print("\n" + "=" * 70)
    print(f"Generating {output_path.name}")
    print("=" * 70)

    # Generate float kernels
    print("\nFloat kernels:")
    float_kernels = {}

    for log_n in range(1, max_log_n + 1):
        if log_n in results:
            params_dict = results[log_n]['best_params']
            params = KernelParams(**params_dict)
        else:
            # Use sensible defaults
            if log_n <= 14:
                params = KernelParams(strategy='iterative', radix=4, unroll_factor=2)
            else:
                params = KernelParams(strategy='recursive', threshold=14, radix=4, unroll_factor=2)

        try:
            kernel = generate_kernel_v7(log_n, params)
            float_kernels[log_n] = kernel
            strategy_str = params.short_desc()
            print(f"  log_n={log_n:2d}: {strategy_str}")
        except Exception as e:
            # Fallback to basic params
            try:
                kernel = generate_kernel_v7(log_n, KernelParams())
                float_kernels[log_n] = kernel
                print(f"  log_n={log_n:2d}: default (fallback)")
            except Exception as e2:
                print(f"  log_n={log_n:2d}: FAILED - {e2}")

    # Generate double kernels if results provided
    double_kernels = None
    if double_results is not None:
        print("\nDouble kernels:")
        double_kernels = {}

        for log_n in range(1, max_log_n + 1):
            if log_n in double_results:
                params_dict = double_results[log_n]['best_params']
                params = KernelParams(**params_dict)
            else:
                # Use sensible defaults for double (radix=2 forced)
                if log_n <= 14:
                    params = KernelParams(strategy='iterative', radix=2, unroll_factor=2)
                else:
                    params = KernelParams(strategy='recursive', threshold=14, radix=2, unroll_factor=2)

            try:
                kernel = generate_kernel_double_v7(log_n, params)
                double_kernels[log_n] = kernel
                strategy_str = params.short_desc()
                print(f"  log_n={log_n:2d}: {strategy_str}")
            except Exception as e:
                # Fallback to basic params
                try:
                    kernel = generate_kernel_double_v7(log_n, KernelParams())
                    double_kernels[log_n] = kernel
                    print(f"  log_n={log_n:2d}: default (fallback)")
                except Exception as e2:
                    print(f"  log_n={log_n:2d}: FAILED - {e2}")

    header = generate_full_header(float_kernels, max_log_n, double_kernels_by_logn=double_kernels)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(header)

    print(f"\nGenerated {output_path} ({len(header)} bytes)")


# =============================================================================
# Comparison Benchmark
# =============================================================================

def run_comparison(results: Dict[int, Dict[str, Any]]):
    """Run comparison benchmark against V6 and FXT."""
    print("\n" + "=" * 70)
    print("Running comparison benchmark")
    print("=" * 70)

    sizes = sorted(results.keys())
    if not sizes:
        print("No results to compare")
        return

    sizes_str = ', '.join(str(s) for s in sizes if s <= 26)

    comparison_code = f'''// Comparison: V7 vs V6 vs V3 vs FXT
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <arm_neon.h>

#include "fxttypes.h"
#include "walsh/walshwak.h"
#include "fht_neon_v3.h"
#include "fht_neon_v6.h"
#include "fht_neon_v7.h"

template<typename Func>
double benchmark(Func fn, float* buf, size_t n, int iterations) {{
    std::vector<double> times;
    times.reserve(iterations);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < 5; ++i) fn(buf, n);

    for (int i = 0; i < iterations; ++i) {{
        if (i % 10 == 0) {{
            for (size_t j = 0; j < n; ++j) buf[j] = dist(rng);
        }}
        auto start = std::chrono::high_resolution_clock::now();
        fn(buf, n);
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
    }}
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}}

int main() {{
    std::vector<int> sizes = {{{sizes_str}}};

    std::cout << "\\n";
    std::cout << "==========================================================================\\n";
    std::cout << "Comparison: V7 (Grid Search) vs V6 vs V3 vs FXT\\n";
    std::cout << "==========================================================================\\n";
    std::cout << std::setw(6) << "log_n"
              << std::setw(12) << "N"
              << std::setw(10) << "V3"
              << std::setw(10) << "V6"
              << std::setw(10) << "V7"
              << std::setw(10) << "FXT"
              << std::setw(10) << "Best"
              << std::setw(12) << "V7 vs Best"
              << "\\n";
    std::cout << "--------------------------------------------------------------------------\\n";

    for (int log_n : sizes) {{
        const size_t n = 1UL << log_n;
        std::vector<float> buf(n);

        int iterations = 50;
        if (log_n >= 22) iterations = 10;
        else if (log_n >= 20) iterations = 20;

        double t_v3 = benchmark([log_n](float* b, size_t) {{ fht_neon_v3_float(b, log_n); }}, buf.data(), n, iterations);
        double t_v6 = benchmark([log_n](float* b, size_t) {{ fht_neon_v6_float(b, log_n); }}, buf.data(), n, iterations);
        double t_v7 = benchmark([log_n](float* b, size_t) {{ fht_neon_v7_float(b, log_n); }}, buf.data(), n, iterations);
        double t_fxt = benchmark([log_n](float* b, size_t) {{ walsh_wak(b, log_n); }}, buf.data(), n, iterations);

        double best_other = std::min({{t_v3, t_v6, t_fxt}});
        std::string best_name = "V3";
        if (best_other == t_v6) best_name = "V6";
        else if (best_other == t_fxt) best_name = "FXT";

        double v7_vs_best = (best_other / t_v7 - 1.0) * 100.0;
        std::string v7_label = v7_vs_best >= 0 ? "+" : "";

        std::cout << std::setw(6) << log_n
                  << std::setw(12) << n
                  << std::setw(10) << std::fixed << std::setprecision(2) << t_v3
                  << std::setw(10) << t_v6
                  << std::setw(10) << t_v7
                  << std::setw(10) << t_fxt
                  << std::setw(10) << best_name
                  << std::setw(10) << v7_label << std::setprecision(1) << v7_vs_best << "%"
                  << "\\n";
    }}
    std::cout << "==========================================================================\\n";
    std::cout << "(V7 vs Best: positive = V7 is faster)\\n";
    return 0;
}}
'''

    bench_path = SCRIPT_DIR / 'comparison_v7.cpp'
    bench_path.write_text(comparison_code)

    compile_cmd = [
        'clang++', '-O3', '-std=c++17', '-ffast-math',
        '-march=armv8-a+simd',
        '-I.', '-I./include',
        '-o', 'comparison_v7', 'comparison_v7.cpp'
    ]

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True, cwd=SCRIPT_DIR)
        print("Compiled comparison benchmark")
    except subprocess.CalledProcessError as e:
        print(f"Compile error: {e.stderr}")
        return

    try:
        result = subprocess.run(['./comparison_v7'], capture_output=True, text=True, check=True, cwd=SCRIPT_DIR, timeout=300)
        print(result.stdout)
    except Exception as e:
        print(f"Runtime error: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Grid search optimizer for V7 FHT NEON kernels'
    )
    parser.add_argument(
        '--sizes', type=int, nargs='+',
        help='Specific log_n sizes to optimize (default: 4-26)'
    )
    parser.add_argument(
        '--max-logn', type=int, default=26,
        help='Maximum log_n to optimize (default: 26)'
    )
    parser.add_argument(
        '--min-logn', type=int, default=4,
        help='Minimum log_n to optimize (default: 4)'
    )
    parser.add_argument(
        '--quick', '-q', action='store_true',
        help='Quick mode: test fewer configurations'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show all configurations tested'
    )
    parser.add_argument(
        '--no-early-stop', action='store_true',
        help='Disable early stopping'
    )
    parser.add_argument(
        '--no-header', action='store_true',
        help='Skip header generation'
    )
    parser.add_argument(
        '--no-compare', action='store_true',
        help='Skip comparison benchmark'
    )
    parser.add_argument(
        '--continue', dest='continue_', action='store_true',
        help='Continue from previous results'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output header file path (default: include/fht/neon/fht_neon.h)'
    )
    parser.add_argument(
        '--precision', '-p', type=str, default='float',
        choices=['float', 'double', 'both'],
        help='Precision to optimize: float, double, or both (default: float)'
    )

    args = parser.parse_args()

    # Determine sizes to optimize
    if args.sizes:
        sizes = args.sizes
    else:
        sizes = list(range(args.min_logn, args.max_logn + 1))

    float_results = {}
    double_results = {}

    # Optimize float precision
    if args.precision in ('float', 'both'):
        float_sizes = sizes.copy()

        # Load previous results if continuing
        if args.continue_:
            float_results = load_results(RESULTS_PATH)
            float_sizes = [s for s in float_sizes if s not in float_results]
            print(f"Continuing from previous float results, {len(float_sizes)} sizes remaining")

        # Run optimization
        if float_sizes:
            new_results = optimize_all_sizes(
                float_sizes,
                quick=args.quick,
                verbose=args.verbose,
                early_stop=not args.no_early_stop,
                precision='float'
            )
            float_results.update(new_results)

        if float_results:
            save_results(float_results, RESULTS_PATH, 'float')
            print_summary(float_results)

    # Optimize double precision
    if args.precision in ('double', 'both'):
        double_sizes = sizes.copy()

        # Load previous results if continuing
        if args.continue_:
            double_results = load_results(RESULTS_PATH_DOUBLE)
            double_sizes = [s for s in double_sizes if s not in double_results]
            print(f"Continuing from previous double results, {len(double_sizes)} sizes remaining")

        # Run optimization
        if double_sizes:
            new_results = optimize_all_sizes(
                double_sizes,
                quick=args.quick,
                verbose=args.verbose,
                early_stop=not args.no_early_stop,
                precision='double'
            )
            double_results.update(new_results)

        if double_results:
            save_results(double_results, RESULTS_PATH_DOUBLE, 'double')
            print_summary(double_results)

    # Generate header with both float and double (if optimized)
    if not args.no_header:
        # Load float results if not optimized but needed for header
        if not float_results:
            float_results = load_results(RESULTS_PATH)

        # Load double results if not optimized
        if not double_results and args.precision == 'both':
            double_results = load_results(RESULTS_PATH_DOUBLE)

        # Determine if we should include double in the header
        include_double = double_results if (args.precision in ('double', 'both') or double_results) else None

        output_path = Path(args.output) if args.output else None
        generate_header(float_results, max_log_n=30, output_path=output_path,
                       double_results=include_double)

    if not args.no_compare and args.precision == 'float':
        run_comparison(float_results)

    print("\n" + "=" * 70)
    print("V7 Optimization complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
