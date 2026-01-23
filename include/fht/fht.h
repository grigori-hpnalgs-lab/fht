/*
 * Fast Hadamard Transform Library
 * SPDX-License-Identifier: MIT
 *
 * A portable, high-performance Fast Hadamard Transform (FHT) library
 * with optimized implementations for x86 (SSE/AVX) and ARM (NEON).
 *
 * x86 code: Copyright (c) 2015 Alexandr Andoni, Piotr Indyk, Thijs Laarhoven,
 *           Ilya Razenshteyn, Ludwig Schmidt
 * ARM code: Copyright (c) 2026 FHT Library Contributors
 *
 * Usage:
 *   #include <fht/fht.h>
 *
 *   float buf[1024];  // Must be power of 2
 *   // ... fill buf ...
 *   fht_float(buf, 10);  // log2(1024) = 10
 *
 * API:
 *   int fht_float(float *buf, int log_n);         // In-place single precision
 *   int fht_double(double *buf, int log_n);       // In-place double precision
 *   int fht_float_oop(float *in, float *out, int log_n);   // Out-of-place
 *   int fht_double_oop(double *in, double *out, int log_n);
 *
 * Returns 0 on success, non-zero on error (invalid log_n).
 */

#ifndef FHT_H
#define FHT_H

#include "fht_config.h"

/*
 * =============================================================================
 * Platform-specific Implementation
 * =============================================================================
 *
 * The appropriate implementation is included based on detected platform.
 * All platforms provide the same API: fht_float, fht_double, fht_float_oop,
 * fht_double_oop functions.
 */

#if defined(FHT_PLATFORM_ARM)

/* ARM NEON Implementation */
#if defined(FHT_USE_OPTIMIZED_HEADER) && defined(FHT_OPTIMIZED_HEADER_PATH)
#include FHT_OPTIMIZED_HEADER_PATH
#else
#include "neon/fht_neon.h"
#endif

#include <string.h>

static inline int fht_float(float *buf, int log_n) {
    return fht_neon_v7_float(buf, log_n);
}

static inline int fht_double(double *buf, int log_n) {
    return fht_neon_v7_double(buf, log_n);
}

static inline int fht_float_oop(float *in, float *out, int log_n) {
    if (log_n < 0 || log_n > 30) return 1;
    if (log_n == 0) {
        out[0] = in[0];
        return 0;
    }
    memcpy(out, in, sizeof(float) * (1UL << log_n));
    return fht_neon_v7_float(out, log_n);
}

static inline int fht_double_oop(double *in, double *out, int log_n) {
    if (log_n < 0 || log_n > 30) return 1;
    if (log_n == 0) {
        out[0] = in[0];
        return 0;
    }
    memcpy(out, in, sizeof(double) * (1UL << log_n));
    return fht_neon_v7_double(out, log_n);
}

/* C++ overloads for ARM */
#ifdef __cplusplus
static inline int fht(float *buf, int log_n) {
    return fht_float(buf, log_n);
}

static inline int fht(double *buf, int log_n) {
    return fht_double(buf, log_n);
}

static inline int fht(float *in, float *out, int log_n) {
    return fht_float_oop(in, out, log_n);
}

static inline int fht(double *in, double *out, int log_n) {
    return fht_double_oop(in, out, log_n);
}
#endif /* __cplusplus */

#elif defined(FHT_PLATFORM_X86)

/* x86 SSE/AVX Implementation (from FFHT) - includes C++ overloads */
#include "x86/fht_x86.h"

#else

#error "FHT: No implementation available for this platform"

#endif /* Platform selection */

#endif /* FHT_H */
