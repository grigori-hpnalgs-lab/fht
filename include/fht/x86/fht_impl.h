/*
 * FFHT - Fast Fast Hadamard Transform (x86 SSE/AVX)
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2015 Alexandr Andoni, Piotr Indyk, Thijs Laarhoven,
 * Ilya Razenshteyn, Ludwig Schmidt
 *
 * Adapted for fht library distribution.
 */

#ifndef FHT_X86_IMPL_H
#define FHT_X86_IMPL_H

#include "fast_copy.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __AVX__
#include "fht_avx.c"
#define FHT_VECTOR_WIDTH (32u)
#else
#include "fht_sse.c"
#define FHT_VECTOR_WIDTH (16u)
#endif

static inline int fht_float_oop(float *in, float *out, int log_n) {
    fht_fast_copy(out, in, sizeof(float) << log_n);
    return fht_float(out, log_n);
}

static inline int fht_double_oop(double *in, double *out, int log_n) {
    fht_fast_copy(out, in, sizeof(double) << log_n);
    return fht_double(out, log_n);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FHT_X86_IMPL_H */
