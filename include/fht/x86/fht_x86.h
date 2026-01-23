/*
 * FFHT - Fast Fast Hadamard Transform (x86 SSE/AVX)
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2015 Alexandr Andoni, Piotr Indyk, Thijs Laarhoven,
 * Ilya Razenshteyn, Ludwig Schmidt
 *
 * Adapted for fht library distribution.
 */

#ifndef FHT_X86_H
#define FHT_X86_H

#define FHT_HEADER_ONLY

/* Include the implementation first - defines fht_float, fht_double, fht_float_oop, fht_double_oop */
#include "fht_impl.h"

/* C++ overloads for convenience */
#ifdef __cplusplus
static inline int fht(float *buf, int log_n) {
    return fht_float(buf, log_n);
}

static inline int fht(double *buf, int log_n) {
    return fht_double(buf, log_n);
}

static inline int fht(float *buf, float *out, int log_n) {
    return fht_float_oop(buf, out, log_n);
}

static inline int fht(double *buf, double *out, int log_n) {
    return fht_double_oop(buf, out, log_n);
}
#endif

#endif /* FHT_X86_H */
