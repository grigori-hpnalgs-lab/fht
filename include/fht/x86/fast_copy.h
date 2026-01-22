/*
 * FFHT - Fast memory copy using SIMD
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2015 Alexandr Andoni, Piotr Indyk, Thijs Laarhoven,
 * Ilya Razenshteyn, Ludwig Schmidt
 *
 * Adapted for fht library distribution.
 */

#ifndef FHT_FAST_COPY_H
#define FHT_FAST_COPY_H

#include <stdlib.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#ifndef FHT_FAST_COPY_MEMCPY_THRESHOLD
#define FHT_FAST_COPY_MEMCPY_THRESHOLD ((size_t)1ull << 20)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* These functions assume that the size of memory being copied is a power of 2. */

#if _FEATURE_AVX512F
static inline void *fht_fast_copy(void *out, void *in, size_t n) {
    if (n >= FHT_FAST_COPY_MEMCPY_THRESHOLD) {
        return memcpy(out, in, n);
    }
    n >>= 6;
    for (__m512 *ov = (__m512 *)out, *iv = (__m512 *)in; n--;) {
        _mm512_storeu_ps((float *)(ov++), _mm512_loadu_ps((float *)(iv++)));
    }
    return out;
}
#elif __AVX2__
static inline void *fht_fast_copy(void *out, void *in, size_t n) {
    if (n >= FHT_FAST_COPY_MEMCPY_THRESHOLD) {
        return memcpy(out, in, n);
    }
    n >>= 5;
    for (__m256 *ov = (__m256 *)out, *iv = (__m256 *)in; n--;) {
        _mm256_storeu_ps((float *)(ov++), _mm256_loadu_ps((float *)(iv++)));
    }
    return out;
}
#elif __SSE2__
static inline void *fht_fast_copy(void *out, void *in, size_t n) {
    if (n >= FHT_FAST_COPY_MEMCPY_THRESHOLD) {
        return memcpy(out, in, n);
    }
    n >>= 4;
    for (__m128 *ov = (__m128 *)out, *iv = (__m128 *)in; n--;) {
        _mm_storeu_ps((float *)(ov++), _mm_loadu_ps((float *)(iv++)));
    }
    return out;
}
#else
static inline void *fht_fast_copy(void *out, void *in, size_t n) {
    return memcpy(out, in, n);
}
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FHT_FAST_COPY_H */
