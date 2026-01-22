/*
 * Fast Hadamard Transform Library - Platform Configuration
 * SPDX-License-Identifier: MIT
 *
 * Automatic platform detection and configuration macros.
 */

#ifndef FHT_CONFIG_H
#define FHT_CONFIG_H

/* Platform detection */
#if defined(__aarch64__) || defined(_M_ARM64)
#define FHT_PLATFORM_ARM 1
#define FHT_PLATFORM_NAME "ARM64"
#elif defined(__x86_64__) || defined(_M_X64)
#define FHT_PLATFORM_X86 1
#define FHT_PLATFORM_X86_64 1
#define FHT_PLATFORM_NAME "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
#define FHT_PLATFORM_X86 1
#define FHT_PLATFORM_X86_32 1
#define FHT_PLATFORM_NAME "x86"
#else
#error "FHT: Unsupported platform. Requires x86, x86_64, or ARM64."
#endif

/* SIMD detection */
#if defined(FHT_PLATFORM_ARM)
#define FHT_HAS_NEON 1
#endif

#if defined(FHT_PLATFORM_X86)
#if defined(__AVX__)
#define FHT_HAS_AVX 1
#endif
#if defined(__SSE2__) || defined(_M_X64)
#define FHT_HAS_SSE2 1
#endif
#endif

/* Version info */
#define FHT_VERSION_MAJOR 1
#define FHT_VERSION_MINOR 0
#define FHT_VERSION_PATCH 0
#define FHT_VERSION_STRING "1.0.0"

#endif /* FHT_CONFIG_H */
