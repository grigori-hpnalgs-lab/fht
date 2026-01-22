/*
 * FHT Reference Implementation
 *
 * A simple, correct (but slow) recursive implementation of the
 * Fast Hadamard Transform for use in testing.
 *
 * This implementation prioritizes clarity and correctness over performance.
 */

#ifndef FHT_REFERENCE_H
#define FHT_REFERENCE_H

#include <cstddef>
#include <vector>
#include <cstring>

namespace fht_reference {

/**
 * Simple recursive FHT implementation.
 *
 * The Hadamard transform satisfies:
 *   H_n = [H_{n-1}  H_{n-1}]
 *         [H_{n-1} -H_{n-1}]
 *
 * For input x, output y = H_n * x where:
 *   y[i] = sum_{j} H_n[i][j] * x[j]
 *
 * The fast algorithm processes in-place with butterfly operations.
 */
template<typename T>
void fht_recursive(T* buf, size_t n) {
    if (n <= 1) return;

    size_t half = n / 2;

    // Butterfly: for each pair (buf[i], buf[i + half])
    // compute (buf[i] + buf[i + half], buf[i] - buf[i + half])
    for (size_t i = 0; i < half; ++i) {
        T a = buf[i];
        T b = buf[i + half];
        buf[i] = a + b;
        buf[i + half] = a - b;
    }

    // Recurse on both halves
    fht_recursive(buf, half);
    fht_recursive(buf + half, half);
}

/**
 * Reference FHT for float.
 * Returns 0 on success, 1 on error.
 */
inline int fht_float(float* buf, int log_n) {
    if (log_n < 0 || log_n > 30) return 1;
    if (log_n == 0) return 0;

    size_t n = static_cast<size_t>(1) << log_n;
    fht_recursive(buf, n);
    return 0;
}

/**
 * Reference FHT for double.
 * Returns 0 on success, 1 on error.
 */
inline int fht_double(double* buf, int log_n) {
    if (log_n < 0 || log_n > 30) return 1;
    if (log_n == 0) return 0;

    size_t n = static_cast<size_t>(1) << log_n;
    fht_recursive(buf, n);
    return 0;
}

/**
 * Out-of-place reference FHT for float.
 */
inline int fht_float_oop(const float* in, float* out, int log_n) {
    if (log_n < 0 || log_n > 30) return 1;
    if (log_n == 0) {
        out[0] = in[0];
        return 0;
    }

    size_t n = static_cast<size_t>(1) << log_n;
    std::memcpy(out, in, n * sizeof(float));
    fht_recursive(out, n);
    return 0;
}

/**
 * Out-of-place reference FHT for double.
 */
inline int fht_double_oop(const double* in, double* out, int log_n) {
    if (log_n < 0 || log_n > 30) return 1;
    if (log_n == 0) {
        out[0] = in[0];
        return 0;
    }

    size_t n = static_cast<size_t>(1) << log_n;
    std::memcpy(out, in, n * sizeof(double));
    fht_recursive(out, n);
    return 0;
}

} // namespace fht_reference

#endif // FHT_REFERENCE_H
