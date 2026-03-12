#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <fht/fht.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

// num_threads convention:
//   -1 = all cores (OMP default)
//    0 or 1 = single-threaded (no OMP)
//    N > 1 = use exactly N threads

// Resolve num_threads: -1 → OMP default, 0 → 1, else keep
static int resolve_threads(int nt) {
#ifdef _OPENMP
    if (nt == -1) return omp_get_max_threads();
    if (nt == 0) return 1;
    return nt;
#else
    (void)nt;
    return 1;
#endif
}

static int ilog2(size_t n) {
    if (n == 0 || (n & (n - 1)) != 0) return -1;
    int log_n = 0;
    while (n > 1) { n >>= 1; log_n++; }
    return log_n;
}

// ── 1D in-place (real) ──

static void fht_1d_f32(nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> arr) {
    int log_n = ilog2(arr.shape(0));
    if (log_n < 0) throw nb::value_error("Array length must be a power of 2");
    int rc = fht_float(arr.data(), log_n);
    if (rc) throw nb::value_error("fht_float failed (invalid log_n)");
}

static void fht_1d_f64(nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> arr) {
    int log_n = ilog2(arr.shape(0));
    if (log_n < 0) throw nb::value_error("Array length must be a power of 2");
    int rc = fht_double(arr.data(), log_n);
    if (rc) throw nb::value_error("fht_double failed (invalid log_n)");
}

// ── 2D in-place (row-major: parallelize over rows) ──

static void fht_2d_f32_rows(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr, int num_threads) {
    size_t nrows = arr.shape(0);
    size_t ncols = arr.shape(1);
    int log_n = ilog2(ncols);
    if (log_n < 0) throw nb::value_error("Column count must be a power of 2");
    float *data = arr.data();
    int nt = resolve_threads(num_threads);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    #endif
    for (size_t i = 0; i < nrows; i++) {
        fht_float(data + i * ncols, log_n);
    }
}

static void fht_2d_f64_rows(nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr, int num_threads) {
    size_t nrows = arr.shape(0);
    size_t ncols = arr.shape(1);
    int log_n = ilog2(ncols);
    if (log_n < 0) throw nb::value_error("Column count must be a power of 2");
    double *data = arr.data();
    int nt = resolve_threads(num_threads);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    #endif
    for (size_t i = 0; i < nrows; i++) {
        fht_double(data + i * ncols, log_n);
    }
}

// ── 2D in-place (column-major / F-contiguous: parallelize over columns) ──

static void fht_2d_f32_cols(nb::ndarray<float, nb::ndim<2>, nb::f_contig, nb::device::cpu> arr, int num_threads) {
    size_t nrows = arr.shape(0);
    size_t ncols = arr.shape(1);
    int log_n = ilog2(nrows);
    if (log_n < 0) throw nb::value_error("Row count must be a power of 2");
    float *data = arr.data();
    int nt = resolve_threads(num_threads);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    #endif
    for (size_t j = 0; j < ncols; j++) {
        fht_float(data + j * nrows, log_n);
    }
}

static void fht_2d_f64_cols(nb::ndarray<double, nb::ndim<2>, nb::f_contig, nb::device::cpu> arr, int num_threads) {
    size_t nrows = arr.shape(0);
    size_t ncols = arr.shape(1);
    int log_n = ilog2(nrows);
    if (log_n < 0) throw nb::value_error("Row count must be a power of 2");
    double *data = arr.data();
    int nt = resolve_threads(num_threads);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    #endif
    for (size_t j = 0; j < ncols; j++) {
        fht_double(data + j * nrows, log_n);
    }
}

// ── Complex: deinterleave, transform, reinterleave — all in C++ with OpenMP ──

template <typename T, int (*fht_fn)(T*, int)>
static void fht_complex_1d_impl(
    nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> arr,
    nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> scratch)
{
    size_t n2 = arr.shape(0);
    size_t n = n2 / 2;
    int log_n = ilog2(n);
    if (log_n < 0) throw nb::value_error("Complex array length must be a power of 2");

    T *interleaved = arr.data();
    T *buf = scratch.data();
    T *re = buf;
    T *im = buf + n;

    for (size_t i = 0; i < n; i++) {
        re[i] = interleaved[2 * i];
        im[i] = interleaved[2 * i + 1];
    }

    fht_fn(re, log_n);
    fht_fn(im, log_n);

    for (size_t i = 0; i < n; i++) {
        interleaved[2 * i]     = re[i];
        interleaved[2 * i + 1] = im[i];
    }
}

template <typename T, int (*fht_fn)(T*, int)>
static void fht_complex_2d_rows_impl(
    nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr,
    nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> scratch_re,
    nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> scratch_im,
    int num_threads)
{
    size_t nrows = arr.shape(0);
    size_t ncols2 = arr.shape(1);
    size_t ncols = ncols2 / 2;
    int log_n = ilog2(ncols);
    if (log_n < 0) throw nb::value_error("Column count (complex) must be a power of 2");

    T *data = arr.data();
    T *re_buf = scratch_re.data();
    T *im_buf = scratch_im.data();
    int nt = resolve_threads(num_threads);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    #endif
    for (size_t i = 0; i < nrows; i++) {
        T *row = data + i * ncols2;
        T *re = re_buf + i * ncols;
        T *im = im_buf + i * ncols;
        for (size_t j = 0; j < ncols; j++) {
            re[j] = row[2 * j];
            im[j] = row[2 * j + 1];
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    #endif
    for (size_t i = 0; i < nrows; i++) {
        fht_fn(re_buf + i * ncols, log_n);
        fht_fn(im_buf + i * ncols, log_n);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
    #endif
    for (size_t i = 0; i < nrows; i++) {
        T *row = data + i * ncols2;
        T *re = re_buf + i * ncols;
        T *im = im_buf + i * ncols;
        for (size_t j = 0; j < ncols; j++) {
            row[2 * j]     = re[j];
            row[2 * j + 1] = im[j];
        }
    }
}

NB_MODULE(_core, m) {
    m.def("fht_1d_f32", &fht_1d_f32, nb::arg("arr").noconvert());
    m.def("fht_1d_f64", &fht_1d_f64, nb::arg("arr").noconvert());
    m.def("fht_2d_f32_rows", &fht_2d_f32_rows, nb::arg("arr").noconvert(), nb::arg("num_threads"));
    m.def("fht_2d_f64_rows", &fht_2d_f64_rows, nb::arg("arr").noconvert(), nb::arg("num_threads"));
    m.def("fht_2d_f32_cols", &fht_2d_f32_cols, nb::arg("arr").noconvert(), nb::arg("num_threads"));
    m.def("fht_2d_f64_cols", &fht_2d_f64_cols, nb::arg("arr").noconvert(), nb::arg("num_threads"));
    m.def("fht_complex_1d_f32", &fht_complex_1d_impl<float, fht_float>,
          nb::arg("arr").noconvert(), nb::arg("scratch").noconvert());
    m.def("fht_complex_1d_f64", &fht_complex_1d_impl<double, fht_double>,
          nb::arg("arr").noconvert(), nb::arg("scratch").noconvert());
    m.def("fht_complex_2d_f32_rows", &fht_complex_2d_rows_impl<float, fht_float>,
          nb::arg("arr").noconvert(), nb::arg("scratch_re").noconvert(),
          nb::arg("scratch_im").noconvert(), nb::arg("num_threads"));
    m.def("fht_complex_2d_f64_rows", &fht_complex_2d_rows_impl<double, fht_double>,
          nb::arg("arr").noconvert(), nb::arg("scratch_re").noconvert(),
          nb::arg("scratch_im").noconvert(), nb::arg("num_threads"));
}
