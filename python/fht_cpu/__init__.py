"""Fast Hadamard Transform with SIMD acceleration.

On macOS, we pre-load a single libomp.dylib before importing the C extension
to prevent "libomp already initialized" crashes when numpy (or other packages)
also link to OpenMP.
"""

import os as _os
import sys as _sys

if _sys.platform == "darwin":
    import ctypes as _ctypes

    _LIBOMP_SEARCH = [
        # conda / pixi / mamba environment
        _os.path.join(_sys.prefix, "lib", "libomp.dylib"),
        # Homebrew Apple Silicon
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        # Homebrew Intel
        "/usr/local/opt/libomp/lib/libomp.dylib",
    ]

    _loaded = False
    for _p in _LIBOMP_SEARCH:
        if _os.path.isfile(_p):
            try:
                _ctypes.CDLL(_p, mode=_ctypes.RTLD_GLOBAL)
                _loaded = True
                break
            except OSError:
                continue

    if not _loaded:
        import warnings as _w
        _w.warn(
            "fht_cpu: could not locate libomp.dylib. "
            "If using conda/pixi: conda install llvm-openmp  "
            "If using pip: brew install libomp",
            RuntimeWarning,
            stacklevel=2,
        )

import warnings
import numpy as np
from ._core import (
    fht_1d_f32, fht_1d_f64,
    fht_2d_f32_rows, fht_2d_f64_rows,
    fht_2d_f32_cols, fht_2d_f64_cols,
    fht_complex_1d_f32, fht_complex_1d_f64,
    fht_complex_2d_f32_rows, fht_complex_2d_f64_rows,
)

__all__ = ["fht"]
__version__ = "1.0.0"

_SUPPORTED_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)


def fht(
    x: np.ndarray,
    axis: int = -1,
    inplace: bool = True,
    out: np.ndarray | None = None,
    num_threads: int = -1,
) -> np.ndarray:
    """Fast Hadamard Transform.

    Parameters
    ----------
    x : np.ndarray
        1D or 2D array of dtype float32, float64, complex64, or complex128.
        The size along the transform axis must be a power of 2.
    axis : int
        Axis along which to apply the transform (default: -1, last axis).
        Only used for 2D arrays.
    inplace : bool
        If True (default), transform in-place and return x.
        If False, allocate a new array and return it (x is unchanged).
        Ignored when ``out`` is provided.
    out : np.ndarray, optional
        Preallocated output array. Must have the same shape and dtype as x.
        When provided, x is copied into out and the transform is applied
        to out. The original x is not modified.
    num_threads : int
        Number of OpenMP threads for 2D arrays.
        -1 (default): use all available cores.
        0 or 1: single-threaded (no OpenMP).
        N > 1: use exactly N threads.
        Ignored for 1D arrays.

    Returns
    -------
    np.ndarray
        The transformed array.
    """
    if x.ndim not in (1, 2):
        raise ValueError("Only 1D and 2D arrays are supported")

    # ── dtype validation ──
    is_complex = np.issubdtype(x.dtype, np.complexfloating)

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"Unsupported dtype {x.dtype}; "
            "must be float32, float64, complex64, or complex128"
        )

    # ── resolve output target ──
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(
                f"out.shape {out.shape} != x.shape {x.shape}"
            )
        if out.dtype != x.dtype:
            raise TypeError(
                f"out.dtype {out.dtype} != x.dtype {x.dtype}"
            )
        np.copyto(out, x)
        x = out
    elif not inplace:
        x = x.copy()

    if is_complex:
        return _fht_complex(x, axis, num_threads)
    else:
        return _fht_real(x, axis, num_threads)


def _fht_real(x: np.ndarray, axis: int, num_threads: int) -> np.ndarray:
    if x.ndim == 1:
        x = _ensure_contiguous_1d(x)
        if x.dtype == np.float32:
            fht_1d_f32(x)
        else:
            fht_1d_f64(x)
    else:
        _fht_2d(x, axis, num_threads)
    return x


def _fht_complex(x: np.ndarray, axis: int, num_threads: int) -> np.ndarray:
    """Handle complex via C++ deinterleave→transform→reinterleave (with OpenMP)."""
    real_dtype = np.float32 if x.dtype == np.complex64 else np.float64

    if x.ndim == 1:
        n = x.shape[0]
        flat = x.view(real_dtype)  # (n*2,) interleaved
        scratch = np.empty(n * 2, dtype=real_dtype)
        if real_dtype == np.float32:
            fht_complex_1d_f32(flat, scratch)
        else:
            fht_complex_1d_f64(flat, scratch)
    else:
        axis_norm = axis % 2
        if axis_norm == 1:
            # transform along axis=1 (rows in C-order)
            if not x.flags["C_CONTIGUOUS"]:
                warnings.warn(
                    "Complex array is not C-contiguous but transform axis=1 "
                    "requires row-contiguous data. Copying to C-contiguous layout.",
                    stacklevel=3,
                )
                tmp = np.ascontiguousarray(x)
                _fht_complex_2d_rows(tmp, real_dtype, num_threads)
                x[...] = tmp
            else:
                _fht_complex_2d_rows(x, real_dtype, num_threads)
        else:
            # transform along axis=0 — transpose to make columns into rows,
            # transform, transpose back
            if x.flags["F_CONTIGUOUS"]:
                # columns are contiguous, view as transposed C-contiguous
                xt = np.ascontiguousarray(x.T)
                _fht_complex_2d_rows(xt, real_dtype, num_threads)
                x[...] = xt.T
            else:
                warnings.warn(
                    "Complex array is not F-contiguous but transform axis=0 "
                    "requires column-contiguous data. Copying.",
                    stacklevel=3,
                )
                xt = np.ascontiguousarray(x.T)
                _fht_complex_2d_rows(xt, real_dtype, num_threads)
                x[...] = xt.T
    return x


def _fht_complex_2d_rows(x: np.ndarray, real_dtype: np.dtype, num_threads: int) -> None:
    """In-place complex FHT along rows of a C-contiguous 2D complex array."""
    nrows, ncols = x.shape
    flat = x.view(real_dtype).reshape(nrows, ncols * 2)
    scratch_re = np.empty((nrows, ncols), dtype=real_dtype)
    scratch_im = np.empty((nrows, ncols), dtype=real_dtype)
    if real_dtype == np.float32:
        fht_complex_2d_f32_rows(flat, scratch_re, scratch_im, num_threads)
    else:
        fht_complex_2d_f64_rows(flat, scratch_re, scratch_im, num_threads)


def _fht_2d(x: np.ndarray, axis: int, num_threads: int) -> np.ndarray:
    """In-place FHT along `axis` of a 2D real array."""
    axis = axis % 2  # normalize -1 -> 1, etc.

    if axis == 1:
        # Transform along columns (axis=1) → need rows to be contiguous
        if x.flags["C_CONTIGUOUS"]:
            if x.dtype == np.float32:
                fht_2d_f32_rows(x, num_threads)
            else:
                fht_2d_f64_rows(x, num_threads)
        else:
            warnings.warn(
                "Array is not C-contiguous but transform axis=1 requires "
                "row-contiguous data. Copying to C-contiguous layout.",
                stacklevel=3,
            )
            tmp = np.ascontiguousarray(x)
            if tmp.dtype == np.float32:
                fht_2d_f32_rows(tmp, num_threads)
            else:
                fht_2d_f64_rows(tmp, num_threads)
            x[...] = tmp
    else:
        # Transform along rows (axis=0) → need columns to be contiguous
        if x.flags["F_CONTIGUOUS"]:
            if x.dtype == np.float32:
                fht_2d_f32_cols(x, num_threads)
            else:
                fht_2d_f64_cols(x, num_threads)
        else:
            warnings.warn(
                "Array is not F-contiguous but transform axis=0 requires "
                "column-contiguous data. Copying to Fortran-contiguous layout.",
                stacklevel=3,
            )
            tmp = np.asfortranarray(x)
            if tmp.dtype == np.float32:
                fht_2d_f32_cols(tmp, num_threads)
            else:
                fht_2d_f64_cols(tmp, num_threads)
            x[...] = tmp
    return x


def _ensure_contiguous_1d(x: np.ndarray) -> np.ndarray:
    if not x.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "1D array is not contiguous. Copying to contiguous layout.",
            stacklevel=4,
        )
        tmp = np.ascontiguousarray(x)
        return tmp
    return x
