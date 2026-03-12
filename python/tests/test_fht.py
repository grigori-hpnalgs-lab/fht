"""Tests for the fht Python package."""

import numpy as np
import pytest
import warnings

from fht import fht


# ── helpers ──

def reference_fht(x):
    """Naive recursive Hadamard transform for testing."""
    n = len(x)
    if n == 1:
        return x.copy()
    h = n // 2
    a = reference_fht(x[:h])
    b = reference_fht(x[h:])
    return np.concatenate([a + b, a - b])


# ── 1D real ──

@pytest.mark.parametrize("log_n", range(1, 15))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_1d_real(log_n, dtype):
    n = 1 << log_n
    rng = np.random.default_rng(42 + log_n)
    x = rng.standard_normal(n).astype(dtype)
    expected = reference_fht(x.astype(np.float64)).astype(dtype)
    result = fht(x, inplace=True)
    assert result is x
    atol = 1e-3 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-5)


# ── self-inverse property ──

@pytest.mark.parametrize("log_n", [4, 8, 12])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_self_inverse(log_n, dtype):
    n = 1 << log_n
    rng = np.random.default_rng(123)
    orig = rng.standard_normal(n).astype(dtype)
    x = orig.copy()
    fht(x)
    fht(x)
    x /= n
    atol = 1e-3 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(x, orig, atol=atol, rtol=1e-5)


# ── 1D complex ──

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_1d_complex(dtype):
    n = 256
    rng = np.random.default_rng(77)
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    re = rng.standard_normal(n).astype(real_dtype)
    im = rng.standard_normal(n).astype(real_dtype)
    x = (re + 1j * im).astype(dtype)

    expected_re = reference_fht(re.astype(np.float64)).astype(real_dtype)
    expected_im = reference_fht(im.astype(np.float64)).astype(real_dtype)
    expected = (expected_re + 1j * expected_im).astype(dtype)

    result = fht(x, inplace=True)
    atol = 1e-3 if dtype == np.complex64 else 1e-10
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-5)


# ── 2D real ──

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_2d_rows(dtype):
    nrows, ncols = 8, 64
    rng = np.random.default_rng(99)
    x = rng.standard_normal((nrows, ncols)).astype(dtype)
    expected = np.stack([reference_fht(x[i].astype(np.float64)).astype(dtype)
                         for i in range(nrows)])
    result = fht(x, axis=-1, inplace=True)
    assert result is x
    atol = 1e-3 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_2d_cols(dtype):
    nrows, ncols = 64, 8
    rng = np.random.default_rng(99)
    x = rng.standard_normal((nrows, ncols)).astype(dtype)
    expected = np.stack([reference_fht(x[:, j].astype(np.float64)).astype(dtype)
                         for j in range(ncols)], axis=1)
    result = fht(x, axis=0, inplace=True)
    atol = 1e-3 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-5)


# ── 2D complex ──

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_2d_complex(dtype):
    nrows, ncols = 4, 32
    rng = np.random.default_rng(55)
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    x = (rng.standard_normal((nrows, ncols)) + 1j * rng.standard_normal((nrows, ncols))).astype(dtype)

    expected_rows = []
    for i in range(nrows):
        re_ref = reference_fht(x[i].real.astype(np.float64)).astype(real_dtype)
        im_ref = reference_fht(x[i].imag.astype(np.float64)).astype(real_dtype)
        expected_rows.append((re_ref + 1j * im_ref).astype(dtype))
    expected = np.stack(expected_rows)

    result = fht(x, axis=-1, inplace=True)
    atol = 1e-3 if dtype == np.complex64 else 1e-10
    np.testing.assert_allclose(result, expected, atol=atol, rtol=1e-5)


# ── contiguity warning ──

def test_warns_on_non_contiguous_2d():
    x = np.asfortranarray(np.random.randn(8, 64).astype(np.float64))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fht(x, axis=-1)
        assert len(w) == 1
        assert "contiguous" in str(w[0].message).lower()


# ── inplace=False ──

def test_inplace_false():
    x = np.random.randn(64).astype(np.float64)
    orig = x.copy()
    result = fht(x, inplace=False)
    np.testing.assert_array_equal(x, orig)  # original unchanged
    expected = reference_fht(orig)
    np.testing.assert_allclose(result, expected, atol=1e-10)


# ── out parameter ──

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_out_1d(dtype):
    x = np.random.default_rng(10).standard_normal(128).astype(dtype)
    out = np.empty_like(x)
    result = fht(x, out=out)
    assert result is out
    expected = reference_fht(x.astype(np.float64)).astype(dtype)
    atol = 1e-3 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(out, expected, atol=atol, rtol=1e-5)
    # original unchanged
    np.testing.assert_array_equal(x, np.random.default_rng(10).standard_normal(128).astype(dtype))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_out_complex(dtype):
    rng = np.random.default_rng(11)
    x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(dtype)
    orig = x.copy()
    out = np.empty_like(x)
    result = fht(x, out=out)
    assert result is out
    np.testing.assert_array_equal(x, orig)  # original unchanged


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_out_2d(dtype):
    rng = np.random.default_rng(12)
    x = rng.standard_normal((4, 64)).astype(dtype)
    orig = x.copy()
    out = np.empty_like(x)
    result = fht(x, axis=-1, out=out)
    assert result is out
    np.testing.assert_array_equal(x, orig)
    expected = np.stack([reference_fht(orig[i].astype(np.float64)).astype(dtype)
                         for i in range(4)])
    atol = 1e-3 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(out, expected, atol=atol, rtol=1e-5)


def test_out_shape_mismatch():
    x = np.random.randn(64).astype(np.float64)
    out = np.empty(128, dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        fht(x, out=out)


def test_out_dtype_mismatch():
    x = np.random.randn(64).astype(np.float64)
    out = np.empty(64, dtype=np.float32)
    with pytest.raises(TypeError, match="dtype"):
        fht(x, out=out)


# ── error cases ──

def test_bad_dtype():
    with pytest.raises(TypeError, match="Unsupported dtype"):
        fht(np.array([1, 2, 3, 4], dtype=np.int32))


def test_non_power_of_2():
    with pytest.raises(ValueError, match="power of 2"):
        fht(np.random.randn(6).astype(np.float64))


def test_3d_error():
    with pytest.raises(ValueError, match="1D and 2D"):
        fht(np.random.randn(2, 4, 8).astype(np.float64))
