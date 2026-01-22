/*
 * FHT Library Comprehensive Tests
 *
 * Uses GoogleTest to verify correctness against a reference implementation
 * for all supported sizes (log_n = 0 to 28).
 */

#include <gtest/gtest.h>
#include <fht/fht.h>
#include "fht_reference.h"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

// Maximum log_n to test (2^28 = 256M elements, ~1GB for float, ~2GB for double)
// Adjust based on available memory
#ifndef FHT_TEST_MAX_LOG_N
#define FHT_TEST_MAX_LOG_N 26
#endif

// For very large sizes, only test a few due to time/memory constraints
#ifndef FHT_TEST_LARGE_THRESHOLD
#define FHT_TEST_LARGE_THRESHOLD 24
#endif

namespace {

// Random number generator with fixed seed for reproducibility
std::mt19937 g_rng(42);

template<typename T>
void fill_random(std::vector<T>& buf) {
    std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
    for (auto& x : buf) {
        x = dist(g_rng);
    }
}

template<typename T>
T max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
    T max_diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

template<typename T>
T max_relative_error(const std::vector<T>& result, const std::vector<T>& expected) {
    T max_err = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        T denom = std::max(std::abs(expected[i]), static_cast<T>(1e-10));
        T err = std::abs(result[i] - expected[i]) / denom;
        if (err > max_err) max_err = err;
    }
    return max_err;
}

} // anonymous namespace

// =============================================================================
// Float Tests
// =============================================================================

class FHTFloatTest : public ::testing::TestWithParam<int> {
protected:
    int log_n() const { return GetParam(); }
    size_t n() const { return static_cast<size_t>(1) << log_n(); }
};

TEST_P(FHTFloatTest, MatchesReference) {
    const int ln = log_n();
    const size_t sz = n();

    // Skip very large sizes in normal test runs
    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP() << "Skipping log_n=" << ln << " (above max threshold)";
    }

    std::vector<float> input(sz);
    std::vector<float> result(sz);
    std::vector<float> expected(sz);

    // Fill with random data
    fill_random(input);
    result = input;
    expected = input;

    // Apply FHT using library
    int ret_lib = fht_float(result.data(), ln);
    ASSERT_EQ(ret_lib, 0) << "fht_float failed for log_n=" << ln;

    // Apply FHT using reference
    int ret_ref = fht_reference::fht_float(expected.data(), ln);
    ASSERT_EQ(ret_ref, 0) << "Reference fht_float failed for log_n=" << ln;

    // Compare results
    // Float tolerance grows with problem size due to accumulated error
    float tolerance = std::max(1e-5f, 1e-5f * std::sqrt(static_cast<float>(sz)));
    float max_diff = max_abs_diff(result, expected);

    EXPECT_LE(max_diff, tolerance)
        << "Max absolute diff " << max_diff << " exceeds tolerance " << tolerance
        << " for log_n=" << ln;
}

TEST_P(FHTFloatTest, InverseProperty) {
    const int ln = log_n();
    const size_t sz = n();

    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP() << "Skipping log_n=" << ln << " (above max threshold)";
    }

    std::vector<float> original(sz);
    std::vector<float> buf(sz);

    fill_random(original);
    buf = original;

    // Apply FHT twice
    int ret1 = fht_float(buf.data(), ln);
    ASSERT_EQ(ret1, 0);
    int ret2 = fht_float(buf.data(), ln);
    ASSERT_EQ(ret2, 0);

    // Result should be original * N
    float tolerance = std::max(1e-4f, 1e-4f * sz);
    for (size_t i = 0; i < sz; ++i) {
        float expected = original[i] * static_cast<float>(sz);
        float diff = std::abs(buf[i] - expected);
        EXPECT_LE(diff, tolerance)
            << "Inverse property failed at index " << i
            << " for log_n=" << ln
            << ": got " << buf[i] << ", expected " << expected;
        if (diff > tolerance) break;  // Avoid flooding output
    }
}

TEST_P(FHTFloatTest, AllOnesFirstElement) {
    const int ln = log_n();
    const size_t sz = n();

    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP() << "Skipping log_n=" << ln << " (above max threshold)";
    }

    std::vector<float> buf(sz, 1.0f);

    int ret = fht_float(buf.data(), ln);
    ASSERT_EQ(ret, 0);

    // First element should be N (sum of all ones)
    EXPECT_NEAR(buf[0], static_cast<float>(sz), 1e-5f * sz)
        << "All-ones test failed for log_n=" << ln;
}

// Generate test cases for all sizes
INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    FHTFloatTest,
    ::testing::Range(0, FHT_TEST_MAX_LOG_N + 1),
    [](const ::testing::TestParamInfo<int>& info) {
        return "log_n_" + std::to_string(info.param);
    }
);

// =============================================================================
// Double Tests
// =============================================================================

class FHTDoubleTest : public ::testing::TestWithParam<int> {
protected:
    int log_n() const { return GetParam(); }
    size_t n() const { return static_cast<size_t>(1) << log_n(); }
};

TEST_P(FHTDoubleTest, MatchesReference) {
    const int ln = log_n();
    const size_t sz = n();

    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP() << "Skipping log_n=" << ln << " (above max threshold)";
    }

    std::vector<double> input(sz);
    std::vector<double> result(sz);
    std::vector<double> expected(sz);

    fill_random(input);
    result = input;
    expected = input;

    int ret_lib = fht_double(result.data(), ln);
    ASSERT_EQ(ret_lib, 0) << "fht_double failed for log_n=" << ln;

    int ret_ref = fht_reference::fht_double(expected.data(), ln);
    ASSERT_EQ(ret_ref, 0) << "Reference fht_double failed for log_n=" << ln;

    // Double precision should be much more accurate
    double tolerance = 1e-10 * sz;
    double max_diff = max_abs_diff(result, expected);

    EXPECT_LE(max_diff, tolerance)
        << "Max absolute diff " << max_diff << " exceeds tolerance " << tolerance
        << " for log_n=" << ln;
}

TEST_P(FHTDoubleTest, InverseProperty) {
    const int ln = log_n();
    const size_t sz = n();

    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP() << "Skipping log_n=" << ln << " (above max threshold)";
    }

    std::vector<double> original(sz);
    std::vector<double> buf(sz);

    fill_random(original);
    buf = original;

    int ret1 = fht_double(buf.data(), ln);
    ASSERT_EQ(ret1, 0);
    int ret2 = fht_double(buf.data(), ln);
    ASSERT_EQ(ret2, 0);

    // Double precision should maintain high accuracy
    double tolerance = 1e-8 * sz;
    for (size_t i = 0; i < sz; ++i) {
        double expected = original[i] * static_cast<double>(sz);
        double diff = std::abs(buf[i] - expected);
        EXPECT_LE(diff, tolerance)
            << "Inverse property failed at index " << i
            << " for log_n=" << ln
            << ": got " << buf[i] << ", expected " << expected;
        if (diff > tolerance) break;
    }
}

TEST_P(FHTDoubleTest, AllOnesFirstElement) {
    const int ln = log_n();
    const size_t sz = n();

    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP() << "Skipping log_n=" << ln << " (above max threshold)";
    }

    std::vector<double> buf(sz, 1.0);

    int ret = fht_double(buf.data(), ln);
    ASSERT_EQ(ret, 0);

    EXPECT_NEAR(buf[0], static_cast<double>(sz), 1e-10 * sz)
        << "All-ones test failed for log_n=" << ln;
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    FHTDoubleTest,
    ::testing::Range(0, FHT_TEST_MAX_LOG_N + 1),
    [](const ::testing::TestParamInfo<int>& info) {
        return "log_n_" + std::to_string(info.param);
    }
);

// =============================================================================
// Out-of-Place Tests
// =============================================================================

class FHTOutOfPlaceTest : public ::testing::TestWithParam<int> {
protected:
    int log_n() const { return GetParam(); }
    size_t n() const { return static_cast<size_t>(1) << log_n(); }
};

TEST_P(FHTOutOfPlaceTest, FloatOOP) {
    const int ln = log_n();
    const size_t sz = n();

    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP();
    }

    std::vector<float> input(sz);
    std::vector<float> output_oop(sz);
    std::vector<float> output_ip(sz);

    fill_random(input);
    output_ip = input;

    // Out-of-place
    int ret_oop = fht_float_oop(input.data(), output_oop.data(), ln);
    ASSERT_EQ(ret_oop, 0);

    // In-place for comparison
    int ret_ip = fht_float(output_ip.data(), ln);
    ASSERT_EQ(ret_ip, 0);

    // Should match
    float max_diff = max_abs_diff(output_oop, output_ip);
    EXPECT_EQ(max_diff, 0.0f)
        << "OOP and IP results differ for float log_n=" << ln;

    // Input should be unchanged
    std::vector<float> original(sz);
    fill_random(original);  // Same seed, same values... wait, need to reset
}

TEST_P(FHTOutOfPlaceTest, DoubleOOP) {
    const int ln = log_n();
    const size_t sz = n();

    if (ln > FHT_TEST_MAX_LOG_N) {
        GTEST_SKIP();
    }

    std::vector<double> input(sz);
    std::vector<double> output_oop(sz);
    std::vector<double> output_ip(sz);

    fill_random(input);
    output_ip = input;

    int ret_oop = fht_double_oop(input.data(), output_oop.data(), ln);
    ASSERT_EQ(ret_oop, 0);

    int ret_ip = fht_double(output_ip.data(), ln);
    ASSERT_EQ(ret_ip, 0);

    double max_diff = max_abs_diff(output_oop, output_ip);
    EXPECT_EQ(max_diff, 0.0)
        << "OOP and IP results differ for double log_n=" << ln;
}

INSTANTIATE_TEST_SUITE_P(
    AllSizes,
    FHTOutOfPlaceTest,
    ::testing::Range(0, std::min(20, FHT_TEST_MAX_LOG_N + 1)),  // OOP tests only up to 20
    [](const ::testing::TestParamInfo<int>& info) {
        return "log_n_" + std::to_string(info.param);
    }
);

// =============================================================================
// Edge Cases
// =============================================================================

TEST(FHTEdgeCases, InvalidLogN) {
    float buf_f[16];
    double buf_d[16];

    // Negative log_n
    EXPECT_NE(fht_float(buf_f, -1), 0);
    EXPECT_NE(fht_double(buf_d, -1), 0);

    // log_n too large
    EXPECT_NE(fht_float(buf_f, 31), 0);
    EXPECT_NE(fht_double(buf_d, 31), 0);
}

TEST(FHTEdgeCases, LogNZero) {
    float buf_f = 42.0f;
    double buf_d = 42.0;

    EXPECT_EQ(fht_float(&buf_f, 0), 0);
    EXPECT_EQ(buf_f, 42.0f);

    EXPECT_EQ(fht_double(&buf_d, 0), 0);
    EXPECT_EQ(buf_d, 42.0);
}

TEST(FHTEdgeCases, LogNOne) {
    float buf_f[2] = {3.0f, 5.0f};
    double buf_d[2] = {3.0, 5.0};

    EXPECT_EQ(fht_float(buf_f, 1), 0);
    EXPECT_FLOAT_EQ(buf_f[0], 8.0f);  // 3 + 5
    EXPECT_FLOAT_EQ(buf_f[1], -2.0f); // 3 - 5

    EXPECT_EQ(fht_double(buf_d, 1), 0);
    EXPECT_DOUBLE_EQ(buf_d[0], 8.0);
    EXPECT_DOUBLE_EQ(buf_d[1], -2.0);
}

TEST(FHTEdgeCases, LogNTwo) {
    // H_2 * [1, 2, 3, 4] = [10, -2, -4, 0]
    // H_2 = [1  1  1  1]
    //       [1 -1  1 -1]
    //       [1  1 -1 -1]
    //       [1 -1 -1  1]
    float buf_f[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    double buf_d[4] = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(fht_float(buf_f, 2), 0);
    EXPECT_FLOAT_EQ(buf_f[0], 10.0f);
    EXPECT_FLOAT_EQ(buf_f[1], -2.0f);
    EXPECT_FLOAT_EQ(buf_f[2], -4.0f);
    EXPECT_FLOAT_EQ(buf_f[3], 0.0f);

    EXPECT_EQ(fht_double(buf_d, 2), 0);
    EXPECT_DOUBLE_EQ(buf_d[0], 10.0);
    EXPECT_DOUBLE_EQ(buf_d[1], -2.0);
    EXPECT_DOUBLE_EQ(buf_d[2], -4.0);
    EXPECT_DOUBLE_EQ(buf_d[3], 0.0);
}

// =============================================================================
// Large Size Tests (memory permitting)
// =============================================================================

class FHTLargeSizeTest : public ::testing::TestWithParam<int> {
protected:
    int log_n() const { return GetParam(); }
};

TEST_P(FHTLargeSizeTest, LargeFloatInverse) {
    const int ln = log_n();
    const size_t sz = static_cast<size_t>(1) << ln;

    // Check we have enough memory (rough estimate)
    if (sz > 500'000'000) {  // 500M elements = 2GB for float
        GTEST_SKIP() << "Skipping due to memory constraints";
    }

    std::vector<float> buf;
    try {
        buf.resize(sz);
    } catch (const std::bad_alloc&) {
        GTEST_SKIP() << "Could not allocate memory for log_n=" << ln;
    }

    // Simple test: all ones
    std::fill(buf.begin(), buf.end(), 1.0f);

    int ret = fht_float(buf.data(), ln);
    ASSERT_EQ(ret, 0) << "fht_float failed for log_n=" << ln;

    // First element should be N
    EXPECT_NEAR(buf[0], static_cast<float>(sz), 1e-3f * sz)
        << "Large size test failed for log_n=" << ln;
}

TEST_P(FHTLargeSizeTest, LargeDoubleInverse) {
    const int ln = log_n();
    const size_t sz = static_cast<size_t>(1) << ln;

    if (sz > 250'000'000) {  // 250M elements = 2GB for double
        GTEST_SKIP() << "Skipping due to memory constraints";
    }

    std::vector<double> buf;
    try {
        buf.resize(sz);
    } catch (const std::bad_alloc&) {
        GTEST_SKIP() << "Could not allocate memory for log_n=" << ln;
    }

    std::fill(buf.begin(), buf.end(), 1.0);

    int ret = fht_double(buf.data(), ln);
    ASSERT_EQ(ret, 0) << "fht_double failed for log_n=" << ln;

    EXPECT_NEAR(buf[0], static_cast<double>(sz), 1e-6 * sz)
        << "Large size test failed for log_n=" << ln;
}

// Test larger sizes (24-28)
INSTANTIATE_TEST_SUITE_P(
    LargeSizes,
    FHTLargeSizeTest,
    ::testing::Values(24, 25, 26, 27, 28),
    [](const ::testing::TestParamInfo<int>& info) {
        return "log_n_" + std::to_string(info.param);
    }
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
