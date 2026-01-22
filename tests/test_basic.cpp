/*
 * FHT Library Basic Test
 *
 * Verifies that the Hadamard transform is working correctly.
 */

#include <fht/fht.h>
#include <cmath>
#include <cstdio>
#include <vector>

bool test_identity() {
    // FHT applied twice should return original (scaled by N)
    const int log_n = 10;
    const int n = 1 << log_n;

    std::vector<float> buf(n);
    std::vector<float> original(n);

    // Initialize with some data
    for (int i = 0; i < n; ++i) {
        buf[i] = static_cast<float>(i % 17) - 8.0f;
        original[i] = buf[i];
    }

    // Apply FHT twice
    int ret1 = fht_float(buf.data(), log_n);
    if (ret1 != 0) {
        printf("FAIL: fht_float returned %d\n", ret1);
        return false;
    }

    int ret2 = fht_float(buf.data(), log_n);
    if (ret2 != 0) {
        printf("FAIL: second fht_float returned %d\n", ret2);
        return false;
    }

    // Check that we get back original * N
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = original[i] * n;
        float err = std::fabs(buf[i] - expected);
        if (err > max_err) max_err = err;
    }

    if (max_err > 1e-3f * n) {
        printf("FAIL: identity test max error = %f\n", max_err);
        return false;
    }

    printf("PASS: identity test (max error = %e)\n", max_err);
    return true;
}

bool test_small_sizes() {
    printf("Testing small sizes...\n");

    for (int log_n = 1; log_n <= 10; ++log_n) {
        int n = 1 << log_n;
        std::vector<float> buf(n, 1.0f);

        int ret = fht_float(buf.data(), log_n);
        if (ret != 0) {
            printf("FAIL: log_n=%d returned %d\n", log_n, ret);
            return false;
        }

        // First element should be N (sum of all ones)
        if (std::fabs(buf[0] - n) > 1e-5f) {
            printf("FAIL: log_n=%d, buf[0]=%f, expected %d\n", log_n, buf[0], n);
            return false;
        }
    }

    printf("PASS: small sizes test\n");
    return true;
}

bool test_invalid_input() {
    float buf[16];

    // Negative log_n should fail
    int ret = fht_float(buf, -1);
    if (ret == 0) {
        printf("FAIL: negative log_n should return error\n");
        return false;
    }

    // log_n = 0 should succeed (single element)
    buf[0] = 42.0f;
    ret = fht_float(buf, 0);
    if (ret != 0) {
        printf("FAIL: log_n=0 should succeed\n");
        return false;
    }
    if (buf[0] != 42.0f) {
        printf("FAIL: log_n=0 should leave element unchanged\n");
        return false;
    }

    printf("PASS: invalid input test\n");
    return true;
}

int main() {
    printf("FHT Library Basic Tests\n");
    printf("=======================\n");
    printf("Platform: %s\n\n", FHT_PLATFORM_NAME);

    bool all_passed = true;

    all_passed &= test_identity();
    all_passed &= test_small_sizes();
    all_passed &= test_invalid_input();

    printf("\n");
    if (all_passed) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("Some tests FAILED\n");
        return 1;
    }
}
