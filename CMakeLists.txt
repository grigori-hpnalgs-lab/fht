cmake_minimum_required(VERSION 3.14)

project(fht
    VERSION 1.0.0
    DESCRIPTION "Fast Hadamard Transform library with SSE/AVX and NEON support"
    LANGUAGES CXX C
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Options
option(FHT_OPTIMIZE_FOR_HOST "Re-optimize NEON kernels for this host (ARM only)" OFF)
option(FHT_BUILD_TESTS "Build test programs" OFF)
set(FHT_OPTIMIZATION_SIZES "4;8;12;16;20;24" CACHE STRING
    "Sizes (log_n) to optimize when FHT_OPTIMIZE_FOR_HOST is ON")

# Header-only interface library
add_library(fht INTERFACE)
add_library(fht::fht ALIAS fht)

target_include_directories(fht INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(fht INTERFACE cxx_std_11)

# Platform detection and configuration
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    message(STATUS "FHT: ARM64 platform detected, using NEON implementation")
    target_compile_definitions(fht INTERFACE FHT_PLATFORM_ARM=1)

    # Optional re-optimization at build time
    if(FHT_OPTIMIZE_FOR_HOST)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)

        set(FHT_OPT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/fht_neon_optimized.h")

        # Convert list to space-separated string for the command
        string(REPLACE ";" " " FHT_SIZES_STR "${FHT_OPTIMIZATION_SIZES}")

        add_custom_command(
            OUTPUT ${FHT_OPT_HEADER}
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/optimize_v7_grid.py
                --sizes ${FHT_OPTIMIZATION_SIZES}
                --output ${FHT_OPT_HEADER}
                --quick
                --no-compare
            DEPENDS
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/optimize_v7_grid.py
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_neon_v7.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
            COMMENT "Optimizing FHT NEON kernels for this host (sizes: ${FHT_SIZES_STR})..."
            VERBATIM
        )

        add_custom_target(fht_optimize ALL DEPENDS ${FHT_OPT_HEADER})

        target_compile_definitions(fht INTERFACE
            FHT_USE_OPTIMIZED_HEADER=1
            FHT_OPTIMIZED_HEADER_PATH="${FHT_OPT_HEADER}"
        )
        target_include_directories(fht INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        )

        message(STATUS "FHT: Re-optimization enabled. Header will be generated during build.")
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(STATUS "FHT: x86_64 platform detected, using SSE/AVX implementation")
    target_compile_definitions(fht INTERFACE FHT_PLATFORM_X86=1)

    # Check for AVX support
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    if(COMPILER_SUPPORTS_AVX)
        message(STATUS "FHT: AVX support detected, recommend adding -mavx to compile flags")
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i386|i686)$")
    message(STATUS "FHT: x86 (32-bit) platform detected, using SSE implementation")
    target_compile_definitions(fht INTERFACE FHT_PLATFORM_X86=1)

else()
    message(WARNING "FHT: Unknown platform '${CMAKE_SYSTEM_PROCESSOR}', build may fail")
endif()

# Installation
install(TARGETS fht
    EXPORT fhtTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/fht
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT fhtTargets
    FILE fhtTargets.cmake
    NAMESPACE fht::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fht
)

# Package configuration
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fhtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fhtConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fht
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/fhtConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/fhtConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/fhtConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fht
)

# Tests
if(FHT_BUILD_TESTS)
    enable_testing()

    # Use CPM to fetch GoogleTest
    include(cmake/CPM.cmake)

    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.14.0
        OPTIONS
            "INSTALL_GTEST OFF"
            "gtest_force_shared_crt ON"
    )

    # Platform-specific compile options
    set(FHT_TEST_COMPILE_OPTIONS "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        list(APPEND FHT_TEST_COMPILE_OPTIONS -march=armv8-a+simd)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        if(COMPILER_SUPPORTS_AVX)
            list(APPEND FHT_TEST_COMPILE_OPTIONS -mavx)
        endif()
    endif()

    # Basic test (legacy, quick sanity check)
    add_executable(fht_test_basic tests/test_basic.cpp)
    target_link_libraries(fht_test_basic PRIVATE fht::fht)
    target_compile_options(fht_test_basic PRIVATE ${FHT_TEST_COMPILE_OPTIONS})
    add_test(NAME fht_basic COMMAND fht_test_basic)

    # Comprehensive GoogleTest suite
    add_executable(fht_test tests/test_fht.cpp)
    target_link_libraries(fht_test PRIVATE fht::fht GTest::gtest_main)
    target_include_directories(fht_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_options(fht_test PRIVATE ${FHT_TEST_COMPILE_OPTIONS})
    target_compile_features(fht_test PRIVATE cxx_std_17)

    # If optimizing for host, tests must wait for the optimized header
    if(TARGET fht_optimize)
        add_dependencies(fht_test_basic fht_optimize)
        add_dependencies(fht_test fht_optimize)
    endif()

    # Allow customizing max test size via cmake
    if(DEFINED FHT_TEST_MAX_LOG_N)
        target_compile_definitions(fht_test PRIVATE FHT_TEST_MAX_LOG_N=${FHT_TEST_MAX_LOG_N})
    endif()

    include(GoogleTest)
    gtest_discover_tests(fht_test)
endif()

# Print summary
message(STATUS "")
message(STATUS "FHT Configuration Summary:")
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  Platform:             ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Install prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Optimize for host:    ${FHT_OPTIMIZE_FOR_HOST}")
message(STATUS "  Build tests:          ${FHT_BUILD_TESTS}")
message(STATUS "")
