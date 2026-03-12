cmake_minimum_required(VERSION 3.17)
project(fht_python LANGUAGES CXX C)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# OpenMP — try standard find first, then fall back to Homebrew hints for AppleClang
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    foreach(_prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS "${_prefix}/lib/libomp.dylib")
            message(STATUS "Using Homebrew libomp from ${_prefix}")
            set(OpenMP_C_FLAGS "-Xclang -fopenmp")
            set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${_prefix}/lib/libomp.dylib")
            include_directories("${_prefix}/include")
            find_package(OpenMP)
            break()
        endif()
    endforeach()
endif()
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP not found — building without parallelism")
endif()

nanobind_add_module(_core src/fht_bindings.cpp)

# Point to the header-only FHT library
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Platform-specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    target_compile_options(_core PRIVATE -march=armv8-a+simd)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx" HAS_AVX)
    if(HAS_AVX)
        target_compile_options(_core PRIVATE -mavx)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _core LIBRARY DESTINATION fht_cpu)
