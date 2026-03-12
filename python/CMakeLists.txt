cmake_minimum_required(VERSION 3.17)
project(fht_python LANGUAGES CXX C)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# OpenMP
find_package(OpenMP)

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

install(TARGETS _core LIBRARY DESTINATION fht)
