# SPDX-License-Identifier: MIT
# Vessel GPU module.
#
# Works standalone:      cmake -S gpu -B build -G Ninja
# or from the top level: add_subdirectory(gpu)
#
# Targets
#   vessel_gpu_core   static lib: common + CPU kernels + simulated NCCL + worker front-end (no CUDA)
#   vessel_gpu_cuda   static lib: CUDA kernels + CUDA/NCCL backend (only when CUDA is enabled)
#   vessel-gpu-worker executable (contract in include/vessel/gpu/worker.hpp)
#   vessel_gpu_tests  GoogleTest suite (registered with CTest)
cmake_minimum_required(VERSION 3.24)  # CUDA_ARCHITECTURES/check_language/CUDAToolkit, gtest_discover_tests DISCOVERY_MODE

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(VESSEL_GPU_STANDALONE ON)
else()
  set(VESSEL_GPU_STANDALONE OFF)
endif()

project(vessel_gpu VERSION 0.1.0 LANGUAGES CXX)

if(VESSEL_GPU_STANDALONE AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # Benchmarks are meaningless at -O0; default standalone builds to Release.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(VESSEL_GPU_CUDA AUTO CACHE STRING "Build the CUDA backend: AUTO (if a CUDA compiler is found), ON (required), OFF")
set_property(CACHE VESSEL_GPU_CUDA PROPERTY STRINGS AUTO ON OFF)
set(VESSEL_GPU_NCCL AUTO CACHE STRING "Use NCCL for the CUDA split-K/all-reduce ops: AUTO, ON, OFF")
set_property(CACHE VESSEL_GPU_NCCL PROPERTY STRINGS AUTO ON OFF)
set(NCCL_ROOT "" CACHE PATH "NCCL install prefix (contains include/nccl.h and lib/libnccl.so)")
option(VESSEL_GPU_TESTS "Build the vessel_gpu_tests GoogleTest suite" ON)
option(VESSEL_GPU_WERROR "Treat host-compiler warnings as errors" ON)
option(VESSEL_GPU_NATIVE "Compile CPU kernels with -march=native (sim backend performance; not relocatable)" ON)

# Default GPU architectures: Ampere (A100 = 80, RTX 30xx/A10 = 86), Ada (89), Hopper (90).
# Honour a user-provided CMAKE_CUDA_ARCHITECTURES (must be read before enable_language(CUDA),
# which otherwise fills in the compiler's default).
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(_vessel_default_archs "${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(_vessel_default_archs "80;86;89;90")
endif()
set(VESSEL_GPU_CUDA_ARCHITECTURES "${_vessel_default_archs}" CACHE STRING "CUDA architectures for the GPU kernels")

string(TOUPPER "${VESSEL_GPU_CUDA}" _vessel_cuda_mode)
string(TOUPPER "${VESSEL_GPU_NCCL}" _vessel_nccl_mode)
foreach(_mode IN ITEMS _vessel_cuda_mode _vessel_nccl_mode)
  if(NOT ${_mode} MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "vessel-gpu: VESSEL_GPU_CUDA/VESSEL_GPU_NCCL must be AUTO, ON or OFF (got '${${_mode}}')")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Toolchain probing
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

set(VESSEL_HAVE_CUDA OFF)
set(VESSEL_HAVE_NCCL OFF)
if(NOT _vessel_cuda_mode STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(VESSEL_HAVE_CUDA ON)
  elseif(_vessel_cuda_mode STREQUAL "ON")
    message(FATAL_ERROR "vessel-gpu: VESSEL_GPU_CUDA=ON but no CUDA compiler was found (set CMAKE_CUDA_COMPILER or CUDACXX)")
  endif()
endif()

if(VESSEL_HAVE_CUDA AND NOT _vessel_nccl_mode STREQUAL "OFF")
  find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME} ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include)
  # Some distributions (e.g. the pip/conda wheels) only ship the SONAME, not the dev symlink.
  find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME} ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(VESSEL_HAVE_NCCL ON)
  elseif(_vessel_nccl_mode STREQUAL "ON")
    message(FATAL_ERROR "vessel-gpu: VESSEL_GPU_NCCL=ON but NCCL was not found (set NCCL_ROOT)")
  endif()
endif()

message(STATUS "vessel-gpu: CUDA backend: ${VESSEL_HAVE_CUDA}  NCCL: ${VESSEL_HAVE_NCCL}")
if(VESSEL_HAVE_CUDA)
  message(STATUS "vessel-gpu: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, archs ${VESSEL_GPU_CUDA_ARCHITECTURES}")
endif()

# ---------------------------------------------------------------------------
# Common compile settings (target-scoped: never leak into the parent project)
# ---------------------------------------------------------------------------
add_library(vessel_gpu_warnings INTERFACE)
target_compile_options(vessel_gpu_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion>
  $<$<AND:$<BOOL:${VESSEL_GPU_WERROR}>,$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>)

set(_vessel_native_flag "")
if(VESSEL_GPU_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native VESSEL_GPU_HAS_MARCH_NATIVE)
  if(VESSEL_GPU_HAS_MARCH_NATIVE)
    set(_vessel_native_flag -march=native)
  endif()
endif()

# ---------------------------------------------------------------------------
# vessel_gpu_core — pure C++20
# ---------------------------------------------------------------------------
add_library(vessel_gpu_core STATIC
  src/common/common.cpp
  src/cpu/gemm_cpu.cpp
  src/cpu/attention_cpu.cpp
  src/sim/communicator.cpp
  src/sim/backend.cpp
  src/worker/worker.cpp)
add_library(vessel::gpu_core ALIAS vessel_gpu_core)
target_include_directories(vessel_gpu_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(vessel_gpu_core PUBLIC cxx_std_20)
set_target_properties(vessel_gpu_core PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_link_libraries(vessel_gpu_core PUBLIC Threads::Threads PRIVATE vessel_gpu_warnings)
# -march=native applies to the whole library (not individual files) so that
# inline functions shared between TUs are never compiled for different ISAs.
target_compile_options(vessel_gpu_core PRIVATE ${_vessel_native_flag}
  $<$<AND:$<CONFIG:Release,RelWithDebInfo>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3>)

# ---------------------------------------------------------------------------
# vessel_gpu_cuda — CUDA kernels + backend
# ---------------------------------------------------------------------------
if(VESSEL_HAVE_CUDA)
  set(_vessel_cuda_sources
    src/cuda/sgemm.cu
    src/cuda/flash_attention.cu
    src/cuda/backend.cu)
  if(VESSEL_HAVE_NCCL)
    list(APPEND _vessel_cuda_sources src/cuda/nccl_ops.cu)
  endif()
  add_library(vessel_gpu_cuda STATIC ${_vessel_cuda_sources})
  add_library(vessel::gpu_cuda ALIAS vessel_gpu_cuda)
  set_target_properties(vessel_gpu_cuda PROPERTIES
    CUDA_ARCHITECTURES "${VESSEL_GPU_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(vessel_gpu_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(vessel_gpu_cuda PUBLIC VESSEL_HAVE_CUDA=1)
  target_compile_options(vessel_gpu_cuda PRIVATE
    # -lineinfo: source correlation in Nsight Compute at zero runtime cost.
    # -Xptxas=-warn-spills: fail loudly in the log if a register tile spills.
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas=-warn-spills --expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${VESSEL_GPU_WERROR}>>:-Werror=all-warnings>)
  target_link_libraries(vessel_gpu_cuda PUBLIC vessel_gpu_core CUDA::cudart)
  if(VESSEL_HAVE_NCCL)
    target_compile_definitions(vessel_gpu_cuda PUBLIC VESSEL_HAVE_NCCL=1)
    target_include_directories(vessel_gpu_cuda PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(vessel_gpu_cuda PRIVATE ${NCCL_LIBRARY})
  endif()
endif()

# ---------------------------------------------------------------------------
# vessel-gpu-worker
# ---------------------------------------------------------------------------
add_executable(vessel-gpu-worker src/worker/main.cpp)
target_link_libraries(vessel-gpu-worker PRIVATE vessel_gpu_core vessel_gpu_warnings)
if(VESSEL_HAVE_CUDA)
  target_link_libraries(vessel-gpu-worker PRIVATE vessel_gpu_cuda)
endif()
install(TARGETS vessel-gpu-worker RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(VESSEL_GPU_TESTS)
  find_package(GTest)
  if(GTest_FOUND)
    if(VESSEL_GPU_STANDALONE)
      enable_testing()
    endif()
    include(GoogleTest)
    add_executable(vessel_gpu_tests
      tests/test_common.cpp
      tests/test_ring_allreduce.cpp
      tests/test_gemm_cpu.cpp
      tests/test_attention_cpu.cpp
      tests/test_worker.cpp
      tests/test_simt_kernels.cpp)
    # test_simt_kernels.cpp compiles the CUDA kernel headers as host C++.
    target_include_directories(vessel_gpu_tests PRIVATE src tests)
    set_target_properties(vessel_gpu_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(vessel_gpu_tests PRIVATE vessel_gpu_core vessel_gpu_warnings GTest::gtest GTest::gtest_main)
    gtest_discover_tests(vessel_gpu_tests DISCOVERY_TIMEOUT 60 PROPERTIES LABELS "gpu;unit" TIMEOUT 120)

    # End-to-end: run the real binary with small problems and require ok=1.
    set(_vessel_smoke
      "gemm\;--m\;67\;--n\;45\;--k\;33"
      "splitk\;--m\;64\;--n\;50\;--k\;99\;--ranks\;3"
      "flash\;--bh\;2\;--seq\;77\;--dim\;32\;--causal"
      "allreduce\;--count\;100003\;--ranks\;4")
    foreach(_case IN LISTS _vessel_smoke)
      list(GET _case 0 _op)
      list(SUBLIST _case 1 -1 _rest)
      add_test(NAME vessel_gpu_worker_smoke_${_op}
        COMMAND vessel-gpu-worker --op ${_op} --backend sim --iters 2 --verify ${_rest})
      set_tests_properties(vessel_gpu_worker_smoke_${_op} PROPERTIES
        PASS_REGULAR_EXPRESSION "(^|\n)ok=1\n" LABELS "gpu;smoke" TIMEOUT 120)
    endforeach()
    add_test(NAME vessel_gpu_worker_rejects_bad_args COMMAND vessel-gpu-worker --op nope)
    set_tests_properties(vessel_gpu_worker_rejects_bad_args PROPERTIES
      PASS_REGULAR_EXPRESSION "(^|\n)ok=0\n" LABELS "gpu;smoke")
  else()
    message(WARNING "vessel-gpu: GoogleTest not found; vessel_gpu_tests will not be built")
  endif()
endif()
