# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(vessel VERSION 0.4.0 LANGUAGES C CXX
        DESCRIPTION "Minimal container runtime with eBPF observability, gRPC control plane and GPU jobs")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(VESSEL_BUILD_DAEMON "Build vesseld/vesselctl (needs gRPC + protobuf)" ON)
option(VESSEL_BUILD_TESTS  "Build unit/integration tests (GoogleTest)" ON)
option(VESSEL_BUILD_GPU    "Build the GPU worker (CUDA optional, CPU-simulated backend always)" ON)
set(VESSEL_EBPF "AUTO" CACHE STRING "Build eBPF tracer: AUTO|ON|OFF")
set_property(CACHE VESSEL_EBPF PROPERTY STRINGS AUTO ON OFF)
option(VESSEL_SANITIZE "Build with ASan+UBSan" OFF)
option(VESSEL_WERROR "Treat warnings as errors" OFF)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
  message(FATAL_ERROR "Vessel targets Linux/Android only (namespaces, cgroups, eBPF)")
endif()

include(VesselWarnings)
if(VESSEL_BUILD_TESTS)
  enable_testing()  # before any add_subdirectory so gpu/ tests register too
endif()

# --------------------------------------------------------------------------
# libvessel_common / libvessel_cgroup / libvessel_runtime / libvessel_trace
# Pure C++20 + Linux UAPI; no gRPC dependency (embeddable, testable).
# --------------------------------------------------------------------------
add_library(vessel_common STATIC
  src/common/fs.cpp src/common/strings.cpp src/common/log.cpp src/common/id.cpp)
target_include_directories(vessel_common PUBLIC include)
target_compile_definitions(vessel_common PUBLIC VESSEL_VERSION="${PROJECT_VERSION}")
vessel_set_warnings(vessel_common)
if(ANDROID)
  target_link_libraries(vessel_common PUBLIC log)
endif()

add_library(vessel_cgroup STATIC
  src/cgroup/parse.cpp src/cgroup/manager.cpp src/cgroup/v1.cpp src/cgroup/v2.cpp)
target_link_libraries(vessel_cgroup PUBLIC vessel_common)
vessel_set_warnings(vessel_cgroup)

add_library(vessel_runtime STATIC
  src/runtime/launcher.cpp src/runtime/child.cpp src/runtime/exec.cpp
  src/runtime/seccomp.cpp src/runtime/capabilities.cpp src/runtime/syscalls.cpp)
target_link_libraries(vessel_runtime PUBLIC vessel_cgroup)
vessel_set_warnings(vessel_runtime)

find_package(Threads REQUIRED)
add_library(vessel_trace STATIC
  src/trace/aggregator.cpp src/trace/ptrace_session.cpp src/trace/ebpf_tracer.cpp)
target_link_libraries(vessel_trace PUBLIC vessel_runtime Threads::Threads)
vessel_set_warnings(vessel_trace)

include(VesselBpf)
vessel_configure_bpf(vessel_trace)

# --------------------------------------------------------------------------
# Control plane
# --------------------------------------------------------------------------
if(VESSEL_BUILD_DAEMON)
  include(VesselGrpc)
  vessel_add_proto_library(vessel_proto ${CMAKE_CURRENT_SOURCE_DIR}/proto vessel/v1/vessel.proto)

  add_library(vessel_daemon_lib STATIC
    src/daemon/container_manager.cpp src/daemon/service.cpp
    src/daemon/event_bus.cpp src/daemon/log_buffer.cpp)
  target_include_directories(vessel_daemon_lib PUBLIC src/daemon)
  target_link_libraries(vessel_daemon_lib PUBLIC vessel_trace vessel_proto)
  vessel_set_warnings(vessel_daemon_lib)

  add_executable(vesseld src/daemon/main.cpp)
  target_link_libraries(vesseld PRIVATE vessel_daemon_lib)
  vessel_set_warnings(vesseld)

  add_executable(vesselctl src/cli/main.cpp)
  target_link_libraries(vesselctl PRIVATE vessel_proto vessel_common)
  vessel_set_warnings(vesselctl)

  install(TARGETS vesseld vesselctl RUNTIME DESTINATION bin)
endif()

if(VESSEL_BUILD_GPU AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/CMakeLists.txt)
  add_subdirectory(gpu)
endif()

add_executable(vessel_bench bench/bench_runtime.cpp)
target_link_libraries(vessel_bench PRIVATE vessel_trace)
vessel_set_warnings(vessel_bench)

if(VESSEL_BUILD_TESTS)
  add_subdirectory(tests)
endif()
