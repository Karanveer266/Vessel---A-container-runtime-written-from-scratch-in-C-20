# SPDX-License-Identifier: Apache-2.0
find_package(GTest REQUIRED)
include(GoogleTest)

# Pure unit tests: no privileges, no kernel features beyond procfs.
add_executable(vessel_unit_tests
  common_test.cpp cgroup_parse_test.cpp seccomp_test.cpp syscalls_test.cpp
  aggregator_test.cpp)
target_link_libraries(vessel_unit_tests PRIVATE vessel_trace GTest::gtest_main)
vessel_set_warnings(vessel_unit_tests)
gtest_discover_tests(vessel_unit_tests DISCOVERY_TIMEOUT 30)

# Integration tests: create real namespaces/cgroups. Each test probes the
# kernel first and GTEST_SKIPs with a reason when a feature is unavailable,
# so they are safe to run anywhere (CI runners, laptops, gVisor).
add_executable(vessel_runtime_tests runtime_test.cpp cgroup_live_test.cpp ptrace_test.cpp)
target_link_libraries(vessel_runtime_tests PRIVATE vessel_trace GTest::gtest_main)
vessel_set_warnings(vessel_runtime_tests)
gtest_discover_tests(vessel_runtime_tests DISCOVERY_TIMEOUT 30 PROPERTIES RUN_SERIAL TRUE)

if(TARGET vessel_daemon_lib)
  add_executable(vessel_daemon_tests daemon_unit_test.cpp daemon_e2e_test.cpp)
  target_link_libraries(vessel_daemon_tests PRIVATE vessel_daemon_lib GTest::gtest_main)
  vessel_set_warnings(vessel_daemon_tests)
  if(TARGET vessel-gpu-worker)
    target_compile_definitions(vessel_daemon_tests PRIVATE VESSEL_GPU_WORKER_PATH="$<TARGET_FILE:vessel-gpu-worker>")
    add_dependencies(vessel_daemon_tests vessel-gpu-worker)
  else()
    target_compile_definitions(vessel_daemon_tests PRIVATE VESSEL_GPU_WORKER_PATH="")
  endif()
  gtest_discover_tests(vessel_daemon_tests DISCOVERY_TIMEOUT 30 PROPERTIES RUN_SERIAL TRUE)
endif()
