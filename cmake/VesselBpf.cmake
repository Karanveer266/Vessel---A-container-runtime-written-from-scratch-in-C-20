# SPDX-License-Identifier: Apache-2.0
# Builds bpf/vessel_trace.bpf.c with clang -target bpf, generates the libbpf
# skeleton with bpftool and links libbpf. The resulting object is *embedded*
# in the daemon binary via the skeleton (no .o file to install/locate).
function(vessel_configure_bpf target)
  if(VESSEL_EBPF STREQUAL "OFF")
    message(STATUS "eBPF tracer: disabled")
    return()
  endif()
  find_program(VESSEL_BPF_CLANG NAMES clang clang-19 clang-18 clang-17 clang-16 clang-15)
  find_program(VESSEL_BPFTOOL NAMES bpftool PATHS /usr/sbin /usr/local/sbin)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBBPF QUIET IMPORTED_TARGET libbpf>=0.8)
  endif()
  if(NOT VESSEL_BPF_CLANG OR NOT VESSEL_BPFTOOL OR NOT LIBBPF_FOUND)
    if(VESSEL_EBPF STREQUAL "ON")
      message(FATAL_ERROR "VESSEL_EBPF=ON but clang/bpftool/libbpf not found")
    endif()
    message(STATUS "eBPF tracer: unavailable (clang=${VESSEL_BPF_CLANG} bpftool=${VESSEL_BPFTOOL} libbpf=${LIBBPF_FOUND}) — ptrace only")
    return()
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(arch x86)
    set(multiarch x86_64-linux-gnu)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(arch arm64)
    set(multiarch aarch64-linux-gnu)
  else()
    message(STATUS "eBPF tracer: unsupported arch ${CMAKE_SYSTEM_PROCESSOR}")
    return()
  endif()

  set(src ${PROJECT_SOURCE_DIR}/bpf/vessel_trace.bpf.c)
  set(gen ${CMAKE_CURRENT_BINARY_DIR}/bpf)
  set(obj ${gen}/vessel_trace.bpf.o)
  set(skel ${gen}/vessel_trace.skel.h)
  add_custom_command(
    OUTPUT ${obj}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${gen}
    COMMAND ${VESSEL_BPF_CLANG} -O2 -g -target bpf -D__TARGET_ARCH_${arch}
            -I/usr/include/${multiarch} -I${PROJECT_SOURCE_DIR}/bpf
            -Wall -Werror -c ${src} -o ${obj}
    DEPENDS ${src} ${PROJECT_SOURCE_DIR}/bpf/vessel_trace.h
    COMMENT "clang -target bpf vessel_trace.bpf.c"
    VERBATIM)
  add_custom_command(
    OUTPUT ${skel}
    COMMAND ${VESSEL_BPFTOOL} gen skeleton ${obj} name vessel_trace_bpf > ${skel}
    DEPENDS ${obj}
    COMMENT "bpftool gen skeleton"
    VERBATIM)
  add_custom_target(vessel_bpf_skel DEPENDS ${skel})
  add_dependencies(${target} vessel_bpf_skel)
  target_include_directories(${target} PRIVATE ${gen} ${PROJECT_SOURCE_DIR}/bpf)
  target_compile_definitions(${target} PUBLIC VESSEL_HAVE_EBPF=1)
  target_link_libraries(${target} PUBLIC PkgConfig::LIBBPF)
  message(STATUS "eBPF tracer: enabled (${arch})")
endfunction()
