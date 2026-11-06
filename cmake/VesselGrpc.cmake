# SPDX-License-Identifier: Apache-2.0
# Locates protobuf + gRPC either through their CMake config packages (vcpkg,
# source builds, Android NDK toolchains) or pkg-config (Debian/Ubuntu ship
# grpc++ without CMake configs), and generates C++ stubs.
find_package(Protobuf QUIET)
find_package(gRPC CONFIG QUIET)
find_package(OpenSSL REQUIRED)  # CRYPTO_memcmp for constant-time token checks

if(gRPC_FOUND)
  set(VESSEL_GRPC_LIBS gRPC::grpc++)
  get_target_property(VESSEL_GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
  pkg_check_modules(PROTOBUF_PC REQUIRED IMPORTED_TARGET protobuf)
  set(VESSEL_GRPC_LIBS PkgConfig::GRPCPP PkgConfig::PROTOBUF_PC)
  find_program(VESSEL_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
endif()
if(NOT Protobuf_PROTOC_EXECUTABLE)
  find_program(Protobuf_PROTOC_EXECUTABLE protoc REQUIRED)
endif()

function(vessel_add_proto_library name proto_root proto_rel)
  get_filename_component(stem ${proto_rel} NAME_WE)
  get_filename_component(dir ${proto_rel} DIRECTORY)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/gen)
  set(srcs ${out}/${dir}/${stem}.pb.cc ${out}/${dir}/${stem}.grpc.pb.cc)
  set(hdrs ${out}/${dir}/${stem}.pb.h ${out}/${dir}/${stem}.grpc.pb.h)
  add_custom_command(
    OUTPUT ${srcs} ${hdrs}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${out}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
      --proto_path=${proto_root}
      --cpp_out=${out}
      --grpc_out=${out}
      --plugin=protoc-gen-grpc=${VESSEL_GRPC_CPP_PLUGIN}
      ${proto_root}/${proto_rel}
    DEPENDS ${proto_root}/${proto_rel}
    COMMENT "protoc ${proto_rel}"
    VERBATIM)
  add_library(${name} STATIC ${srcs})
  target_include_directories(${name} SYSTEM PUBLIC ${out})
  target_link_libraries(${name} PUBLIC ${VESSEL_GRPC_LIBS} OpenSSL::Crypto Threads::Threads)
endfunction()
