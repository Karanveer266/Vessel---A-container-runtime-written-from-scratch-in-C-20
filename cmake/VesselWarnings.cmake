# SPDX-License-Identifier: Apache-2.0
function(vessel_set_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
      -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wnull-dereference
      -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough>
    $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra>)
  if(VESSEL_WERROR)
    target_compile_options(${target} PRIVATE -Werror)
  endif()
  # Hardening: these binaries run as root and parse untrusted input.
  target_compile_options(${target} PRIVATE -fstack-protector-strong -fno-plt
    $<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=3>)
  if(VESSEL_SANITIZE)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
  get_target_property(type ${target} TYPE)
  if(type STREQUAL "EXECUTABLE")
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_options(${target} PRIVATE -pie -Wl,-z,relro,-z,now -Wl,-z,noexecstack)
  endif()
endfunction()
