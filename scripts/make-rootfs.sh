#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build a minimal rootfs for demos/tests.
#   scripts/make-rootfs.sh busybox  DIR   # ~1 MB: one static busybox + applet links
#   scripts/make-rootfs.sh alpine   DIR   # ~8 MB: Alpine minirootfs (has apk)
set -euo pipefail
kind=${1:?busybox|alpine}; dir=${2:?target dir}
mkdir -p "$dir"
case "$kind" in
  busybox)
    bb=$(command -v busybox || true)
    [[ -n "$bb" ]] || { echo "install busybox-static first" >&2; exit 1; }
    file -L "$bb" | grep -q 'statically linked' || echo "warning: $bb is not static; it may not run in the container" >&2
    mkdir -p "$dir/bin" "$dir/etc" "$dir/tmp"
    cp "$bb" "$dir/bin/busybox"
    for a in $("$bb" --list); do ln -sf busybox "$dir/bin/$a"; done
    echo "root:x:0:0:root:/root:/bin/sh" > "$dir/etc/passwd"
    ;;
  alpine)
    arch=$(uname -m)
    ver=3.20.3
    url="https://dl-cdn.alpinelinux.org/alpine/v${ver%.*}/releases/$arch/alpine-minirootfs-$ver-$arch.tar.gz"
    curl -fsSL "$url" | tar -xz -C "$dir"
    cp /etc/resolv.conf "$dir/etc/" 2>/dev/null || true
    ;;
  *) echo "unknown kind $kind" >&2; exit 1 ;;
esac
echo "rootfs ready at $dir"
