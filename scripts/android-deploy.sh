#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Cross-compile vesseld for arm64 Android and run it on a ROOTED device.
#
#   ANDROID_NDK=/path/to/ndk scripts/android-deploy.sh [--token SECRET]
#
# Then from the Vessel Console app on the same device connect to
# 127.0.0.1:50051, or from a workstation:
#   adb forward tcp:50051 tcp:50051 && vesselctl --addr 127.0.0.1:50051 info
#
# gRPC for Android is built via vcpkg (arm64-android triplet) because the
# NDK does not ship it; this is the slow part (~15 min, cached afterwards).
set -euo pipefail

: "${ANDROID_NDK:?set ANDROID_NDK to an NDK r26+ install}"
ABI=${ABI:-arm64-v8a}
API=${API:-29}
TOKEN=""
[[ "${1:-}" == "--token" ]] && TOKEN="$2"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$ROOT/build-android-$ABI"
VCPKG=${VCPKG_ROOT:-$HOME/vcpkg}

if [[ ! -x "$VCPKG/vcpkg" ]]; then
  git clone --depth 1 https://github.com/microsoft/vcpkg "$VCPKG"
  "$VCPKG/bootstrap-vcpkg.sh" -disableMetrics
fi
export ANDROID_NDK_HOME="$ANDROID_NDK"
"$VCPKG/vcpkg" install grpc protobuf openssl libbpf --triplet arm64-android

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DVCPKG_TARGET_TRIPLET=arm64-android \
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-$API" \
  -DVESSEL_BUILD_TESTS=OFF -DVESSEL_BUILD_GPU=OFF -DVESSEL_EBPF=AUTO \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target vesseld vesselctl

adb root >/dev/null 2>&1 || true
adb shell 'mkdir -p /data/local/vessel/rootfs/bin'
adb push "$BUILD/vesseld" "$BUILD/vesselctl" /data/local/vessel/
# A static arm64 busybox gives containers a minimal userland.
if [[ -f "$ROOT/third_party/busybox-arm64" ]]; then
  adb push "$ROOT/third_party/busybox-arm64" /data/local/vessel/rootfs/bin/busybox
  adb shell 'cd /data/local/vessel/rootfs/bin && for l in sh ls cat echo sleep ps; do ln -sf busybox $l; done'
fi
adb shell "su -c 'setenforce 0; cd /data/local/vessel && \
  nohup ./vesseld --listen 127.0.0.1:50051 --state-dir /data/local/vessel/state \
  --rootfs /data/local/vessel/rootfs ${TOKEN:+--token $TOKEN} > vesseld.log 2>&1 &'"
echo "vesseld started on device. Logs: adb shell cat /data/local/vessel/vesseld.log"
echo "NOTE: setenforce 0 is for development only; see platform/android/sepolicy for a real policy."
