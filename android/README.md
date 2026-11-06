# Vessel Console (Android)

Operator console for the Vessel container runtime. It talks to `vesseld` over
gRPC and covers host capabilities, container lifecycle, live cgroup stats,
eBPF/ptrace syscall traces, logs, exec and GPU jobs.

* Kotlin 2.1, Jetpack Compose (Material 3), single activity, Navigation Compose (type-safe routes)
* gRPC: `grpc-kotlin` coroutine stubs + `protobuf-javalite`, OkHttp transport
* minSdk 26, target/compileSdk 35, Java 17 bytecode (builds on JDK 17 or 21)

## Modules

```
android/
├── app/            UI: Compose screens, ViewModels, DataStore settings, manual DI
├── core/api/       gRPC: codegen from ../proto, VesselClient, auth interceptor, channel mgmt
└── core/model/     Pure Kotlin: domain models, VesselRepository interface, ring buffers,
                    formatters/parsers, flow operators (retry/backoff, batching)
```

Dependencies only point downward: `app → core:api → core:model`.

* **`:core:model`** is plain Kotlin/JVM with no Android, protobuf or gRPC.
  ViewModels depend only on the `VesselRepository` interface defined here,
  so ViewModel tests use an in-memory fake and run in milliseconds.
* **`:core:api`** generates its code from the repo's single contract,
  `proto/vessel/v1/vessel.proto`. Gradle references it via `srcDir("../proto")`
  and never copies it. Generated classes never leave this module: `Mappers.kt`
  converts them to domain types, and `Errors.kt` turns gRPC `Status` into
  `VesselException`. This is also a JVM module, so the in-process gRPC tests
  need no Robolectric.
* **`:app`** handles Compose UI, ViewModels, DataStore persistence, Keystore
  token encryption and the Android network-aware channel.

## Build

Requirements: JDK 17+ and an Android SDK with `platforms;android-35` and `build-tools;35.0.0`.

```bash
cd android
export ANDROID_HOME=/path/to/android-sdk        # or create local.properties with sdk.dir=...
./gradlew :app:assembleDebug                    # → app/build/outputs/apk/debug/app-debug.apk
./gradlew testDebugUnitTest :core:model:test :core:api:test   # all JVM unit tests
./gradlew :app:installDebug                     # install on a connected device/emulator
```

The wrapper pins Gradle 8.11.1 and checks the distribution's SHA-256.

## Pointing the app at a daemon

Open the **Host** tab and enter host, port, bearer token and TLS on/off.
Settings are persisted (the token is encrypted with an Android Keystore key).
Editing them reconnects every live stream automatically.

`vesseld` listens on a unix socket by default. The app needs TCP, so start
the daemon with `--listen`:

| Scenario | Daemon side | App settings |
|---|---|---|
| **USB, daemon on your workstation** | `vesseld --listen 127.0.0.1:50051` | run `adb reverse tcp:50051 tcp:50051`, then host `127.0.0.1`, port `50051` |
| **Emulator, daemon on the host machine** | `vesseld --listen 127.0.0.1:50051` | host `10.0.2.2` (the emulator's alias for host loopback) |
| **Same rooted Android device** (vesseld runs on the phone as root, e.g. `su -c vesseld --listen 127.0.0.1:50051`) | loopback listener | host `127.0.0.1` (no adb needed) |
| **LAN** | `vesseld --listen 0.0.0.0:50051 --token $TOKEN --tls-cert cert.pem --tls-key key.pem` | host = server IP, token, **TLS on** |

For a daemon started with `--token T` (or `$VESSEL_TOKEN`), enter `T` as the
bearer token. Without a token the daemon disables auth, which is only
appropriate on loopback.

`adb reverse` forwards the *device's* `localhost:50051` to the *workstation's*
`localhost:50051` over USB. That means plaintext and a loopback-only daemon
are fine in this setup: traffic never touches the network.

The app warns whenever a token would be sent unencrypted to a non-loopback
host. For TLS with a self-signed CA, install the CA certificate on the device.
The network security config trusts user CAs.

Auth: every call carries `authorization: Bearer <token>` (`AuthInterceptor`).
If the token is blank, no header is sent, for daemons running without auth on
loopback.

## Architecture notes

* **UDF / state:** each screen has a ViewModel that exposes one immutable
  `StateFlow<UiState>`. Composables are stateless renderers that send intents
  back as method calls.
* **DI:** manual, through `AppContainer` (three singletons). The choice and
  its trade-offs are explained in `di/AppContainer.kt`.
* **Streams are lifecycle-aware.** The UI collects with
  `collectAsStateWithLifecycle()`, and ViewModels share streams with
  `WhileSubscribed(5 s)`. Backgrounding the app or switching detail tabs
  therefore cancels the gRPC call, while a rotation does not.
* **Reconnects:** streams retry with exponential backoff and jitter
  (0.5 s → 15 s cap, attempt counter resets after any item) and report
  `Reconnecting(attempt)` in-band so the UI can show a banner. Idempotent
  reads retry twice. Mutations are never retried automatically.
* **Bounded memory:**
  * trace: 2,000-event ring buffer, events batched every 250 ms
  * charts: 120-point primitive `FloatRingBuffer`s
  * logs: 5,000 lines
  * GPU history: 10 runs
* **Create container:** the main form covers name, rootfs, argv, env,
  hostname, network, seccomp, trace backend, CPU/memory/PIDs/cpuset limits and
  auto-remove. A collapsible **Advanced** section adds working directory,
  read-only rootfs, CPU weight (1..10000), disable swap, and repeatable bind
  mounts (host source, absolute container target, read-only). Mount targets
  must be absolute, cannot be `/` or contain `..`, and must be unique. The
  section opens automatically when it contains a validation error.
* **Charts:** custom Compose `Canvas` (`ui/components/LineChart.kt`) with no
  chart library. They support touch-and-drag crosshairs with tooltips.

## Tests

```bash
./gradlew :core:model:test :core:api:test testDebugUnitTest
```

* `core:model`: ring buffers, formatters, quantity/argv parsers, backoff,
  retry/batch flow operators, UTF-8 line assembly, stats history, filters
* `core:api`: proto ↔ domain mappers (including unknown-enum fallback),
  gRPC status mapping, and `VesselClient` against an **in-process gRPC
  server**. These cover auth header, retry policy, stream reconnect, unique
  trace keys, and resubscribing on settings change.
* `app`: every ViewModel with a fake repository on `kotlinx-coroutines-test`
  virtual time (lifecycle stop/grace period, debounced filters, bounded
  buffers, pause/clear, polling, form validation including mounts and CPU
  weight), plus chart geometry
