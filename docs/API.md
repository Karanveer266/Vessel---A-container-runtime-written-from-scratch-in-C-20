# Vessel API (vessel.v1)

Single source of truth: [`proto/vessel/v1/vessel.proto`](../proto/vessel/v1/vessel.proto).
It is consumed by the daemon (grpc++), `vesselctl` (grpc++) and the Android
console (grpc-kotlin + protobuf-javalite). Nothing is copied; the Android
Gradle build references the file in place.

## Compatibility rules

* Field numbers are never reused; removed fields become `reserved`.
* Every enum has `*_UNSPECIFIED = 0`. Clients map unknown values to
  "unspecified" (the Android mappers handle `UNRECOGNIZED` explicitly).
* Units are part of field names (`_bytes`, `_usec`, `_ns`, `_millicores`).
* New behaviour is additive: new optional fields, new RPCs. Breaking
  changes go to `vessel.v2` in a new package.

## Authentication

`authorization: Bearer <token>` metadata on every call when the daemon is
started with `--token` / `$VESSEL_TOKEN`. The comparison is constant-time
(`CRYPTO_memcmp`). TLS is enabled with `--tls-cert/--tls-key`.

## Error model

Daemon errors carry an errno and the stage that failed. They map to gRPC
status codes like this:

| errno | gRPC status | Typical cause |
|---|---|---|
| ENOENT | NOT_FOUND | unknown container id/name/prefix, missing rootfs |
| EEXIST | ALREADY_EXISTS | duplicate container name |
| EINVAL | INVALID_ARGUMENT | empty argv, bad name, unknown syscall filter |
| EPERM/EACCES | PERMISSION_DENIED | daemon lacks privilege for a namespace/cgroup op |
| EBUSY/EALREADY | FAILED_PRECONDITION | remove running container without force, double start |
| ENOTSUP/ENOSYS | UNIMPLEMENTED | eBPF requested but unavailable, limits without cgroups |
| ETIMEDOUT | DEADLINE_EXCEEDED | freezer did not settle |
| ENOSPC/ENOMEM | RESOURCE_EXHAUSTED | all eBPF container slots in use |

The message always names the failed setup stage, e.g.
`container setup failed at stage 'pivot_root' (op #12): Invalid argument (errno 22)`.

## Streams and back-pressure

| RPC | Producer | Policy when the client is slow |
|---|---|---|
| `StreamStats` | sampled per tick | samples are *state*: `Write()` blocks under HTTP/2 flow control and the next tick simply happens later. Nothing queues up |
| `StreamTrace` | eBPF poller / ptrace threads | per-subscriber `BoundedQueue` (8192, drop-oldest). The loss is reported in-band as `TraceEvent.dropped_events` |
| `StreamLogs` | log pump | per-container 1 MiB ring with byte-offset cursors. A reader that falls behind skips ahead instead of blocking the container |

## Container references

Anywhere a `ContainerRef.id` is accepted you can pass the full 12-char id, a
unique prefix of at least 3 characters, or the container name.
