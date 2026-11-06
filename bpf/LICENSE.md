The eBPF program in this directory (`vessel_trace.bpf.c`) is licensed under
GPL-2.0-only, because it calls GPL-only kernel helpers
(`bpf_get_current_cgroup_id`, `bpf_ringbuf_*`) and the kernel's verifier refuses
those helpers to programs that declare a non-GPL license.

`vessel_trace.h` is dual-licensed (GPL-2.0-only OR Apache-2.0) so that the
Apache-2.0 user-space daemon can include it.

Everything else in the repository is Apache-2.0 (see the top-level LICENSE).
