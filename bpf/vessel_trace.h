/* SPDX-License-Identifier: (GPL-2.0-only OR Apache-2.0) */
/*
 * Event ABI shared by the eBPF program (kernel side) and vesseld (user side).
 *
 * Rules: fixed-width types only, explicit padding, no pointers, natural
 * alignment, total size a multiple of 8. The struct is copied verbatim out of
 * the BPF ring buffer, so any layout drift between the two compilers would be
 * silent corruption — guarded by _Static_assert on both sides.
 */
#ifndef VESSEL_TRACE_H
#define VESSEL_TRACE_H

#ifdef __bpf__
#include <linux/types.h>
#else
#include <linux/types.h>
#endif

#define VESSEL_COMM_LEN 16
#define VESSEL_MAX_CONTAINERS 256
#define VESSEL_MAX_TRACKED_TASKS 65536

enum vessel_ev_kind {
	VESSEL_EV_SYSCALL = 1,
	VESSEL_EV_MEMORY = 2,
	VESSEL_EV_NET = 3,
	VESSEL_EV_EXEC = 4,
	VESSEL_EV_EXIT = 5,
	VESSEL_EV_OOM_KILL = 6,
};

struct vessel_event {
	__u64 ts_ns;      /* bpf_ktime_get_ns() (CLOCK_MONOTONIC) */
	__u64 latency_ns; /* sys_exit - sys_enter                 */
	__u64 cgroup_id;
	__u64 args[3];
	__s64 ret;
	__u64 bytes;      /* MEMORY: length arg; NET: skb->len    */
	__u32 tgid;
	__u32 tid;
	__s32 nr;         /* syscall nr, -1 if n/a                */
	__u16 kind;       /* enum vessel_ev_kind                  */
	__u8 egress;
	__u8 slot;        /* container slot (index into userspace table) */
	char comm[VESSEL_COMM_LEN];
};

/* Per-(container, syscall) exact aggregates, updated in-kernel. */
struct vessel_sc_key {
	__u32 slot;
	__s32 nr;
};
struct vessel_sc_val {
	__u64 count;
	__u64 errors;
	__u64 total_ns;
	__u64 max_ns;
};

/* Per-container network byte/packet counters (cgroup_skb). */
struct vessel_net_val {
	__u64 rx_bytes, tx_bytes, rx_packets, tx_packets;
};

/* Runtime configuration, written by userspace into a 1-entry array map. */
struct vessel_cfg {
	__u64 min_latency_ns; /* don't emit ring events below this (stats still counted) */
	__u32 errors_only;    /* only emit failing syscalls                */
	__u32 emit_net;       /* emit per-packet NET events (expensive)    */
	__u32 mute;           /* no subscribers: aggregate only, emit nothing */
	__u32 pad;
};

#ifndef __bpf__
#ifdef __cplusplus
static_assert(sizeof(struct vessel_event) == 96, "vessel_event ABI drift");
static_assert(sizeof(struct vessel_sc_key) == 8, "vessel_sc_key ABI drift");
#endif
#else
_Static_assert(sizeof(struct vessel_event) == 96, "vessel_event ABI drift");
#endif

#endif /* VESSEL_TRACE_H */
