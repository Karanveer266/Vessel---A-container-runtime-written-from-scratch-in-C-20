// SPDX-License-Identifier: GPL-2.0-only
/*
 * vessel_trace.bpf.c — kernel-side observability for Vessel containers.
 *
 * Programs
 *   tp/raw_syscalls/sys_enter    stash {nr, args, ts} per thread
 *   tp/raw_syscalls/sys_exit     join with enter, aggregate, emit event
 *   tp/sched/sched_process_fork  propagate "tracked" to children/threads
 *   tp/sched/sched_process_exec  EXEC events
 *   tp/sched/sched_process_exit  untrack + EXIT events
 *   tp/oom/mark_victim           OOM_KILL events for tracked tasks
 *   cgroup_skb/ingress|egress    per-container byte/packet counters
 *
 * Design choices (details.md §eBPF has the long version)
 *   * Classic tracepoints with hand-declared, ABI-stable context structs —
 *     NOT tp_btf/fentry + CO-RE. Many Android/vendor kernels ship without
 *     CONFIG_DEBUG_INFO_BTF; raw_syscalls/sched tracepoint formats have been
 *     stable for >10 years. The cost is ~20-40ns extra per event vs. fentry,
 *     which is irrelevant next to the syscall itself.
 *   * Two container-membership mechanisms:
 *       1. cgroup id  (bpf_get_current_cgroup_id) — authoritative on cgroup v2
 *       2. task tree  (tid map seeded by vesseld, propagated on fork) — works
 *          on cgroup v1 hosts (most Android devices) where (1) is meaningless.
 *   * Exact counting in-kernel (PERCPU_HASH, no locks, no atomics) and
 *     *filtered sampling* to user space through a BPF ring buffer: the ring
 *     can drop under load (counted), the aggregates never do.
 *   * Ring buffer (5.8+) over perf buffer: one shared buffer (better memory
 *     efficiency), preserves cross-CPU ordering, reserve/commit avoids an
 *     extra copy.
 */
#include <linux/bpf.h>
#include <linux/types.h>
#include <bpf/bpf_helpers.h>

#include "vessel_trace.h"

char LICENSE[] SEC("license") = "GPL";  /* bpf_get_current_cgroup_id etc. are GPL-only */

/* ---- tracepoint contexts (see /sys/kernel/tracing/events/.../format) ---- */
struct tp_sys_enter {
	__u64 __common;          /* common_type/flags/preempt_count/pid */
	long id;
	unsigned long args[6];
};
struct tp_sys_exit {
	__u64 __common;
	long id;
	long ret;
};
struct tp_fork {
	__u64 __common;
	char parent_comm[16];
	__s32 parent_pid;
	char child_comm[16];
	__s32 child_pid;
};
struct tp_exec {
	__u64 __common;
	__u32 filename_loc;      /* __data_loc char[] filename */
	__s32 pid;
	__s32 old_pid;
};
struct tp_exit {
	__u64 __common;
	char comm[16];
	__s32 pid;
	__s32 prio;
};
struct tp_mark_victim {
	__u64 __common;
	__s32 pid;
};

/* ---- maps ---------------------------------------------------------------- */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, VESSEL_MAX_CONTAINERS);
	__type(key, __u64);        /* cgroup id */
	__type(value, __u8);       /* slot      */
} cgroup_slots SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, VESSEL_MAX_TRACKED_TASKS);
	__type(key, __u32);        /* tid (task->pid) */
	__type(value, __u8);       /* slot            */
} tracked_tasks SEC(".maps");

struct enter_rec {
	__u64 ts;
	__u64 args[3];
	__s32 nr;
	__u8 slot;
	__u8 pad[3];
};
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, VESSEL_MAX_TRACKED_TASKS);
	__type(key, __u32);        /* tid */
	__type(value, struct enter_rec);
} inflight SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_HASH);
	__uint(max_entries, 16384);
	__type(key, struct vessel_sc_key);
	__type(value, struct vessel_sc_val);
} sc_stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, VESSEL_MAX_CONTAINERS);
	__type(key, __u32);
	__type(value, struct vessel_net_val);
} net_stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, VESSEL_MAX_CONTAINERS);
	__type(key, __u32);
	__type(value, struct vessel_cfg);
} slot_cfg SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, __u64);
} ring_drops SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 4 << 20);   /* 4 MiB, power of two, page multiple */
} events SEC(".maps");

/* ---- helpers --------------------------------------------------------------- */

/* Returns the container slot of the current task or -1. Cheapest test first:
 * the cgroup-id lookup is a single hash probe and covers cgroup v2 hosts. */
static __always_inline int current_slot(void)
{
	__u64 cg = bpf_get_current_cgroup_id();
	__u8 *s = bpf_map_lookup_elem(&cgroup_slots, &cg);
	if (s)
		return *s;
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	s = bpf_map_lookup_elem(&tracked_tasks, &tid);
	return s ? *s : -1;
}

static __always_inline void count_drop(void)
{
	__u32 z = 0;
	__u64 *d = bpf_map_lookup_elem(&ring_drops, &z);
	if (d)
		(*d)++;          /* per-CPU: no atomic needed */
}

static __always_inline struct vessel_event *reserve(__u16 kind, int slot)
{
	struct vessel_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
	if (!e) {
		count_drop();
		return 0;
	}
	__u64 id = bpf_get_current_pid_tgid();
	e->ts_ns = bpf_ktime_get_ns();
	e->latency_ns = 0;
	e->cgroup_id = bpf_get_current_cgroup_id();
	e->args[0] = e->args[1] = e->args[2] = 0;
	e->ret = 0;
	e->bytes = 0;
	e->tgid = id >> 32;
	e->tid = (__u32)id;
	e->nr = -1;
	e->kind = kind;
	e->egress = 0;
	e->slot = (__u8)slot;
	bpf_get_current_comm(e->comm, sizeof(e->comm));
	return e;
}

/* Syscall numbers we classify as memory events (arch specific). */
#if defined(__TARGET_ARCH_x86)
#define NR_MMAP 9
#define NR_MUNMAP 11
#define NR_BRK 12
#define NR_MREMAP 25
#elif defined(__TARGET_ARCH_arm64)
#define NR_MMAP 222
#define NR_MUNMAP 215
#define NR_BRK 214
#define NR_MREMAP 216
#else
#error "define __TARGET_ARCH_x86 or __TARGET_ARCH_arm64"
#endif

/* ---- syscalls -------------------------------------------------------------- */

SEC("tracepoint/raw_syscalls/sys_enter")
int vessel_sys_enter(struct tp_sys_enter *ctx)
{
	int slot = current_slot();
	if (slot < 0)
		return 0;        /* fast path for every non-container task on the host */

	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	struct enter_rec r = {};
	r.ts = bpf_ktime_get_ns();
	r.nr = (__s32)ctx->id;
	r.slot = (__u8)slot;
	r.args[0] = ctx->args[0];
	r.args[1] = ctx->args[1];
	r.args[2] = ctx->args[2];
	bpf_map_update_elem(&inflight, &tid, &r, BPF_ANY);
	return 0;
}

SEC("tracepoint/raw_syscalls/sys_exit")
int vessel_sys_exit(struct tp_sys_exit *ctx)
{
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	struct enter_rec *r = bpf_map_lookup_elem(&inflight, &tid);
	if (!r)
		return 0;

	__u64 now = bpf_ktime_get_ns();
	__u64 lat = now - r->ts;
	long ret = ctx->ret;
	__u32 slot = r->slot;
	__s32 nr = r->nr;

	/* 1. exact aggregates — always, regardless of filters / ring pressure */
	struct vessel_sc_key k = { .slot = slot, .nr = nr };
	struct vessel_sc_val *v = bpf_map_lookup_elem(&sc_stats, &k);
	if (v) {
		v->count++;
		v->total_ns += lat;
		if (ret < 0)
			v->errors++;
		if (lat > v->max_ns)
			v->max_ns = lat;
	} else {
		struct vessel_sc_val nv = { .count = 1, .errors = ret < 0, .total_ns = lat, .max_ns = lat };
		bpf_map_update_elem(&sc_stats, &k, &nv, BPF_NOEXIST);
	}

	/* 2. filtered event stream */
	struct vessel_cfg *cfg = bpf_map_lookup_elem(&slot_cfg, &slot);
	int emit = 1;
	int mute = 0;
	if (cfg) {
		mute = cfg->mute;
		if (lat < cfg->min_latency_ns)
			emit = 0;
		if (cfg->errors_only && ret >= 0)
			emit = 0;
	}
	int is_mem = nr == NR_MMAP || nr == NR_MUNMAP || nr == NR_BRK || nr == NR_MREMAP;
	if (!mute && (emit || is_mem)) {
		struct vessel_event *e = reserve(is_mem ? VESSEL_EV_MEMORY : VESSEL_EV_SYSCALL, slot);
		if (e) {
			e->nr = nr;
			e->ret = ret;
			e->latency_ns = lat;
			e->args[0] = r->args[0];
			e->args[1] = r->args[1];
			e->args[2] = r->args[2];
			if (is_mem)
				e->bytes = nr == NR_MREMAP ? r->args[2] : (nr == NR_BRK ? 0 : r->args[1]);
			bpf_ringbuf_submit(e, 0);
		}
	}
	bpf_map_delete_elem(&inflight, &tid);
	return 0;
}

/* ---- process lifecycle ----------------------------------------------------- */

SEC("tracepoint/sched/sched_process_fork")
int vessel_fork(struct tp_fork *ctx)
{
	__u32 parent = (__u32)ctx->parent_pid;
	__u8 *s = bpf_map_lookup_elem(&tracked_tasks, &parent);
	if (s) {
		__u32 child = (__u32)ctx->child_pid;
		__u8 slot = *s;
		bpf_map_update_elem(&tracked_tasks, &child, &slot, BPF_ANY);
	}
	return 0;
}

SEC("tracepoint/sched/sched_process_exec")
int vessel_exec(struct tp_exec *ctx)
{
	int slot = current_slot();
	if (slot < 0)
		return 0;
	__u32 s32 = (__u32)slot;
	struct vessel_cfg *cfg = bpf_map_lookup_elem(&slot_cfg, &s32);
	if (cfg && cfg->mute)
		return 0;
	struct vessel_event *e = reserve(VESSEL_EV_EXEC, slot);
	if (e)
		bpf_ringbuf_submit(e, 0);
	return 0;
}

SEC("tracepoint/sched/sched_process_exit")
int vessel_exit(struct tp_exit *ctx)
{
	int slot = current_slot();
	__u32 tid = (__u32)bpf_get_current_pid_tgid();
	if (slot >= 0) {
		__u64 id = bpf_get_current_pid_tgid();
		if ((__u32)id == (__u32)(id >> 32)) {   /* thread-group leader only */
			struct vessel_event *e = reserve(VESSEL_EV_EXIT, slot);
			if (e)
				bpf_ringbuf_submit(e, 0);
		}
	}
	bpf_map_delete_elem(&tracked_tasks, &tid);
	bpf_map_delete_elem(&inflight, &tid);    /* exit_group never returns */
	return 0;
}

SEC("tracepoint/oom/mark_victim")
int vessel_oom(struct tp_mark_victim *ctx)
{
	__u32 pid = (__u32)ctx->pid;
	__u8 *s = bpf_map_lookup_elem(&tracked_tasks, &pid);
	/* mark_victim runs in the context of the allocating task, which is
	 * usually (not always) in the same container, so also try the cgroup. */
	int slot = s ? *s : current_slot();
	if (slot < 0)
		return 0;
	struct vessel_event *e = reserve(VESSEL_EV_OOM_KILL, slot);
	if (e) {
		e->tgid = pid;
		e->tid = pid;
		bpf_ringbuf_submit(e, 0);
	}
	return 0;
}

/* ---- network (cgroup v2 only; attached per container cgroup) ------------- */

static __always_inline int account_skb(struct __sk_buff *skb, int egress)
{
	__u64 cg = bpf_skb_cgroup_id(skb);
	__u8 *s = bpf_map_lookup_elem(&cgroup_slots, &cg);
	if (!s)
		return 1;
	__u32 slot = *s;
	struct vessel_net_val *n = bpf_map_lookup_elem(&net_stats, &slot);
	if (n) {
		if (egress) {
			n->tx_bytes += skb->len;
			n->tx_packets++;
		} else {
			n->rx_bytes += skb->len;
			n->rx_packets++;
		}
	}
	struct vessel_cfg *cfg = bpf_map_lookup_elem(&slot_cfg, &slot);
	if (cfg && cfg->emit_net) {
		struct vessel_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
		if (e) {
			__builtin_memset(e, 0, sizeof(*e));
			e->ts_ns = bpf_ktime_get_ns();
			e->cgroup_id = cg;
			e->kind = VESSEL_EV_NET;
			e->bytes = skb->len;
			e->egress = egress;
			e->slot = (__u8)slot;
			e->nr = -1;
			bpf_ringbuf_submit(e, 0);
		} else {
			count_drop();
		}
	}
	return 1;                /* observe only: never drop traffic */
}

SEC("cgroup_skb/ingress")
int vessel_skb_ingress(struct __sk_buff *skb) { return account_skb(skb, 0); }

SEC("cgroup_skb/egress")
int vessel_skb_egress(struct __sk_buff *skb) { return account_skb(skb, 1); }
