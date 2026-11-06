// SPDX-License-Identifier: Apache-2.0
// User-space half of the eBPF backend (libbpf + bpftool-generated skeleton).
//
// Compiled only when VESSEL_HAVE_EBPF is defined (clang + bpftool + libbpf
// found at configure time); otherwise EbpfTracer::open() returns ENOTSUP and
// the daemon selects the ptrace backend.
#include "vessel/trace/trace.h"

#ifdef VESSEL_HAVE_EBPF

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <unistd.h>

#include <atomic>
#include <bitset>
#include <cstring>
#include <thread>

#include "vessel/common/clock.h"
#include "vessel/common/log.h"
#include "vessel_trace.h"
// bpftool-generated C; silence C++-only style warnings for this header only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wpedantic"
#include "vessel_trace.skel.h"
#pragma GCC diagnostic pop

namespace vessel::trace {
namespace {

int libbpfPrint(enum libbpf_print_level lvl, const char* fmt, va_list ap) {
  if (lvl == LIBBPF_DEBUG) return 0;
  char buf[512];
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  size_t n = std::strlen(buf);
  if (n && buf[n - 1] == '\n') buf[n - 1] = 0;
  VLOG_DEBUG("libbpf", "{}", buf);
  return 0;
}

Kind kindOf(uint16_t k) {
  switch (k) {
    case VESSEL_EV_MEMORY: return Kind::kMemory;
    case VESSEL_EV_NET: return Kind::kNet;
    case VESSEL_EV_EXEC: return Kind::kExec;
    case VESSEL_EV_EXIT: return Kind::kExit;
    case VESSEL_EV_OOM_KILL: return Kind::kOomKill;
    default: return Kind::kSyscall;
  }
}

class EbpfTracerImpl final : public EbpfTracer {
 public:
  explicit EbpfTracerImpl(Sink sink) : sink_(std::move(sink)) {}
  ~EbpfTracerImpl() override {
    stop_.store(true);
    if (poller_.joinable()) poller_.join();
    for (auto& l : skb_links_)
      for (auto* p : l) if (p) bpf_link__destroy(p);
    if (rb_) ring_buffer__free(rb_);
    if (skel_) vessel_trace_bpf__destroy(skel_);
  }

  Status load() {
    libbpf_set_print(libbpfPrint);
    skel_ = vessel_trace_bpf__open();
    if (!skel_) return Error::fromErrno("bpf open skeleton");
    if (vessel_trace_bpf__load(skel_) != 0) return Error::fromErrno("bpf load (verifier)");
    // Attach the tracepoint programs only; cgroup_skb programs are attached
    // per container cgroup in attach(). The OOM tracepoint is optional
    // (absent on some kernels).
    struct { bpf_program* p; bool optional; } progs[] = {
        {skel_->progs.vessel_sys_enter, false}, {skel_->progs.vessel_sys_exit, false},
        {skel_->progs.vessel_fork, false},      {skel_->progs.vessel_exec, false},
        {skel_->progs.vessel_exit, false},      {skel_->progs.vessel_oom, true},
    };
    for (auto& [p, optional] : progs) {
      bpf_link* l = bpf_program__attach(p);
      if (!l) {
        if (optional) {
          VLOG_WARN("ebpf", "optional program {} not attached", bpf_program__name(p));
          continue;
        }
        return Error::fromErrno(std::string("bpf attach ") + bpf_program__name(p));
      }
      links_.push_back(l);
    }
    rb_ = ring_buffer__new(bpf_map__fd(skel_->maps.events), &EbpfTracerImpl::onSample, this, nullptr);
    if (!rb_) return Error::fromErrno("ring_buffer__new");
    mono_to_real_ = clk::monoToRealtimeOffsetNs();
    poller_ = std::thread([this] { pollLoop(); });
    return okStatus();
  }

  Result<int> attach(const std::string& id, uint64_t cgroup_id, int cgroup_fd) override {
    std::lock_guard lk(mu_);
    int slot = -1;
    for (int i = 0; i < VESSEL_MAX_CONTAINERS; ++i)
      if (!used_[i]) { slot = i; break; }
    if (slot < 0) return Error{ENOSPC, "ebpf: all container slots in use"};
    used_[slot] = true;
    ids_[slot] = id;
    resetSlotMaps(slot);
    if (cgroup_id != 0) {
      __u8 s = static_cast<__u8>(slot);
      bpf_map__update_elem(skel_->maps.cgroup_slots, &cgroup_id, sizeof cgroup_id, &s, sizeof s, BPF_ANY);
    }
    if (cgroup_fd >= 0) {
      skb_links_[slot][0] = bpf_program__attach_cgroup(skel_->progs.vessel_skb_ingress, cgroup_fd);
      skb_links_[slot][1] = bpf_program__attach_cgroup(skel_->progs.vessel_skb_egress, cgroup_fd);
      if (!skb_links_[slot][0] || !skb_links_[slot][1])
        VLOG_WARN("ebpf", "cgroup_skb attach failed for {}: network counters unavailable", id);
    }
    cgroup_ids_[slot] = cgroup_id;
    return slot;
  }

  Status trackTask(int slot, pid_t pid) override {
    __u32 tid = static_cast<__u32>(pid);
    __u8 s = static_cast<__u8>(slot);
    if (bpf_map__update_elem(skel_->maps.tracked_tasks, &tid, sizeof tid, &s, sizeof s, BPF_ANY) != 0)
      return Error::fromErrno("ebpf track task");
    return okStatus();
  }

  Status configure(int slot, const FilterConfig& c) override {
    vessel_cfg cfg{};
    cfg.min_latency_ns = c.min_latency_ns;
    cfg.errors_only = c.errors_only;
    cfg.emit_net = c.emit_net;
    cfg.mute = c.mute;
    __u32 k = static_cast<__u32>(slot);
    if (bpf_map__update_elem(skel_->maps.slot_cfg, &k, sizeof k, &cfg, sizeof cfg, BPF_ANY) != 0)
      return Error::fromErrno("ebpf configure");
    return okStatus();
  }

  void detach(int slot) override {
    std::lock_guard lk(mu_);
    if (slot < 0 || slot >= VESSEL_MAX_CONTAINERS || !used_[slot]) return;
    for (auto*& l : skb_links_[slot]) {
      if (l) bpf_link__destroy(l);
      l = nullptr;
    }
    if (cgroup_ids_[slot]) bpf_map__delete_elem(skel_->maps.cgroup_slots, &cgroup_ids_[slot], sizeof(uint64_t), 0);
    used_[slot] = false;
    ids_[slot].clear();
  }

  std::vector<SyscallStat> summary(int slot) override {
    // PERCPU_HASH: one value per *possible* CPU; sum them in user space.
    const int ncpu = libbpf_num_possible_cpus();
    std::vector<vessel_sc_val> vals(static_cast<size_t>(ncpu));
    std::vector<SyscallStat> out;
    vessel_sc_key key{}, next{};
    vessel_sc_key* cur = nullptr;
    int fd = bpf_map__fd(skel_->maps.sc_stats);
    while (bpf_map_get_next_key(fd, cur, &next) == 0) {
      key = next;
      cur = &key;
      if (static_cast<int>(key.slot) != slot) continue;
      if (bpf_map_lookup_elem(fd, &key, vals.data()) != 0) continue;
      SyscallStat s;
      s.nr = key.nr;
      for (const auto& v : vals) {
        s.count += v.count;
        s.errors += v.errors;
        s.total_ns += v.total_ns;
        s.max_ns = std::max<uint64_t>(s.max_ns, v.max_ns);
      }
      out.push_back(s);
    }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.count > b.count; });
    return out;
  }

  NetCounters net(int slot) override {
    const int ncpu = libbpf_num_possible_cpus();
    std::vector<vessel_net_val> vals(static_cast<size_t>(ncpu));
    __u32 k = static_cast<__u32>(slot);
    NetCounters n;
    if (bpf_map_lookup_elem(bpf_map__fd(skel_->maps.net_stats), &k, vals.data()) == 0)
      for (const auto& v : vals) {
        n.rx_bytes += v.rx_bytes;
        n.tx_bytes += v.tx_bytes;
        n.rx_packets += v.rx_packets;
        n.tx_packets += v.tx_packets;
      }
    return n;
  }

  uint64_t drops() override {
    const int ncpu = libbpf_num_possible_cpus();
    std::vector<uint64_t> vals(static_cast<size_t>(ncpu));
    __u32 z = 0;
    uint64_t total = 0;
    if (bpf_map_lookup_elem(bpf_map__fd(skel_->maps.ring_drops), &z, vals.data()) == 0)
      for (auto v : vals) total += v;
    return total;
  }

 private:
  void resetSlotMaps(int slot) {
    vessel_net_val zero{};
    const int ncpu = libbpf_num_possible_cpus();
    std::vector<vessel_net_val> z(static_cast<size_t>(ncpu), zero);
    __u32 k = static_cast<__u32>(slot);
    bpf_map_update_elem(bpf_map__fd(skel_->maps.net_stats), &k, z.data(), BPF_ANY);
    vessel_cfg cfg{};
    bpf_map_update_elem(bpf_map__fd(skel_->maps.slot_cfg), &k, &cfg, BPF_ANY);
    // Remove stale per-syscall aggregates left by a previous occupant.
    int fd = bpf_map__fd(skel_->maps.sc_stats);
    vessel_sc_key key{}, next{};
    vessel_sc_key* cur = nullptr;
    std::vector<vessel_sc_key> stale;
    while (bpf_map_get_next_key(fd, cur, &next) == 0) {
      key = next;
      cur = &key;
      if (static_cast<int>(key.slot) == slot) stale.push_back(key);
    }
    for (auto& s : stale) bpf_map_delete_elem(fd, &s);
  }

  static int onSample(void* ctx, void* data, size_t size) {
    auto* self = static_cast<EbpfTracerImpl*>(ctx);
    if (size < sizeof(vessel_event)) return 0;
    const auto* r = static_cast<const vessel_event*>(data);
    Event e;
    {
      std::lock_guard lk(self->mu_);
      if (!self->used_[r->slot]) return 0;  // slot is u8 < VESSEL_MAX_CONTAINERS
      e.container_id = self->ids_[r->slot];
    }
    e.kind = kindOf(r->kind);
    e.backend = Backend::kEbpf;
    e.ts_unix_ns = static_cast<int64_t>(r->ts_ns) + self->mono_to_real_;
    e.pid = static_cast<int32_t>(r->tgid);
    e.tid = static_cast<int32_t>(r->tid);
    e.comm.assign(r->comm, strnlen(r->comm, sizeof r->comm));
    e.nr = r->nr;
    e.ret = r->ret;
    e.latency_ns = r->latency_ns;
    e.args = {r->args[0], r->args[1], r->args[2]};
    e.bytes = r->bytes;
    e.egress = r->egress != 0;
    uint64_t d = self->drops();
    e.dropped_before = d - self->last_drops_;
    self->last_drops_ = d;
    if (self->sink_) self->sink_(std::move(e));
    return 0;
  }

  void pollLoop() {
    // 100ms timeout bounds shutdown latency; ring_buffer__poll uses epoll and
    // wakes immediately when data is available (no busy polling).
    while (!stop_.load(std::memory_order_relaxed)) {
      int n = ring_buffer__poll(rb_, 100);
      if (n < 0 && n != -EINTR) {
        VLOG_ERROR("ebpf", "ring_buffer__poll failed: {}", n);
        break;
      }
    }
  }

  Sink sink_;
  vessel_trace_bpf* skel_ = nullptr;
  ring_buffer* rb_ = nullptr;
  std::vector<bpf_link*> links_;
  std::array<std::array<bpf_link*, 2>, VESSEL_MAX_CONTAINERS> skb_links_{};
  std::array<uint64_t, VESSEL_MAX_CONTAINERS> cgroup_ids_{};
  std::array<std::string, VESSEL_MAX_CONTAINERS> ids_{};
  std::bitset<VESSEL_MAX_CONTAINERS> used_;
  std::mutex mu_;
  std::thread poller_;
  std::atomic<bool> stop_{false};
  int64_t mono_to_real_ = 0;
  uint64_t last_drops_ = 0;
};

}  // namespace

bool EbpfTracer::compiledIn() noexcept { return true; }

Result<std::unique_ptr<EbpfTracer>> EbpfTracer::open(Sink sink) {
  auto t = std::make_unique<EbpfTracerImpl>(std::move(sink));
  VESSEL_TRY(t->load());
  return std::unique_ptr<EbpfTracer>(std::move(t));
}

}  // namespace vessel::trace

#else  // !VESSEL_HAVE_EBPF

namespace vessel::trace {
bool EbpfTracer::compiledIn() noexcept { return false; }
Result<std::unique_ptr<EbpfTracer>> EbpfTracer::open(Sink) {
  return Error{ENOTSUP, "vessel was built without eBPF support (clang/bpftool/libbpf missing)"};
}
}  // namespace vessel::trace

#endif
