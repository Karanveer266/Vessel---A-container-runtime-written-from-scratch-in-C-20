// SPDX-License-Identifier: Apache-2.0
// Parent side of container creation. See launcher.h for the protocol.
#include "vessel/runtime/launcher.h"

#include <fcntl.h>
#include <linux/sched.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sys/mount.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>

#include "child.h"
#include "vessel/common/fs.h"
#include "vessel/common/log.h"
#include "vessel/common/strings.h"
#include "vessel/runtime/capabilities.h"
#include "vessel/runtime/seccomp.h"

#ifndef CLONE_INTO_CGROUP
#define CLONE_INTO_CGROUP 0x200000000ULL
#endif

namespace vessel::runtime {
namespace {

using detail::ChildPlan;
using detail::Op;
using detail::OpKind;

// Owns every byte the child will touch. Lives on the parent's stack frame for
// the duration of launch(); the child gets a CoW copy at clone time.
struct PlanStorage {
  std::deque<std::string> strings;  // deque: push_back never moves elements
  std::vector<Op> ops;
  std::vector<char*> argv, envp;
  std::vector<const char*> exec_candidates;
  seccomp::Program seccomp;
  ChildPlan plan;

  const char* s(std::string v) { return strings.emplace_back(std::move(v)).c_str(); }

  void op(OpKind k, const char* target, bool optional = false) {
    ops.push_back(Op{k, optional, nullptr, target, nullptr, nullptr, 0, 0755});
  }
  void mount(const char* src, const char* target, const char* fstype, unsigned long flags,
             const char* data = nullptr, bool optional = false) {
    ops.push_back(Op{OpKind::kMount, optional, src, target, fstype, data, flags, 0});
  }
  void mkdirP(const std::string& root, const std::string& rel, bool optional = false) {
    // Emit one mkdir per path component so the child never needs to parse paths.
    std::string cur = root;
    for (auto part : str::split(rel, '/')) {
      cur = fs::join(cur, part);
      ops.push_back(Op{OpKind::kMkdir, optional, nullptr, s(cur), nullptr, nullptr, 0, 0755});
    }
  }
  void touch(const std::string& path, bool optional = false) {
    ops.push_back(Op{OpKind::kTouch, optional, nullptr, s(path), nullptr, nullptr, 0, 0644});
  }
  void symlink(const char* target_of_link, const std::string& linkpath) {
    ops.push_back(Op{OpKind::kSymlink, true, target_of_link, s(linkpath), nullptr, nullptr, 0, 0});
  }
  void bind(const std::string& src, const std::string& dst, bool ro, bool optional = false) {
    const char* d = s(dst);
    mount(s(src), d, nullptr, MS_BIND | MS_REC, nullptr, optional);
    // MS_RDONLY is ignored on the initial bind; it needs a remount. Inside a
    // user namespace the remount must also preserve "locked" flags
    // (nosuid/nodev/noexec) of the source or the kernel returns EPERM, so
    // we request them all — they are only ever more restrictive.
    if (ro) mount(nullptr, d, nullptr, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV, nullptr, optional);
  }
};

// Device nodes every POSIX program assumes exist. Bind-mounted from the host
// because mknod(2) is not permitted inside a user namespace.
constexpr const char* kDevNodes[] = {"null", "zero", "full", "random", "urandom", "tty"};

// procfs entries that leak host information or offer kernel attack surface.
// Masked with /dev/null (files) or made read-only (dirs), same list as the
// OCI runtime-spec defaults.
constexpr const char* kMaskedFiles[] = {"/proc/kcore", "/proc/keys", "/proc/timer_list", "/proc/sched_debug",
                                        "/proc/latency_stats", "/proc/sysrq-trigger"};
constexpr const char* kReadOnlyDirs[] = {"/proc/sys", "/proc/irq", "/proc/bus", "/proc/fs"};

void buildMountPlan(PlanStorage& st, const RuntimeSpec& spec, const LaunchOptions& opts, const std::string& root) {
  if (spec.new_cgroup_ns) st.op(OpKind::kUnshareCgroup, nullptr, /*optional=*/true);

  // 1. Stop mount events propagating back to the host (systemd makes "/"
  //    MS_SHARED by default; without this our proc/tmpfs mounts would appear
  //    in the host namespace).
  st.mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE);

  // 2. pivot_root requires new_root to be a mount point.
  const char* r = st.s(root);
  if (spec.rootfs == "/") st.mount("/", r, nullptr, MS_BIND | MS_REC);
  else st.mount(r, r, nullptr, MS_BIND | MS_REC);

  // 3. User-requested bind mounts (before /proc,/dev so they cannot shadow them).
  for (const auto& m : spec.mounts) {
    st.mkdirP(root, m.target);
    st.bind(m.source, fs::join(root, m.target), m.read_only);
  }

  // 4. /proc — must be a *new* procfs instance so it reflects our pid ns.
  st.mkdirP(root, "proc");
  st.mount("proc", st.s(fs::join(root, "proc")), "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC);

  // 5. /dev — private tmpfs populated with a minimal set of nodes.
  st.mkdirP(root, "dev");
  st.mount("tmpfs", st.s(fs::join(root, "dev")), "tmpfs", MS_NOSUID | MS_STRICTATIME, "mode=755,size=65536k");
  for (const char* n : kDevNodes) {
    std::string dst = fs::join(root, std::string("dev/") + n);
    st.touch(dst, true);
    st.bind(std::string("/dev/") + n, dst, false, /*optional=*/true);
  }
  for (const auto& dev : spec.devices) {  // e.g. GPU passthrough
    std::string dst = fs::join(root, dev);
    st.touch(dst);
    st.bind(dev, dst, false);
  }
  st.mkdirP(root, "dev/pts");
  st.mount("devpts", st.s(fs::join(root, "dev/pts")), "devpts", MS_NOSUID | MS_NOEXEC,
           "newinstance,ptmxmode=0666,mode=0620", true);
  st.symlink("pts/ptmx", fs::join(root, "dev/ptmx"));
  st.mkdirP(root, "dev/shm");
  st.mount("shm", st.s(fs::join(root, "dev/shm")), "tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC, "mode=1777,size=65536k");
  st.symlink("/proc/self/fd", fs::join(root, "dev/fd"));
  st.symlink("/proc/self/fd/0", fs::join(root, "dev/stdin"));
  st.symlink("/proc/self/fd/1", fs::join(root, "dev/stdout"));
  st.symlink("/proc/self/fd/2", fs::join(root, "dev/stderr"));

  // 6. /sys read-only. sysfs can only be mounted by the owner of the netns,
  //    so this works with a private netns and is skipped in host-net mode.
  st.mkdirP(root, "sys", true);
  st.mount("sysfs", st.s(fs::join(root, "sys")), "sysfs", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC, nullptr, true);
  if (!opts.skip_cgroupfs) {
    // cgroup2 mounted after unshare(CLONE_NEWCGROUP) shows only our subtree:
    // the workload sees its own limits at /sys/fs/cgroup (what JVMs, Go and
    // .NET runtimes read to size their heaps/GOMAXPROCS).
    st.mount("cgroup2", st.s(fs::join(root, "sys/fs/cgroup")), "cgroup2", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC,
             nullptr, true);
  }

  // 7. Writable scratch when the image is read-only.
  if (spec.read_only_rootfs || spec.rootfs == "/") {
    st.mkdirP(root, "tmp", true);
    st.mount("tmpfs", st.s(fs::join(root, "tmp")), "tmpfs", MS_NOSUID | MS_NODEV, "mode=1777,size=65536k", true);
  }

  // 8. Switch root.
  st.op(OpKind::kChdir, r);
  st.op(OpKind::kPivot, nullptr);

  // 9. Post-pivot hardening, paths now relative to the new root.
  for (const char* f : kMaskedFiles) st.mount("/dev/null", f, nullptr, MS_BIND, nullptr, true);
  for (const char* d : kReadOnlyDirs) {
    st.mount(d, d, nullptr, MS_BIND | MS_REC, nullptr, true);
    st.mount(nullptr, d, nullptr, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC, nullptr, true);
  }
  if (spec.read_only_rootfs || spec.rootfs == "/")
    st.mount(nullptr, "/", nullptr, MS_BIND | MS_REMOUNT | MS_RDONLY, nullptr, false);
}

void buildExec(PlanStorage& st, const RuntimeSpec& spec) {
  for (const auto& a : spec.args) st.argv.push_back(const_cast<char*>(st.s(a)));
  st.argv.push_back(nullptr);

  bool has_path = false, has_home = false;
  for (const auto& e : spec.env) {
    has_path |= str::startsWith(e, "PATH=");
    has_home |= str::startsWith(e, "HOME=");
    st.envp.push_back(const_cast<char*>(st.s(e)));
  }
  std::string path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
  if (!has_path) st.envp.push_back(const_cast<char*>(st.s("PATH=" + path)));
  else
    for (const auto& e : spec.env)
      if (str::startsWith(e, "PATH=")) path = e.substr(5);
  if (!has_home) st.envp.push_back(const_cast<char*>(st.s("HOME=/root")));
  st.envp.push_back(const_cast<char*>(st.s("container=vessel")));
  st.envp.push_back(nullptr);

  // PATH resolution happens here, in the parent, producing candidate
  // absolute paths; the child just tries them in order (execvp() allocates
  // and is not async-signal-safe).
  const std::string& prog = spec.args.front();
  if (prog.find('/') != std::string::npos) {
    st.exec_candidates.push_back(st.s(prog));
  } else {
    for (auto dir : str::split(path, ':')) st.exec_candidates.push_back(st.s(fs::join(dir, prog)));
  }
}

Status writeIdMaps(pid_t pid, const RuntimeSpec& spec, bool rootless) {
  auto fmt = [](const std::vector<IdMap>& m) {
    std::string out;
    for (const auto& e : m) out += std::to_string(e.inside) + " " + std::to_string(e.outside) + " " + std::to_string(e.count) + "\n";
    return out;
  };
  const std::string base = "/proc/" + std::to_string(pid);
  // An unprivileged writer must disable setgroups(2) before writing gid_map
  // (CVE-2014-8989: dropping supplementary groups could bypass "negative" ACLs).
  if (rootless) VESSEL_TRY(fs::writeFile(base + "/setgroups", "deny"));
  auto uid = spec.uid_map.empty() ? defaultUidMap(rootless) : spec.uid_map;
  auto gid = spec.gid_map.empty() ? defaultGidMap(rootless) : spec.gid_map;
  VESSEL_TRY(fs::writeFile(base + "/uid_map", fmt(uid)));
  VESSEL_TRY(fs::writeFile(base + "/gid_map", fmt(gid)));
  return okStatus();
}

struct Pipe {
  UniqueFd rd, wr;
  static Result<Pipe> make(int flags = O_CLOEXEC) {
    int fds[2];
    if (::pipe2(fds, flags) != 0) return Error::fromErrno("pipe2");
    return Pipe{UniqueFd(fds[0]), UniqueFd(fds[1])};
  }
};

pid_t cloneChild(uint64_t flags, int cgroup_fd, int* pidfd_out, bool* used_clone3) {
  *used_clone3 = false;
  if (cgroup_fd >= 0) {
    // clone3 + CLONE_INTO_CGROUP (5.7+): the child is born inside its cgroup,
    // so not a single instruction of it is ever accounted to the daemon's
    // cgroup, and there is no window in which it runs unlimited.
    clone_args args{};
    args.flags = flags | CLONE_INTO_CGROUP | CLONE_PIDFD;
    args.exit_signal = SIGCHLD;
    args.cgroup = static_cast<uint64_t>(cgroup_fd);
    args.pidfd = reinterpret_cast<uint64_t>(pidfd_out);
    long r = ::syscall(SYS_clone3, &args, sizeof args);
    if (r >= 0) {
      *used_clone3 = true;
      return static_cast<pid_t>(r);
    }
    VLOG_DEBUG("launcher", "clone3 unavailable ({}), falling back to clone", std::strerror(errno));
  }
  *pidfd_out = -1;
  return static_cast<pid_t>(::syscall(SYS_clone, flags | SIGCHLD, nullptr, nullptr, nullptr, nullptr));
}

}  // namespace

std::vector<IdMap> defaultUidMap(bool rootless) {
  if (rootless) return {{0, static_cast<uint32_t>(::geteuid()), 1}};
  // Real root: identity map of the full range. The security boundary is then
  // capabilities + seccomp + namespaces (Docker's default model); use
  // --userns-remap for shifted ranges (see details.md §user namespaces).
  return {{0, 0, 4294967295u}};
}
std::vector<IdMap> defaultGidMap(bool rootless) {
  if (rootless) return {{0, static_cast<uint32_t>(::getegid()), 1}};
  return {{0, 0, 4294967295u}};
}

int exitCodeFromWaitStatus(int status) noexcept {
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return -1;
}

Status awaitExec(Process& p, bool nonblocking) {
  if (!p.exec_status_fd) return okStatus();
  if (nonblocking) {
    pollfd pfd{p.exec_status_fd.get(), POLLIN, 0};
    int r = ::poll(&pfd, 1, 0);
    if (r == 0) return Error{EAGAIN, "exec pending"};
  }
  detail::ChildError ce{};
  ssize_t n;
  do {
    n = ::read(p.exec_status_fd.get(), &ce, sizeof ce);
  } while (n < 0 && errno == EINTR);
  p.exec_status_fd.reset();
  if (n == 0) return okStatus();  // EOF: O_CLOEXEC closed it => execve succeeded
  if (n != static_cast<ssize_t>(sizeof ce)) return Error{EPROTO, "container setup: truncated error report"};
  std::string what = std::string("container setup failed at stage '") + detail::stageName(ce.stage) + "'";
  if (ce.op_index >= 0) what += " (op #" + std::to_string(ce.op_index) + ")";
  return Error{ce.err, what};
}

Result<Process> launch(const RuntimeSpec& spec, const LaunchOptions& opts) {
  if (spec.args.empty()) return Error::logic("launch: empty argv");
  if (spec.rootfs.empty()) return Error::logic("launch: rootfs not set");

  std::string root = spec.rootfs;
  if (spec.rootfs == "/") {
    if (opts.state_dir.empty()) return Error::logic("launch: host-rootfs mode requires a state_dir");
    root = fs::join(opts.state_dir, "rootfs");
    VESSEL_TRY(fs::mkdirs(root));
  } else if (!fs::isDir(root)) {
    return Error{ENOENT, "rootfs " + root + " is not a directory"};
  }

  // ---- Everything the child needs is allocated *now* ----------------------
  PlanStorage st;
  buildMountPlan(st, spec, opts, root);
  buildExec(st, spec);
  VESSEL_ASSIGN_OR_RETURN(st.seccomp, seccomp::build(spec.seccomp));

  VESSEL_ASSIGN_OR_RETURN(Pipe sync, Pipe::make());
  VESSEL_ASSIGN_OR_RETURN(Pipe err, Pipe::make());
  VESSEL_ASSIGN_OR_RETURN(Pipe out, Pipe::make());
  VESSEL_ASSIGN_OR_RETURN(Pipe errout, Pipe::make());

  ChildPlan& p = st.plan;
  p.sync_rd = sync.rd.get();
  p.err_wr = err.wr.get();
  p.stdout_wr = out.wr.get();
  p.stderr_wr = errout.wr.get();
  p.close_fds[p.n_close_fds++] = sync.wr.get();
  p.close_fds[p.n_close_fds++] = err.rd.get();
  p.close_fds[p.n_close_fds++] = out.rd.get();
  p.close_fds[p.n_close_fds++] = errout.rd.get();
  p.in_userns = true;
  p.hostname = st.s(spec.hostname.empty() ? spec.id.substr(0, 12) : spec.hostname);
  p.loopback_up = spec.network == NetworkMode::kLoopback;
  p.ops = st.ops.data();
  p.n_ops = static_cast<int>(st.ops.size());
  p.workdir = st.s(spec.working_dir.empty() ? "/" : spec.working_dir);
  p.argv = st.argv.data();
  p.envp = st.envp.data();
  p.exec_candidates = st.exec_candidates.data();
  p.n_exec_candidates = static_cast<int>(st.exec_candidates.size());
  p.use_init = spec.use_init;
  p.no_new_privs = spec.no_new_privs;
  p.drop_caps = true;
  auto keep = spec.keep_default_caps ? caps::defaultSet() : std::vector<int>{};
  keep.insert(keep.end(), spec.keep_caps.begin(), spec.keep_caps.end());
  p.caps = caps::makePlan(keep);
  if (spec.seccomp != SeccompProfile::kUnconfined) {
    p.seccomp = st.seccomp.insns.data();
    p.seccomp_len = static_cast<unsigned short>(st.seccomp.insns.size());
  }

  uint64_t flags = CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC;
  if (spec.network != NetworkMode::kHost) flags |= CLONE_NEWNET;

  int pidfd = -1;
  bool used_clone3 = false;
  int cg_fd = opts.cgroup != nullptr ? opts.cgroup->dirFd() : -1;
  pid_t pid = cloneChild(flags, cg_fd, &pidfd, &used_clone3);
  if (pid < 0) return Error::fromErrno("clone(CLONE_NEWUSER|CLONE_NEWPID|CLONE_NEWNS|...)");
  if (pid == 0) detail::childMain(p);  // never returns

  // ---- Parent ---------------------------------------------------------------
  sync.rd.reset();
  err.wr.reset();
  out.wr.reset();
  errout.wr.reset();

  Process proc;
  proc.pid = pid;
  proc.used_clone3 = used_clone3;
  proc.pidfd.reset(pidfd >= 0 ? pidfd : static_cast<int>(::syscall(SYS_pidfd_open, pid, 0)));
  if (proc.pidfd.valid()) ::fcntl(proc.pidfd.get(), F_SETFD, FD_CLOEXEC);

  auto abort_child = [&](const Error& e) -> Error {
    ::kill(pid, SIGKILL);
    ::waitpid(pid, nullptr, 0);
    return e;
  };

  if (auto s = writeIdMaps(pid, spec, opts.rootless); !s.ok()) return abort_child(s.error());
  if (opts.cgroup != nullptr && !used_clone3) {
    if (auto s = opts.cgroup->addProcess(pid); !s.ok()) return abort_child(s.error());
  }
  if (opts.before_release) {
    if (auto s = opts.before_release(pid); !s.ok()) return abort_child(s.error());
  }

  char go = 'g';
  if (::write(sync.wr.get(), &go, 1) != 1) return abort_child(Error::fromErrno("release child"));
  sync.wr.reset();

  proc.exec_status_fd = std::move(err.rd);
  proc.stdout_fd = std::move(out.rd);
  proc.stderr_fd = std::move(errout.rd);
  return proc;
}

}  // namespace vessel::runtime
