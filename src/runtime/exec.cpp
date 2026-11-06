// SPDX-License-Identifier: Apache-2.0
// execIn(): run an extra process inside a live container via setns(2).
//
//   daemon ──fork──► helper ──setns(user,mnt,uts,ipc,net,cgroup,pid)──┐
//                       │   (setns(pid) only affects *children*)      │
//                       └──fork──► payload: caps/nnp/seccomp + execve ◄┘
//
// The helper is moved into the container's cgroup by the parent before it
// forks, so the payload is born accounted and limited. The helper then just
// waits and relays the exit status.
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <deque>

#include "child.h"
#include "vessel/common/clock.h"
#include "vessel/common/fs.h"
#include "vessel/common/strings.h"
#include "vessel/runtime/launcher.h"
#include "vessel/runtime/seccomp.h"

namespace vessel::runtime {
namespace {

struct NsFd {
  int type;
  const char* name;
  UniqueFd fd;
};


}  // namespace

Result<ExecResult> execIn(pid_t init_pid, const RuntimeSpec& spec, cgroup::Cgroup* cg,
                          const std::vector<std::string>& args, const std::vector<std::string>& env,
                          int timeout_ms) {
  if (args.empty()) return Error::logic("exec: empty argv");
  const std::string base = "/proc/" + std::to_string(init_pid) + "/ns/";
  // User namespace first: joining it grants the capabilities needed to join
  // the namespaces it owns. mnt before pid is irrelevant; pid is applied last.
  std::vector<NsFd> ns;
  for (auto [type, name] : {std::pair{CLONE_NEWUSER, "user"}, {CLONE_NEWNS, "mnt"}, {CLONE_NEWUTS, "uts"},
                            {CLONE_NEWIPC, "ipc"}, {CLONE_NEWNET, "net"}, {CLONE_NEWCGROUP, "cgroup"},
                            {CLONE_NEWPID, "pid"}}) {
    UniqueFd fd(::open((base + name).c_str(), O_RDONLY | O_CLOEXEC));
    if (!fd) {
      if (errno == ENOENT) continue;  // kernel without that ns type
      return Error::fromErrno("open " + base + name);
    }
    ns.push_back({type, name, std::move(fd)});
  }

  // Prepare a ChildPlan just like launch(), minus the mount plan.
  std::deque<std::string> strs;
  auto s = [&](std::string v) { return strs.emplace_back(std::move(v)).c_str(); };
  std::vector<char*> argv, envp;
  std::vector<const char*> cands;
  for (const auto& a : args) argv.push_back(const_cast<char*>(s(a)));
  argv.push_back(nullptr);
  std::string path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
  bool has_path = false;
  for (const auto* list : {&spec.env, &env})
    for (const auto& e : *list) {
      if (str::startsWith(e, "PATH=")) { has_path = true; path = e.substr(5); }
      envp.push_back(const_cast<char*>(s(e)));
    }
  if (!has_path) envp.push_back(const_cast<char*>(s("PATH=" + path)));
  envp.push_back(nullptr);
  if (args[0].find('/') != std::string::npos) cands.push_back(s(args[0]));
  else
    for (auto d : str::split(path, ':')) cands.push_back(s(fs::join(d, args[0])));

  VESSEL_ASSIGN_OR_RETURN(auto prog, seccomp::build(spec.seccomp));
  auto keep = spec.keep_default_caps ? caps::defaultSet() : std::vector<int>{};
  keep.insert(keep.end(), spec.keep_caps.begin(), spec.keep_caps.end());

  int sync[2], errp[2], outp[2], errout[2];
  if (::pipe2(sync, O_CLOEXEC) || ::pipe2(errp, O_CLOEXEC) || ::pipe2(outp, O_CLOEXEC) || ::pipe2(errout, O_CLOEXEC))
    return Error::fromErrno("pipe2");
  UniqueFd sync_rd(sync[0]), sync_wr(sync[1]), err_rd(errp[0]), err_wr(errp[1]);
  UniqueFd out_rd(outp[0]), out_wr(outp[1]), eo_rd(errout[0]), eo_wr(errout[1]);

  detail::ChildPlan plan;
  plan.err_wr = err_wr.get();
  plan.workdir = s(spec.working_dir.empty() ? "/" : spec.working_dir);
  plan.argv = argv.data();
  plan.envp = envp.data();
  plan.exec_candidates = cands.data();
  plan.n_exec_candidates = static_cast<int>(cands.size());
  plan.use_init = false;
  plan.no_new_privs = spec.no_new_privs;
  plan.drop_caps = true;
  plan.caps = caps::makePlan(keep);
  if (spec.seccomp != SeccompProfile::kUnconfined) {
    plan.seccomp = prog.insns.data();
    plan.seccomp_len = static_cast<unsigned short>(prog.insns.size());
  }
  std::vector<int> nsfds;
  std::vector<int> nstypes;
  for (auto& n : ns) { nsfds.push_back(n.fd.get()); nstypes.push_back(n.type); }

  pid_t helper = ::fork();
  if (helper < 0) return Error::fromErrno("fork exec helper");
  if (helper == 0) {
    // --- helper (async-signal-safe only) ---
    ::close(sync_wr.get());
    ::close(err_rd.get());
    ::close(out_rd.get());
    ::close(eo_rd.get());
    ::prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0);
    char go;
    if (::read(sync_rd.get(), &go, 1) != 1) detail::childFail(plan, detail::Stage::kSyncWait, EPIPE);
    for (size_t i = 0; i < nsfds.size(); ++i) {
      if (::setns(nsfds[i], nstypes[i]) != 0) {
        // Re-joining our *own* user ns is EINVAL (e.g. host-user mode) — fine.
        if (!(nstypes[i] == CLONE_NEWUSER && errno == EINVAL)) detail::childFail(plan, detail::Stage::kSetId, errno, static_cast<int>(i));
      }
    }
    if (::setresgid(0, 0, 0) != 0 || ::setresuid(0, 0, 0) != 0) detail::childFail(plan, detail::Stage::kSetId, errno);
    pid_t payload = ::fork();
    if (payload < 0) detail::childFail(plan, detail::Stage::kInitFork, errno);
    if (payload == 0) {
      int devnull = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
      ::dup2(devnull, 0);
      ::dup2(out_wr.get(), 1);
      ::dup2(eo_wr.get(), 2);
      detail::finalizeAndExec(plan);
    }
    ::close(err_wr.get());
    ::close(out_wr.get());
    ::close(eo_wr.get());
    int st = 0;
    while (::waitpid(payload, &st, 0) < 0 && errno == EINTR) {
    }
    ::_exit(exitCodeFromWaitStatus(st) & 0xFF);
  }

  sync_rd.reset();
  err_wr.reset();
  out_wr.reset();
  eo_wr.reset();
  auto fail = [&](Error e) -> Error {
    ::kill(helper, SIGKILL);
    ::waitpid(helper, nullptr, 0);
    return e;
  };
  if (cg != nullptr)
    if (auto st = cg->addProcess(helper); !st.ok()) return fail(st.error());
  if (::write(sync_wr.get(), "g", 1) != 1) return fail(Error::fromErrno("release exec helper"));
  sync_wr.reset();

  ExecResult res;
  constexpr size_t kCap = 4 << 20;  // bound memory a chatty command can make us hold
  const int64_t deadline = clk::monotonicNs() + static_cast<int64_t>(timeout_ms > 0 ? timeout_ms : 30000) * 1'000'000;
  pollfd pfds[3] = {{out_rd.get(), POLLIN, 0}, {eo_rd.get(), POLLIN, 0}, {err_rd.get(), POLLIN, 0}};
  int open_fds = 3;
  detail::ChildError ce{};
  bool setup_failed = false;
  while (open_fds > 0) {
    int64_t left_ms = (deadline - clk::monotonicNs()) / 1'000'000;
    if (left_ms <= 0) {
      res.timed_out = true;
      ::kill(helper, SIGKILL);
      break;
    }
    int r = ::poll(pfds, 3, static_cast<int>(std::min<int64_t>(left_ms, 1000)));
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) return fail(Error::fromErrno("poll exec output"));
    for (int i = 0; i < 3; ++i) {
      if (pfds[i].fd < 0 || !(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      if (i == 2) {
        ssize_t n = ::read(pfds[i].fd, &ce, sizeof ce);
        if (n == static_cast<ssize_t>(sizeof ce)) setup_failed = true;
        pfds[i].fd = -1;
        --open_fds;
        continue;
      }
      char buf[8192];
      ssize_t n = ::read(pfds[i].fd, buf, sizeof buf);
      if (n <= 0) {
        pfds[i].fd = -1;
        --open_fds;
        continue;
      }
      std::string& into = i == 0 ? res.out : res.err;
      if (into.size() < kCap) into.append(buf, std::min(static_cast<size_t>(n), kCap - into.size()));
    }
  }
  int st = 0;
  while (::waitpid(helper, &st, 0) < 0 && errno == EINTR) {
  }
  if (setup_failed)
    return Error{ce.err, std::string("exec setup failed at stage '") + detail::stageName(ce.stage) + "'"};
  res.exit_code = res.timed_out ? 137 : exitCodeFromWaitStatus(st);
  return res;
}

}  // namespace vessel::runtime
