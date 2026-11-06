// SPDX-License-Identifier: Apache-2.0
// vesseld — the Vessel container daemon.
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/server_credentials.h>
#include <signal.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "container_manager.h"
#include "service.h"
#include "vessel/common/fs.h"
#include "vessel/common/log.h"

namespace {

void usage() {
  std::cerr << R"(usage: vesseld [options]
  --listen ADDR          gRPC listen address (default unix:///run/vessel/vessel.sock)
                         e.g. 0.0.0.0:50051 to serve the Android console over LAN
  --token TOKEN          require "authorization: Bearer TOKEN" (or $VESSEL_TOKEN)
  --tls-cert FILE        PEM certificate (enables TLS; requires --tls-key)
  --tls-key FILE         PEM private key
  --state-dir DIR        runtime state (default /run/vessel)
  --rootfs DIR           default rootfs for containers (default: host /, read-only)
  --trace auto|ebpf|ptrace|none   default tracing backend (default auto)
  --no-ebpf              never load eBPF programs
  --cgroup-root DIR      override detected cgroup base
  --gpu-worker PATH      vessel-gpu-worker binary (default: next to vesseld)
  --log-level LEVEL      debug|info|warn|error
)";
}

std::string readFileOrDie(const std::string& p) {
  auto r = vessel::fs::readFile(p, 1 << 20);
  if (!r.ok()) {
    std::cerr << "vesseld: " << r.error().str() << "\n";
    std::exit(2);
  }
  return r.value();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace vessel;
  daemon::DaemonConfig cfg;
  std::string listen = "unix:///run/vessel/vessel.sock";
  std::string token = std::getenv("VESSEL_TOKEN") ? std::getenv("VESSEL_TOKEN") : "";
  std::string cert, key;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "vesseld: " << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--listen") listen = next();
    else if (a == "--token") token = next();
    else if (a == "--tls-cert") cert = next();
    else if (a == "--tls-key") key = next();
    else if (a == "--state-dir") cfg.state_dir = next();
    else if (a == "--rootfs") cfg.default_rootfs = next();
    else if (a == "--cgroup-root") cfg.cgroup_root = next();
    else if (a == "--gpu-worker") cfg.gpu_worker = next();
    else if (a == "--no-ebpf") cfg.enable_ebpf = false;
    else if (a == "--trace") {
      std::string t = next();
      cfg.default_trace = t == "ebpf" ? trace::Backend::kEbpf : t == "ptrace" ? trace::Backend::kPtrace
                        : t == "none" ? trace::Backend::kNone : trace::Backend::kAuto;
    } else if (a == "--log-level") {
      log::Level l;
      if (!log::parseLevel(next(), &l)) { usage(); return 2; }
      log::setLevel(l);
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::cerr << "vesseld: unknown option " << a << "\n";
      usage();
      return 2;
    }
  }
  cfg.rootless = ::geteuid() != 0;

  // Block termination signals in *all* threads (inherited by every thread
  // created after this point) and handle them synchronously below with
  // sigwait — no async signal handlers touching daemon state.
  sigset_t term;
  sigemptyset(&term);
  sigaddset(&term, SIGINT);
  sigaddset(&term, SIGTERM);
  ::pthread_sigmask(SIG_BLOCK, &term, nullptr);
  ::signal(SIGPIPE, SIG_IGN);
  // Deliberately NOT a child subreaper: every container process lives in a
  // pid namespace whose init (our mini-init) reaps orphans. As a subreaper
  // the daemon would adopt stray host-namespace orphans that no
  // __WNOTHREAD-scoped waitpid() ever collects — a zombie leak.

  if (listen.rfind("unix://", 0) == 0) {
    std::string path = listen.substr(7);
    (void)fs::mkdirs(path.substr(0, path.rfind('/')), 0755);
    ::unlink(path.c_str());
  }
  bool network = listen.rfind("unix:", 0) != 0;
  if (network && token.empty() && cert.empty())
    VLOG_WARN("vesseld", "listening on {} WITHOUT authentication or TLS — anyone on the network can run containers",
              listen);

  daemon::ContainerManager mgr(cfg);
  if (auto s = mgr.init(); !s.ok()) {
    VLOG_ERROR("vesseld", "init failed: {}", s.error().str());
    return 1;
  }
  daemon::VesselService svc(mgr, token);

  std::shared_ptr<grpc::ServerCredentials> creds;
  if (!cert.empty()) {
    grpc::SslServerCredentialsOptions o;
    o.pem_key_cert_pairs.push_back({readFileOrDie(key), readFileOrDie(cert)});
    creds = grpc::SslServerCredentials(o);
  } else {
    creds = grpc::InsecureServerCredentials();
  }
  grpc::ServerBuilder b;
  int port = 0;
  b.AddListeningPort(listen, creds, &port);
  b.RegisterService(&svc);
  // Keepalive: detect phones that vanished (Wi-Fi drop) so their streams and
  // subscriptions are reclaimed instead of lingering until TCP timeout.
  b.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 20000);
  b.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
  // Accept client pings as often as every 10 s even when no data flows (an
  // idle trace stream on a quiet container). grpc-core's default (5 min)
  // would answer the Android client's 60 s keepalive with GOAWAY
  // "too_many_pings" and force needless reconnects.
  b.AddChannelArgument(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 10000);
  b.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PING_STRIKES, 0);
  b.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  auto server = b.BuildAndStart();
  if (!server) {
    VLOG_ERROR("vesseld", "failed to listen on {}", listen);
    return 1;
  }
  VLOG_INFO("vesseld", "listening on {} (auth={}, tls={}, rootless={})", listen, token.empty() ? "off" : "bearer",
            cert.empty() ? "off" : "on", cfg.rootless);

  int sig = 0;
  sigwait(&term, &sig);
  VLOG_INFO("vesseld", "received {}, shutting down", strsignal(sig));
  svc.shutdown();
  server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(3));
  return 0;  // ~ContainerManager stops and removes every container
}
