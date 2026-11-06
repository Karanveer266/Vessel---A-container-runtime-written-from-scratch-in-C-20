// SPDX-License-Identifier: MIT
// vessel-gpu-worker front-end. See worker.hpp for the contract.
#include "vessel/gpu/worker.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <exception>
#include <ostream>
#include <sstream>

#include "vessel/gpu/sim/backend.hpp"

namespace vessel::gpu {

std::string_view to_string(Op op) noexcept {
  switch (op) {
    case Op::Gemm: return "gemm";
    case Op::SplitK: return "splitk";
    case Op::Flash: return "flash";
    case Op::AllReduce: return "allreduce";
  }
  return "?";
}

std::string_view to_string(Backend b) noexcept {
  switch (b) {
    case Backend::Auto: return "auto";
    case Backend::Cuda: return "cuda";
    case Backend::Sim: return "sim";
  }
  return "?";
}

std::string usage() {
  return R"(usage: vessel-gpu-worker [options]

Runs one GPU (or simulated-GPU) benchmark and prints key=value lines on stdout.

  --op gemm|splitk|flash|allreduce   operation (default gemm)
  --backend cuda|sim|auto            auto = cuda if built with CUDA and a device
                                     is present, else sim (default auto)
  --m N --n N --k N                  GEMM / split-K shape C[m,n] = A[m,k] B[k,n]
                                     (default 1024 each)
  --ranks N                          split-K / all-reduce ranks (default 2)
  --count N                          all-reduce elements (default m*n)
  --bh N --seq N --dim N             attention batch*heads, sequence length,
                                     head dim (default 8, 512, 64; cuda: dim in
                                     {32,64,128})
  --causal                           causal attention mask
  --iters N                          timed iterations (default 10)
  --warmup N                         untimed warm-up iterations (default 1)
  --threads N                        sim backend CPU threads (default: all)
  --seed N                           input RNG seed (default 42)
  --verify                           compare against a reference implementation
  --help                             this text

Always-present output keys: ok error backend time_ms gflops bus_bw_gbps
max_abs_error devices. Exit status 0 iff ok=1. Diagnostics go to stderr.
)";
}

namespace {

bool parse_size(std::string_view s, std::size_t& out) {
  if (s.empty()) return false;
  const char* end = s.data() + s.size();
  auto [p, ec] = std::from_chars(s.data(), end, out);
  return ec == std::errc{} && p == end;
}

}  // namespace

ParseOutcome parse_args(std::span<const std::string_view> args) {
  ParseOutcome po;
  WorkerOptions& o = po.opts;
  auto fail = [&](std::string msg) {
    if (po.error.empty()) po.error = std::move(msg);
  };

  for (std::size_t i = 0; i < args.size() && po.error.empty(); ++i) {
    std::string_view a = args[i];
    std::string_view key = a;
    std::string_view val;
    bool has_inline = false;
    if (a.starts_with("--")) {
      if (auto eq = a.find('='); eq != std::string_view::npos) {
        key = a.substr(0, eq);
        val = a.substr(eq + 1);
        has_inline = true;
      }
    } else {
      fail("unexpected argument '" + std::string(a) + "'");
      break;
    }

    // Flags without a value.
    if (key == "--help" || key == "-h") { po.help = true; continue; }
    if (key == "--causal") { o.causal = true; if (has_inline) fail("--causal takes no value"); continue; }
    if (key == "--verify") { o.verify = true; if (has_inline) fail("--verify takes no value"); continue; }

    if (!has_inline) {
      if (i + 1 >= args.size()) { fail("missing value for " + std::string(key)); break; }
      val = args[++i];
    }

    if (key == "--op") {
      if (val == "gemm") o.op = Op::Gemm;
      else if (val == "splitk") o.op = Op::SplitK;
      else if (val == "flash") o.op = Op::Flash;
      else if (val == "allreduce") o.op = Op::AllReduce;
      else fail("invalid --op '" + std::string(val) + "' (expected gemm|splitk|flash|allreduce)");
      continue;
    }
    if (key == "--backend") {
      if (val == "auto") o.backend = Backend::Auto;
      else if (val == "cuda") o.backend = Backend::Cuda;
      else if (val == "sim") o.backend = Backend::Sim;
      else fail("invalid --backend '" + std::string(val) + "' (expected cuda|sim|auto)");
      continue;
    }

    struct SizeOpt { std::string_view name; std::size_t* dst; std::size_t min; };
    std::size_t seed = 0, threads = 0;
    const SizeOpt sizes[] = {
        {"--m", &o.m, 1},       {"--n", &o.n, 1},         {"--k", &o.k, 1},
        {"--ranks", &o.ranks, 1}, {"--bh", &o.bh, 1},     {"--seq", &o.seq, 1},
        {"--dim", &o.dim, 1},   {"--iters", &o.iters, 1}, {"--warmup", &o.warmup, 0},
        {"--count", &o.count, 1}, {"--seed", &seed, 0},   {"--threads", &threads, 0},
    };
    bool matched = false;
    for (const auto& so : sizes) {
      if (key != so.name) continue;
      matched = true;
      std::size_t v = 0;
      if (!parse_size(val, v)) fail("invalid value '" + std::string(val) + "' for " + std::string(key));
      else if (v < so.min) fail(std::string(key) + " must be >= " + std::to_string(so.min));
      else *so.dst = v;
      break;
    }
    if (!matched) { fail("unknown option '" + std::string(key) + "'"); break; }
    if (key == "--seed") o.seed = seed;
    if (key == "--threads") {
      if (threads > 4096) fail("--threads must be <= 4096");
      o.threads = static_cast<unsigned>(threads);
    }
  }
  if (po.error.empty() && o.ranks > 1024) po.error = "--ranks must be <= 1024";
  return po;
}

OpResult run_worker(const WorkerOptions& opts, const CudaBackend* cuda) {
  OpResult r;
  try {
    Backend b = opts.backend;
    if (b == Backend::Auto) {
      const bool gpu = cuda != nullptr && cuda->device_count != nullptr && cuda->device_count() > 0;
      b = gpu ? Backend::Cuda : Backend::Sim;
    }
    if (b == Backend::Cuda) {
      if (cuda == nullptr || cuda->run == nullptr) {
        r.backend = "cuda";
        r.error = "this binary was built without CUDA support (VESSEL_GPU_CUDA=OFF or no CUDA toolkit)";
        return r;
      }
      r = cuda->run(opts);
      r.backend = "cuda";
    } else {
      r = sim::run(opts);
      r.backend = "sim";
    }
    r.ok = r.error.empty();
  } catch (const std::exception& e) {
    r.ok = false;
    r.error = e.what();
    if (r.backend.empty()) r.backend = opts.backend == Backend::Cuda ? "cuda" : "sim";
  } catch (...) {
    r.ok = false;
    r.error = "unknown exception";
  }
  return r;
}

namespace {

/// One-line value: newlines/CR become spaces, so a malicious or verbose error
/// string cannot inject extra keys into the daemon's parser.
std::string sanitize(std::string_view v) {
  std::string s(v);
  for (char& c : s)
    if (c == '\n' || c == '\r') c = ' ';
  return s;
}

std::string sanitize_key(std::string_view k) {
  std::string s(k);
  for (char& c : s)
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') c = '_';
  return s;
}

std::string fmt(double v) {
  if (!std::isfinite(v)) return std::isnan(v) ? "nan" : (v > 0 ? "inf" : "-inf");
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.6g", v);
  return buf;
}

}  // namespace

std::vector<std::pair<std::string, std::string>> to_key_values(const OpResult& r) {
  std::string devices;
  for (std::size_t i = 0; i < r.devices.size(); ++i) {
    std::string d = r.devices[i];
    for (char& c : d)
      if (c == ',') c = ';';  // ',' is the list separator
    if (i) devices += ',';
    devices += d;
  }
  std::vector<std::pair<std::string, std::string>> kv = {
      {"ok", r.ok ? "1" : "0"},
      {"error", sanitize(r.ok ? std::string_view{} : std::string_view{r.error})},
      {"backend", r.backend.empty() ? "sim" : r.backend},
      {"time_ms", fmt(r.time_ms)},
      {"gflops", fmt(r.gflops)},
      {"bus_bw_gbps", fmt(r.bus_bw_gbps)},
      {"max_abs_error", fmt(r.max_abs_error)},
      {"devices", sanitize(devices)},
  };
  for (const auto& [k, v] : r.extra) {
    std::string key = sanitize_key(k);
    bool reserved = false;
    for (auto rk : kRequiredKeys) reserved = reserved || rk == key;
    if (!reserved && !key.empty()) kv.emplace_back(std::move(key), sanitize(v));
  }
  return kv;
}

std::string format_key_values(const OpResult& r) {
  std::string out;
  for (const auto& [k, v] : to_key_values(r)) {
    out += k;
    out += '=';
    out += v;
    out += '\n';
  }
  return out;
}

std::map<std::string, std::string> parse_key_values(std::string_view text) {
  std::map<std::string, std::string> m;
  while (!text.empty()) {
    const auto nl = text.find('\n');
    std::string_view line = text.substr(0, nl);
    text = nl == std::string_view::npos ? std::string_view{} : text.substr(nl + 1);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string_view::npos) continue;
    m.insert_or_assign(std::string(line.substr(0, eq)), std::string(line.substr(eq + 1)));
  }
  return m;
}

int worker_main(std::span<const std::string_view> args, const CudaBackend* cuda, std::ostream& out,
                std::ostream& err) {
  const ParseOutcome po = parse_args(args);
  if (po.help) {
    out << usage();
    return 0;
  }
  OpResult r;
  if (!po.error.empty()) {
    err << "vessel-gpu-worker: " << po.error << "\n(try --help)\n";
    r.ok = false;
    r.error = po.error;
    r.backend = po.opts.backend == Backend::Cuda ? "cuda" : "sim";
  } else {
    const auto& o = po.opts;
    err << "vessel-gpu-worker: op=" << to_string(o.op) << " backend=" << to_string(o.backend) << " iters=" << o.iters
        << (o.verify ? " verify" : "") << '\n';
    r = run_worker(o, cuda);
    if (!r.ok) err << "vessel-gpu-worker: error: " << r.error << '\n';
  }
  out << format_key_values(r);
  out.flush();
  return r.ok ? 0 : 1;
}

}  // namespace vessel::gpu
