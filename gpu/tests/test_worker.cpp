// SPDX-License-Identifier: MIT
// vessel-gpu-worker output contract (the daemon parses this).
#include <gtest/gtest.h>

#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "vessel/gpu/worker.hpp"

using namespace vessel::gpu;

namespace {

struct Run {
  int code;
  std::string out, err;
  std::map<std::string, std::string> kv;
};

Run run(std::vector<std::string_view> args, const CudaBackend* cuda = nullptr) {
  std::ostringstream out, err;
  Run r;
  r.code = worker_main(args, cuda, out, err);
  r.out = out.str();
  r.err = err.str();
  r.kv = parse_key_values(r.out);
  return r;
}

/// Asserts the textual contract: one key=value per line, no spaces around
/// '=', every required key present, required keys first and in order.
void expect_contract(const std::string& out) {
  std::istringstream ss(out);
  std::string line;
  std::vector<std::string> keys;
  while (std::getline(ss, line)) {
    ASSERT_FALSE(line.empty());
    const auto eq = line.find('=');
    ASSERT_NE(eq, std::string::npos) << line;
    ASSERT_GT(eq, 0U) << line;
    EXPECT_NE(line[eq - 1], ' ') << line;
    if (eq + 1 < line.size()) {
      EXPECT_NE(line[eq + 1], ' ') << line;
    }
    keys.push_back(line.substr(0, eq));
  }
  ASSERT_GE(keys.size(), std::size(kRequiredKeys));
  for (std::size_t i = 0; i < std::size(kRequiredKeys); ++i) EXPECT_EQ(keys[i], kRequiredKeys[i]);
  EXPECT_EQ(std::set<std::string>(keys.begin(), keys.end()).size(), keys.size()) << "duplicate keys";
}

double num(const Run& r, const std::string& k) { return std::strtod(r.kv.at(k).c_str(), nullptr); }

}  // namespace

TEST(WorkerArgs, Defaults) {
  const auto po = parse_args({});
  ASSERT_TRUE(po.error.empty());
  EXPECT_EQ(po.opts.op, Op::Gemm);
  EXPECT_EQ(po.opts.backend, Backend::Auto);
  EXPECT_EQ(po.opts.m, 1024U);
  EXPECT_EQ(po.opts.ranks, 2U);
  EXPECT_EQ(po.opts.bh, 8U);
  EXPECT_EQ(po.opts.seq, 512U);
  EXPECT_EQ(po.opts.dim, 64U);
  EXPECT_EQ(po.opts.iters, 10U);
  EXPECT_FALSE(po.opts.causal);
  EXPECT_FALSE(po.opts.verify);
}

TEST(WorkerArgs, ParsesEverything) {
  std::vector<std::string_view> a = {"--op", "flash", "--backend=sim", "--m", "3", "--n", "4", "--k", "5",
                                     "--ranks", "7", "--bh", "2", "--seq=9", "--dim", "32", "--causal",
                                     "--iters", "3", "--verify", "--threads", "2", "--count", "11"};
  const auto po = parse_args(a);
  ASSERT_TRUE(po.error.empty()) << po.error;
  const auto& o = po.opts;
  EXPECT_EQ(o.op, Op::Flash);
  EXPECT_EQ(o.backend, Backend::Sim);
  EXPECT_EQ(o.m * 100 + o.n * 10 + o.k, 345U);
  EXPECT_EQ(o.ranks, 7U);
  EXPECT_EQ(o.seq, 9U);
  EXPECT_TRUE(o.causal);
  EXPECT_TRUE(o.verify);
  EXPECT_EQ(o.threads, 2U);
  EXPECT_EQ(o.allreduce_count(), 11U);
}

TEST(WorkerArgs, RejectsGarbage) {
  for (std::vector<std::string_view> a : std::vector<std::vector<std::string_view>>{
           {"--op", "conv"}, {"--backend", "tpu"}, {"--m", "-1"}, {"--m", "0"}, {"--m", "12x"},
           {"--iters"}, {"--frobnicate", "1"}, {"positional"}, {"--causal=1"}, {"--ranks", "99999"}}) {
    EXPECT_FALSE(parse_args(a).error.empty()) << a[0];
  }
}

TEST(WorkerContract, HelpPrintsUsageAndExitsZero) {
  const auto r = run({"--help"});
  EXPECT_EQ(r.code, 0);
  EXPECT_NE(r.out.find("--op gemm|splitk|flash|allreduce"), std::string::npos);
}

TEST(WorkerContract, BadArgsStillEmitContract) {
  const auto r = run({"--op", "nope"});
  EXPECT_EQ(r.code, 1);
  expect_contract(r.out);
  EXPECT_EQ(r.kv.at("ok"), "0");
  EXPECT_FALSE(r.kv.at("error").empty());
  EXPECT_FALSE(r.err.empty());  // diagnostics on stderr
}

TEST(WorkerContract, CudaRequestedWithoutCudaFailsCleanly) {
  const auto r = run({"--backend", "cuda", "--op", "gemm", "--m", "8", "--n", "8", "--k", "8"});
  EXPECT_EQ(r.code, 1);
  expect_contract(r.out);
  EXPECT_EQ(r.kv.at("ok"), "0");
  EXPECT_EQ(r.kv.at("backend"), "cuda");
  EXPECT_NE(r.kv.at("error").find("CUDA"), std::string::npos);
}

TEST(WorkerContract, AutoSelectsSimWhenNoDevice) {
  static const CudaBackend no_gpu{[] { return 0; }, [](const WorkerOptions&) {
                                    OpResult r;
                                    r.error = "must not be called";
                                    return r;
                                  }};
  const auto r = run({"--op", "gemm", "--m", "8", "--n", "8", "--k", "8", "--iters", "1"}, &no_gpu);
  EXPECT_EQ(r.code, 0);
  EXPECT_EQ(r.kv.at("backend"), "sim");
}

TEST(WorkerContract, AutoSelectsCudaWhenDevicePresent) {
  static const CudaBackend fake{[] { return 2; }, [](const WorkerOptions&) {
                                  OpResult r;
                                  r.time_ms = 1.5;
                                  r.devices = {"NVIDIA H100 80GB HBM3", "Fake, GPU"};
                                  r.add("ok", std::string("0"));  // reserved key: must be dropped
                                  r.add("tile", std::string("128x128x8"));
                                  return r;
                                }};
  const auto r = run({"--op", "gemm"}, &fake);
  EXPECT_EQ(r.code, 0);
  expect_contract(r.out);
  EXPECT_EQ(r.kv.at("ok"), "1");
  EXPECT_EQ(r.kv.at("backend"), "cuda");
  EXPECT_EQ(r.kv.at("devices"), "NVIDIA H100 80GB HBM3,Fake; GPU");
  EXPECT_EQ(r.kv.at("tile"), "128x128x8");
}

TEST(WorkerContract, ErrorStringsCannotInjectKeys) {
  OpResult res;
  res.error = "bad\nok=1";
  res.add("weird key=x", std::string("v\nw"));
  const auto text = format_key_values(res);
  expect_contract(text);
  const auto kv = parse_key_values(text);
  EXPECT_EQ(kv.at("ok"), "0");
  EXPECT_EQ(kv.at("error"), "bad ok=1");
  EXPECT_EQ(kv.at("weird_key_x"), "v w");
}

struct OpCase {
  std::vector<std::string_view> args;
};

class WorkerOps : public ::testing::TestWithParam<OpCase> {};

TEST_P(WorkerOps, SimRunsAndVerifies) {
  auto args = GetParam().args;
  for (std::string_view a : {"--backend", "sim", "--iters", "2", "--verify", "--threads", "2"}) args.push_back(a);
  const auto r = run(args);
  expect_contract(r.out);
  ASSERT_EQ(r.kv.at("ok"), "1") << r.out << r.err;
  EXPECT_EQ(r.code, 0);
  EXPECT_EQ(r.kv.at("error"), "");
  EXPECT_EQ(r.kv.at("backend"), "sim");
  EXPECT_GT(num(r, "time_ms"), 0.0);
  EXPECT_LE(num(r, "max_abs_error"), num(r, "tolerance"));
  EXPECT_EQ(r.kv.at("devices").rfind("cpu-sim-rank0", 0), 0U);
}

INSTANTIATE_TEST_SUITE_P(
    AllOps, WorkerOps,
    ::testing::Values(OpCase{{"--op", "gemm", "--m", "67", "--n", "45", "--k", "33"}},
                      OpCase{{"--op", "splitk", "--m", "40", "--n", "33", "--k", "77", "--ranks", "3"}},
                      OpCase{{"--op", "flash", "--bh", "2", "--seq", "45", "--dim", "32"}},
                      OpCase{{"--op", "flash", "--bh", "2", "--seq", "45", "--dim", "64", "--causal"}},
                      OpCase{{"--op", "allreduce", "--count", "1001", "--ranks", "4"}}));

TEST(WorkerContract, SplitKReportsRingMetrics) {
  const auto r = run({"--op", "splitk", "--backend", "sim", "--m", "32", "--n", "32", "--k", "64", "--ranks", "4",
                      "--iters", "1"});
  ASSERT_EQ(r.kv.at("ok"), "1");
  EXPECT_EQ(r.kv.at("devices"), "cpu-sim-rank0,cpu-sim-rank1,cpu-sim-rank2,cpu-sim-rank3");
  // 32*32 floats, R=4 divides it: exactly 2*3/4 * 4096 B = 6144 B per rank.
  EXPECT_EQ(r.kv.at("bytes_per_rank"), "6144");
  EXPECT_GT(num(r, "bus_bw_gbps"), 0.0);
}

TEST(WorkerContract, AllReduceBusBwFormula) {
  const auto r = run({"--op", "allreduce", "--backend", "sim", "--count", "40000", "--ranks", "4", "--iters", "2"});
  ASSERT_EQ(r.kv.at("ok"), "1");
  EXPECT_NEAR(num(r, "bus_bw_gbps"), num(r, "algbw_gbps") * 1.5, 1e-3 * num(r, "bus_bw_gbps") + 1e-9);
  EXPECT_EQ(num(r, "bytes_per_rank"), num(r, "bytes_per_rank_theory"));
}
