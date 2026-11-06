// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <string>

#include "container_manager.h"
#include "vessel/v1/vessel.grpc.pb.h"

namespace vessel::daemon {

// Maps vessel::Error (errno-based) to the closest gRPC status code so
// clients can react programmatically (e.g. NOT_FOUND vs FAILED_PRECONDITION).
grpc::Status toGrpc(const Error& e);

class VesselService final : public v1::Vessel::Service {
 public:
  VesselService(ContainerManager& mgr, std::string token) : mgr_(mgr), token_(std::move(token)) {}
  void shutdown() { shutting_down_.store(true); }

  grpc::Status GetHostInfo(grpc::ServerContext*, const v1::GetHostInfoRequest*, v1::HostInfo*) override;
  grpc::Status CreateContainer(grpc::ServerContext*, const v1::CreateContainerRequest*, v1::Container*) override;
  grpc::Status StartContainer(grpc::ServerContext*, const v1::ContainerRef*, v1::Container*) override;
  grpc::Status StopContainer(grpc::ServerContext*, const v1::StopContainerRequest*, v1::Container*) override;
  grpc::Status PauseContainer(grpc::ServerContext*, const v1::ContainerRef*, v1::Container*) override;
  grpc::Status ResumeContainer(grpc::ServerContext*, const v1::ContainerRef*, v1::Container*) override;
  grpc::Status RemoveContainer(grpc::ServerContext*, const v1::RemoveContainerRequest*,
                               v1::RemoveContainerResponse*) override;
  grpc::Status GetContainer(grpc::ServerContext*, const v1::ContainerRef*, v1::Container*) override;
  grpc::Status ListContainers(grpc::ServerContext*, const v1::ListContainersRequest*,
                              v1::ListContainersResponse*) override;
  grpc::Status Exec(grpc::ServerContext*, const v1::ExecRequest*, v1::ExecResponse*) override;
  grpc::Status StreamStats(grpc::ServerContext*, const v1::StreamStatsRequest*,
                           grpc::ServerWriter<v1::StatsSample>*) override;
  grpc::Status StreamTrace(grpc::ServerContext*, const v1::StreamTraceRequest*,
                           grpc::ServerWriter<v1::TraceEvent>*) override;
  grpc::Status StreamLogs(grpc::ServerContext*, const v1::StreamLogsRequest*,
                          grpc::ServerWriter<v1::LogChunk>*) override;
  grpc::Status GetSyscallSummary(grpc::ServerContext*, const v1::ContainerRef*, v1::SyscallSummary*) override;
  grpc::Status RunGpuJob(grpc::ServerContext*, const v1::GpuJobRequest*, v1::GpuJobResult*) override;

 private:
  grpc::Status authorize(grpc::ServerContext* ctx) const;
  ContainerManager& mgr_;
  std::string token_;
  std::atomic<bool> shutting_down_{false};
};

}  // namespace vessel::daemon
