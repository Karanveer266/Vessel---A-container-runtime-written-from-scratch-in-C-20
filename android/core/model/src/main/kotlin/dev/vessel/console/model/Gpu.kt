package dev.vessel.console.model

enum class GpuOp(val label: String, val usesAttentionShape: Boolean, val usesRanks: Boolean) {
    Unspecified("unspecified", false, false),
    Gemm("GEMM", false, false),
    GemmSplitKAllReduce("GEMM split-K + AllReduce", false, true),
    FlashAttention("FlashAttention", true, false),
    AllReduceBench("AllReduce bench", false, true),
}

enum class GpuBackend(val label: String) {
    Unspecified("default"),
    Cuda("CUDA"),
    Simulated("simulated"),
    Auto("auto"),
}

data class GpuJobRequest(
    val op: GpuOp,
    val backend: GpuBackend = GpuBackend.Auto,
    val m: Int = 0,
    val n: Int = 0,
    val k: Int = 0,
    val ranks: Int = 1,
    val batchHeads: Int = 0,
    val seqLen: Int = 0,
    val headDim: Int = 0,
    val causal: Boolean = false,
    val iterations: Int = 10,
    val verify: Boolean = true,
    val limits: ResourceLimits = ResourceLimits(),
    val rootfs: String = "",
)

data class GpuJobResult(
    val ok: Boolean,
    val error: String,
    val backend: GpuBackend,
    val timeMs: Double,
    val gflops: Double,
    val busBandwidthGbps: Double,
    val maxAbsError: Double,
    val containerId: String,
    val deviceNames: String,
    val metrics: Map<String, String>,
)
