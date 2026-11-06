package dev.vessel.console.api

import com.google.protobuf.ByteString
import dev.vessel.api.v1.CgroupVersion as PbCgroupVersion
import dev.vessel.api.v1.ContainerState as PbContainerState
import dev.vessel.api.v1.GpuBackend as PbGpuBackend
import dev.vessel.api.v1.TraceBackend as PbTraceBackend
import dev.vessel.api.v1.TraceEventKind as PbTraceEventKind
import dev.vessel.api.v1.container
import dev.vessel.api.v1.containerSpec
import dev.vessel.api.v1.execResponse
import dev.vessel.api.v1.gpuDevice
import dev.vessel.api.v1.gpuJobResult
import dev.vessel.api.v1.hostInfo
import dev.vessel.api.v1.psi
import dev.vessel.api.v1.resourceLimits
import dev.vessel.api.v1.statsSample
import dev.vessel.api.v1.syscallCount
import dev.vessel.api.v1.syscallSummary
import dev.vessel.api.v1.traceEvent
import dev.vessel.console.model.CgroupVersion
import dev.vessel.console.model.ContainerSpec
import dev.vessel.console.model.ContainerState
import dev.vessel.console.model.GpuBackend
import dev.vessel.console.model.GpuJobRequest
import dev.vessel.console.model.GpuOp
import dev.vessel.console.model.Mount
import dev.vessel.console.model.NetworkMode
import dev.vessel.console.model.Psi
import dev.vessel.console.model.ResourceLimits
import dev.vessel.console.model.SeccompProfile
import dev.vessel.console.model.TraceBackend
import dev.vessel.console.model.TraceEventKind
import dev.vessel.console.model.TraceFilter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MappersTest {

    @Test
    fun `host info maps every field`() {
        val pb = hostInfo {
            kernelRelease = "6.8.0-vessel"
            machine = "aarch64"
            isAndroid = true
            cgroupVersion = PbCgroupVersion.CGROUP_VERSION_V2
            cgroupControllers += listOf("cpu", "memory", "pids")
            ebpfAvailable = true
            rootless = false
            psiAvailable = true
            gpus += gpuDevice {
                index = 0; name = "RTX 4090"; totalMemoryBytes = 24L shl 30; smMajor = 8; smMinor = 9
            }
            daemonVersion = "0.3.0"
            onlineCpus = 8
            totalMemoryBytes = 16L shl 30
            defaultTraceBackend = PbTraceBackend.TRACE_BACKEND_EBPF
        }
        val d = pb.toDomain()
        assertEquals("6.8.0-vessel", d.kernelRelease)
        assertTrue(d.isAndroid)
        assertEquals(CgroupVersion.V2, d.cgroupVersion)
        assertEquals(listOf("cpu", "memory", "pids"), d.cgroupControllers)
        assertEquals("sm_89", d.gpus.single().computeCapability)
        assertEquals(24L shl 30, d.gpus.single().totalMemoryBytes)
        assertEquals(8, d.onlineCpus)
        assertEquals(TraceBackend.Ebpf, d.defaultTraceBackend)
    }

    @Test
    fun `unknown enum values from a newer daemon degrade to Unspecified`() {
        // Simulates a daemon sending enum numbers this client was not compiled with.
        val c = container { id = "abc"; stateValue = 42 }.toDomain()
        assertEquals(ContainerState.Unspecified, c.state)
        assertEquals(CgroupVersion.Unspecified, hostInfo { cgroupVersionValue = 99 }.toDomain().cgroupVersion)
        assertEquals(TraceEventKind.Unspecified, traceEvent { kindValue = 1234 }.toDomain(0).kind)
    }

    @Test
    fun `container spec round-trips domain to proto and back`() {
        val spec = ContainerSpec(
            name = "web",
            rootfs = "/var/lib/vessel/alpine",
            args = listOf("/bin/sh", "-c", "sleep 1000"),
            env = listOf("A=1", "B=2"),
            hostname = "web-1",
            workingDir = "/srv",
            limits = ResourceLimits(500, 100, 256L shl 20, 200L shl 20, 64, "0-1", true),
            network = NetworkMode.Loopback,
            seccomp = SeccompProfile.Strict,
            trace = TraceBackend.Ptrace,
            mounts = listOf(Mount("/host/data", "/data", readOnly = true), Mount("/tmp/x", "/x")),
            readOnlyRootfs = true,
            gpuDevices = listOf(0, 1),
            autoRemove = true,
        )
        assertEquals(spec, spec.toProto().toDomain())
    }

    @Test
    fun `create request carries advanced spec fields on the wire`() {
        val spec = ContainerSpec(
            name = "adv",
            rootfs = "/r",
            args = listOf("true"),
            workingDir = "/srv",
            readOnlyRootfs = true,
            limits = ResourceLimits(cpuWeight = 250, disableSwap = true),
            mounts = listOf(Mount("/host/data", "/data", readOnly = true), Mount("/tmp/x", "/x")),
        )
        val pb = spec.toProto()
        assertEquals("/srv", pb.workingDir)
        assertTrue(pb.readOnlyRootfs)
        assertEquals(250, pb.limits.cpuWeight)
        assertTrue(pb.limits.disableSwap)
        assertEquals(2, pb.mountsCount)
        assertEquals("/host/data", pb.getMounts(0).source)
        assertEquals("/data", pb.getMounts(0).target)
        assertTrue(pb.getMounts(0).readOnly)
        assertFalse(pb.getMounts(1).readOnly)
    }

    @Test
    fun `container maps state, timestamps and missing spec`() {
        val c = container {
            id = "0123456789ab"
            state = PbContainerState.CONTAINER_STATE_STOPPED
            exitCode = 137
            oomKilled = true
            createdAtUnixNs = 1
            startedAtUnixNs = 2
            finishedAtUnixNs = 3
            activeTraceBackend = PbTraceBackend.TRACE_BACKEND_PTRACE
        }.toDomain()
        assertEquals(ContainerState.Stopped, c.state)
        assertEquals(137, c.exitCode)
        assertTrue(c.oomKilled)
        assertEquals(3L, c.finishedAtUnixNs)
        assertEquals("0123456789ab", c.displayName) // no spec name -> id
        assertEquals(ResourceLimits(), c.spec.limits)
        assertEquals(c.copy(spec = c.spec.copy(name = "db")).displayName, "db")
    }

    @Test
    fun `stats sample maps counters and absent PSI to zero`() {
        val s = statsSample {
            containerId = "c"
            cpuMillicoresUsed = 750
            cpuNrThrottled = 12
            memoryCurrentBytes = 100
            memoryMaxBytes = 200
            memoryEventsOomKill = 1
            pidsCurrent = 5
            memoryPressure = psi { someAvg10 = 3.5f; someTotalUsec = 99 }
            droppedSamples = 2
        }.toDomain()
        assertEquals(750, s.cpuMillicoresUsed)
        assertEquals(12L, s.cpuNrThrottled)
        assertEquals(Psi(someAvg10 = 3.5f, someTotalUsec = 99), s.memoryPressure)
        assertEquals(Psi.Zero, s.cpuPressure)
        assertTrue(s.hasOomEvents)
        assertEquals(2L, s.droppedSamples)
    }

    @Test
    fun `trace event carries client sequence and errno return`() {
        val e = traceEvent {
            kind = PbTraceEventKind.TRACE_EVENT_KIND_SYSCALL
            syscallName = "openat"
            ret = -2
            latencyNs = 12_000
            droppedEvents = 7
            backend = PbTraceBackend.TRACE_BACKEND_EBPF
        }.toDomain(seq = 41)
        assertEquals(41L, e.seq)
        assertTrue(e.isError)
        assertEquals(7L, e.droppedEvents)
        assertEquals(TraceBackend.Ebpf, e.backend)
    }

    @Test
    fun `syscall summary is sorted by count even if the daemon is not`() {
        val s = syscallSummary {
            syscalls += syscallCount { name = "read"; count = 10; totalLatencyNs = 1_000 }
            syscalls += syscallCount { name = "write"; count = 50 }
            totalEvents = 60
        }.toDomain()
        assertEquals(listOf("write", "read"), s.syscalls.map { it.name })
        assertEquals(100L, s.syscalls[1].avgLatencyNs)
    }

    @Test
    fun `exec output tolerates invalid utf-8`() {
        val r = execResponse {
            exitCode = 1
            stdout = ByteString.copyFromUtf8("hi\n")
            stderr = ByteString.copyFrom(byteArrayOf(0xff.toByte(), 'x'.code.toByte()))
        }.toDomain()
        assertEquals("hi\n", r.stdout)
        assertTrue(r.stderr.endsWith("x"))
        assertFalse(r.timedOut)
    }

    @Test
    fun `trace filter and gpu request map to wire requests`() {
        val req = TraceFilter("c1", errorsOnly = true, minLatencyNs = 5, syscalls = listOf("read"), kinds = setOf(TraceEventKind.Net))
            .toProto()
        assertEquals("c1", req.id)
        assertTrue(req.errorsOnly)
        assertEquals(5L, req.minLatencyNs)
        assertEquals(listOf("read"), req.syscallsList)
        assertEquals(listOf(PbTraceEventKind.TRACE_EVENT_KIND_NET), req.kindsList)

        val gpu = GpuJobRequest(op = GpuOp.FlashAttention, backend = GpuBackend.Simulated, batchHeads = 8, seqLen = 1024, headDim = 64, causal = true)
            .toProto()
        assertEquals(PbGpuBackend.GPU_BACKEND_SIMULATED, gpu.backend)
        assertEquals(1024, gpu.seqLen)
        assertTrue(gpu.causal)
        assertTrue(gpu.verify)
    }

    @Test
    fun `gpu result metrics are sorted for stable rendering`() {
        val r = gpuJobResult {
            ok = true
            gflops = 1234.5
            metrics.put("z", "1")
            metrics.put("a", "2")
        }.toDomain()
        assertEquals(listOf("a", "z"), r.metrics.keys.toList())
        assertEquals(1234.5, r.gflops, 0.0)
    }

    @Test
    fun `limits zero means unlimited on the wire`() {
        val pb = resourceLimits { }
        assertEquals(ResourceLimits(), pb.toDomain())
        assertEquals(0, ResourceLimits().toProto().cpuMillicores)
        assertEquals(containerSpec { }.toDomain().args, emptyList<String>())
    }
}
