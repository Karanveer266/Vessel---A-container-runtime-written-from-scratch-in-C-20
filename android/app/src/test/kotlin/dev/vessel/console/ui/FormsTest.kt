package dev.vessel.console.ui

import dev.vessel.console.model.GpuOp
import dev.vessel.console.model.Mount
import dev.vessel.console.model.NetworkMode
import dev.vessel.console.model.SeccompProfile
import dev.vessel.console.testing.FakeVesselRepository
import dev.vessel.console.testing.MainDispatcherRule
import dev.vessel.console.ui.create.CreateContainerViewModel
import dev.vessel.console.ui.create.CreateForm
import dev.vessel.console.ui.create.CreateFormValidator
import dev.vessel.console.ui.create.Field
import dev.vessel.console.ui.create.MountRow
import dev.vessel.console.ui.create.Validation
import dev.vessel.console.ui.gpu.GpuForm
import dev.vessel.console.ui.gpu.GpuJobViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class CreateFormValidatorTest {
    private val valid = CreateForm(
        name = "web",
        rootfs = "/var/lib/vessel/alpine",
        command = "/bin/sh -c 'echo hi'",
        env = "A=1\n\n  B=two words \n",
        cpu = "500m",
        memoryMax = "256Mi",
        memoryHigh = "200Mi",
        pidsMax = "64",
        cpuset = "0-1",
        network = NetworkMode.None,
        seccomp = SeccompProfile.Strict,
        autoRemove = true,
    )

    @Test
    fun `valid form maps to a full spec`() {
        val spec = (CreateFormValidator.validate(valid) as Validation.Valid).spec
        assertEquals(listOf("/bin/sh", "-c", "echo hi"), spec.args)
        assertEquals(listOf("A=1", "B=two words"), spec.env)
        assertEquals(500, spec.limits.cpuMillicores)
        assertEquals(256L shl 20, spec.limits.memoryMaxBytes)
        assertEquals(200L shl 20, spec.limits.memoryHighBytes)
        assertEquals(64, spec.limits.pidsMax)
        assertEquals("0-1", spec.limits.cpusetCpus)
        assertEquals(NetworkMode.None, spec.network)
        assertTrue(spec.autoRemove)
    }

    @Test
    fun `empty limits mean unlimited`() {
        val spec = (CreateFormValidator.validate(valid.copy(cpu = "", memoryMax = "", memoryHigh = "", pidsMax = "")) as Validation.Valid).spec
        assertEquals(0, spec.limits.cpuMillicores)
        assertEquals(0L, spec.limits.memoryMaxBytes)
    }

    @Test
    fun `reports every invalid field`() {
        val errors = (
            CreateFormValidator.validate(
                CreateForm(
                    name = "bad name!", rootfs = "relative", command = "'", env = "NOEQUALS", hostname = "-x",
                    cpu = "lots", memoryMax = "1Gi", memoryHigh = "2Gi", pidsMax = "-1", cpuset = "3-1",
                ),
            ) as Validation.Invalid
            ).errors
        assertEquals(
            setOf(Field.Name, Field.Rootfs, Field.Command, Field.Env, Field.Hostname, Field.Cpu, Field.MemoryHigh, Field.PidsMax, Field.Cpuset),
            errors.keys,
        )
    }
}

class CreateFormAdvancedValidatorTest {
    private val base = CreateForm(name = "web", rootfs = "/r", command = "sleep 1")

    private fun valid(f: CreateForm) = (CreateFormValidator.validate(f) as Validation.Valid).spec
    private fun invalid(f: CreateForm) = CreateFormValidator.validate(f) as Validation.Invalid

    @Test
    fun `advanced fields map into the spec`() {
        val spec = valid(
            base.copy(
                workingDir = " /srv ",
                readOnlyRootfs = true,
                cpuWeight = "250",
                disableSwap = true,
                mounts = listOf(
                    MountRow(1, "/host/data", "/data/", readOnly = true),
                    MountRow(2, "", ""), // blank row ignored
                    MountRow(3, "/tmp/x", "/x"),
                ),
            ),
        )
        assertEquals("/srv", spec.workingDir)
        assertTrue(spec.readOnlyRootfs)
        assertEquals(250, spec.limits.cpuWeight)
        assertTrue(spec.limits.disableSwap)
        // Trailing slash normalized away.
        assertEquals(listOf(Mount("/host/data", "/data", true), Mount("/tmp/x", "/x", false)), spec.mounts)
    }

    @Test
    fun `defaults leave advanced fields unset`() {
        val spec = valid(base)
        assertEquals("", spec.workingDir)
        assertEquals(0, spec.limits.cpuWeight)
        assertEquals(emptyList<Mount>(), spec.mounts)
    }

    @Test
    fun `cpu weight must be within 1 to 10000`() {
        assertEquals(1, valid(base.copy(cpuWeight = "1")).limits.cpuWeight)
        assertEquals(10_000, valid(base.copy(cpuWeight = "10000")).limits.cpuWeight)
        listOf("0", "10001", "-5", "abc").forEach {
            assertTrue(it, Field.CpuWeight in invalid(base.copy(cpuWeight = it)).errors)
        }
    }

    @Test
    fun `working dir must be absolute`() {
        assertEquals("Must be an absolute path inside the container", invalid(base.copy(workingDir = "srv")).errors[Field.WorkingDir])
    }

    @Test
    fun `mount target must be absolute and sane`() {
        val e = invalid(
            base.copy(
                mounts = listOf(
                    MountRow(1, "/a", "data"),
                    MountRow(2, "/b", "/"),
                    MountRow(3, "/c", "/x/../etc"),
                    MountRow(4, "/d", ""),
                ),
            ),
        ).mountErrors
        assertEquals("Must start with '/'", e[1]?.target)
        assertEquals("Cannot mount over the root", e[2]?.target)
        assertEquals("Must not contain '..'", e[3]?.target)
        assertEquals("Required", e[4]?.target)
    }

    @Test
    fun `mount source must be an absolute host path`() {
        val e = invalid(base.copy(mounts = listOf(MountRow(7, "relative", "/x"), MountRow(8, "", "/y")))).mountErrors
        assertEquals("Must be an absolute host path", e[7]?.source)
        assertNull(e[7]?.target)
        assertEquals("Required", e[8]?.source)
    }

    @Test
    fun `duplicate targets flag the later row only`() {
        val e = invalid(
            base.copy(mounts = listOf(MountRow(1, "/a", "/data"), MountRow(2, "/b", "/data/"), MountRow(3, "/c", "/other"))),
        ).mountErrors
        assertEquals(setOf(2L), e.keys)
        assertEquals("Duplicate target", e[2]?.target)
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class CreateContainerViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    private val repo = FakeVesselRepository()

    @Test
    fun `errors appear only after first submit, then live`() = runTest {
        val vm = CreateContainerViewModel(repo)
        vm.update { it.copy(rootfs = "") }
        assertTrue(vm.state.value.errors.isEmpty())
        vm.submit()
        assertEquals("Required", vm.state.value.errors[Field.Rootfs])
        vm.update { it.copy(rootfs = "/r") }
        assertNull(vm.state.value.errors[Field.Rootfs])
        assertNull(repo.created)
    }

    @Test
    fun `mount rows can be added, edited and removed with stable ids`() = runTest {
        val vm = CreateContainerViewModel(repo)
        vm.addMount(); vm.addMount(); vm.addMount()
        val ids = vm.state.value.form.mounts.map { it.id }
        assertEquals(3, ids.toSet().size)
        vm.updateMount(ids[2]) { it.copy(source = "/h", target = "/t", readOnly = true) }
        vm.removeMount(ids[0])
        val rows = vm.state.value.form.mounts
        assertEquals(listOf(ids[1], ids[2]), rows.map { it.id })
        assertEquals(MountRow(ids[2], "/h", "/t", true), rows[1])
        vm.addMount()
        assertTrue("ids are never reused", vm.state.value.form.mounts.last().id !in ids)
    }

    @Test
    fun `errors in the advanced section expand it and block submit`() = runTest {
        val vm = CreateContainerViewModel(repo)
        vm.update { it.copy(rootfs = "/r") }
        vm.addMount()
        val id = vm.state.value.form.mounts.single().id
        vm.updateMount(id) { it.copy(source = "/h", target = "relative") }
        assertTrue(!vm.state.value.advancedExpanded)
        vm.submit()
        runCurrent()
        assertTrue(vm.state.value.advancedExpanded)
        assertEquals("Must start with '/'", vm.state.value.mountErrors[id]?.target)
        assertNull(repo.created)
        // Live re-validation after the first attempt.
        vm.updateMount(id) { it.copy(target = "/ok") }
        assertTrue(vm.state.value.mountErrors.isEmpty())
        vm.submit()
        runCurrent()
        assertEquals(listOf(Mount("/h", "/ok", false)), repo.created!!.first.mounts)
    }

    @Test
    fun `toggle advanced`() = runTest {
        val vm = CreateContainerViewModel(repo)
        vm.toggleAdvanced()
        assertTrue(vm.state.value.advancedExpanded)
        vm.toggleAdvanced()
        assertTrue(!vm.state.value.advancedExpanded)
    }

    @Test
    fun `submit creates and starts`() = runTest {
        val vm = CreateContainerViewModel(repo)
        vm.update { it.copy(name = "job", rootfs = "/r", command = "sleep 1") }
        vm.submit()
        runCurrent()
        assertEquals(listOf("sleep", "1"), repo.created!!.first.args)
        assertTrue(repo.created!!.second)
        assertNotNull(vm.state.value.created)
        vm.createdHandled()
        assertNull(vm.state.value.created)
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class GpuJobViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    private val repo = FakeVesselRepository()

    @Test
    fun `form only validates fields relevant to the op`() {
        val (req, errors) = GpuForm(op = GpuOp.FlashAttention, m = "garbage", batchHeads = "8", seqLen = "512", headDim = "64").toRequest()
        assertTrue(errors.isEmpty())
        assertEquals(512, req!!.seqLen)
        assertEquals(0, req.m)
        assertEquals(1, req.ranks)

        val (_, gemmErrors) = GpuForm(op = GpuOp.Gemm, m = "0").toRequest()
        assertEquals(setOf("m"), gemmErrors.keys)

        val (allReduce, _) = GpuForm(op = GpuOp.AllReduceBench, m = "1048576", ranks = "4").toRequest()
        assertEquals(4, allReduce!!.ranks)
        assertEquals(0, allReduce.n)
    }

    @Test
    fun `run records results newest first, bounded`() = runTest {
        val vm = GpuJobViewModel(repo)
        repeat(GpuJobViewModel.MAX_HISTORY + 2) {
            vm.run()
            runCurrent()
        }
        assertEquals(GpuJobViewModel.MAX_HISTORY, vm.state.value.history.size)
        assertEquals(GpuJobViewModel.MAX_HISTORY + 1, vm.state.value.history.first().id)
        assertEquals(GpuOp.Gemm, repo.gpuRequest!!.op)
    }

    @Test
    fun `invalid form never reaches the daemon`() = runTest {
        val vm = GpuJobViewModel(repo)
        vm.update { it.copy(iterations = "") }
        vm.run()
        runCurrent()
        assertNull(repo.gpuRequest)
        assertTrue("iterations" in vm.state.value.errors)
    }
}
