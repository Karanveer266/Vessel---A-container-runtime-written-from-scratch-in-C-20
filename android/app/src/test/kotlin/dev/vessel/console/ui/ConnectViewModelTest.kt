package dev.vessel.console.ui

import dev.vessel.console.model.ConnectionSettings
import dev.vessel.console.model.VesselException
import dev.vessel.console.testing.FakeSettingsStore
import dev.vessel.console.testing.FakeVesselRepository
import dev.vessel.console.testing.MainDispatcherRule
import dev.vessel.console.ui.common.Load
import dev.vessel.console.ui.connect.ConnectViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ConnectViewModelTest {
    @get:Rule val main = MainDispatcherRule()

    private val repo = FakeVesselRepository()
    private val store = FakeSettingsStore(ConnectionSettings(host = "10.0.0.2", port = 50051, token = "t"))

    @Test
    fun `loads saved settings and auto-probes the daemon`() = runTest {
        val vm = ConnectViewModel(store, repo)
        advanceUntilIdle()
        val s = vm.state.value
        assertEquals("10.0.0.2", s.form.host)
        assertEquals("t", s.form.token)
        assertEquals(repo.host, (s.host as Load.Ready).value)
        assertEquals(0, store.saves) // unchanged settings aren't rewritten
    }

    @Test
    fun `editing marks dirty and connect persists before probing`() = runTest {
        val vm = ConnectViewModel(store, repo)
        advanceUntilIdle()
        vm.onFormChange(vm.state.value.form.copy(host = "127.0.0.1", port = "6000"))
        assertTrue(vm.state.value.dirty)
        vm.connect()
        advanceUntilIdle()
        assertEquals(ConnectionSettings("127.0.0.1", 6000, "t", false), store.settings.value)
        assertFalse(vm.state.value.dirty)
        assertEquals(2, repo.hostCalls)
    }

    @Test
    fun `invalid port blocks connect`() = runTest {
        val vm = ConnectViewModel(store, repo)
        advanceUntilIdle()
        vm.onFormChange(vm.state.value.form.copy(port = "99999"))
        assertFalse(vm.state.value.canConnect)
        vm.connect()
        advanceUntilIdle()
        assertEquals(0, store.saves)
    }

    @Test
    fun `auth failure is reported with an actionable message`() = runTest {
        repo.hostError = VesselException.Unauthenticated("bad token")
        val vm = ConnectViewModel(store, repo)
        advanceUntilIdle()
        val failed = vm.state.value.host as Load.Failed
        assertTrue(failed.message, failed.message.contains("token"))
    }

    @Test
    fun `warns when a token would travel in clear text to a remote host`() = runTest {
        val vm = ConnectViewModel(store, repo)
        advanceUntilIdle()
        assertTrue(vm.state.value.insecureWarning)
        vm.onFormChange(vm.state.value.form.copy(useTls = true))
        assertFalse(vm.state.value.insecureWarning)
    }
}
