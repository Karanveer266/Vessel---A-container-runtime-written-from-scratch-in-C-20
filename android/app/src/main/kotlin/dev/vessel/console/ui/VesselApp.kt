package dev.vessel.console.ui

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hasRoute
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.toRoute
import dev.vessel.console.di.AppContainer
import dev.vessel.console.ui.connect.ConnectScreen
import dev.vessel.console.ui.connect.ConnectViewModel
import dev.vessel.console.ui.containers.ContainersScreen
import dev.vessel.console.ui.containers.ContainersViewModel
import dev.vessel.console.ui.create.CreateContainerScreen
import dev.vessel.console.ui.create.CreateContainerViewModel
import dev.vessel.console.ui.detail.ContainerDetailScreen
import dev.vessel.console.ui.gpu.GpuJobScreen
import dev.vessel.console.ui.gpu.GpuJobViewModel
import kotlinx.serialization.Serializable
import kotlin.reflect.KClass

/** Type-safe routes (Navigation Compose 2.8 + kotlinx.serialization). */
sealed interface Route {
    @Serializable data object Connect : Route
    @Serializable data object Containers : Route
    @Serializable data object CreateContainer : Route
    @Serializable data class ContainerDetail(val id: String) : Route
    @Serializable data object GpuJobs : Route
}

private data class TopLevel(val route: Route, val klass: KClass<*>, val label: String, val icon: ImageVector)

private val topLevel = listOf(
    TopLevel(Route.Connect, Route.Connect::class, "Host", Icons.Filled.Home),
    TopLevel(Route.Containers, Route.Containers::class, "Containers", Icons.AutoMirrored.Filled.List),
    TopLevel(Route.GpuJobs, Route.GpuJobs::class, "GPU", Icons.Filled.Build),
)

@Composable
fun VesselApp(container: AppContainer) {
    val nav = rememberNavController()
    val backStack by nav.currentBackStackEntryAsState()
    val destination = backStack?.destination
    val showBottomBar = topLevel.any { t -> destination?.hierarchy?.any { it.hasRoute(t.klass) } == true }

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NavigationBar {
                    topLevel.forEach { t ->
                        NavigationBarItem(
                            selected = destination?.hierarchy?.any { it.hasRoute(t.klass) } == true,
                            onClick = {
                                nav.navigate(t.route) {
                                    popUpTo(nav.graph.findStartDestination().id) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = { Icon(t.icon, contentDescription = null) },
                            label = { Text(t.label) },
                        )
                    }
                }
            }
        },
    ) { padding ->
        NavHost(nav, startDestination = Route.Connect, modifier = Modifier.padding(padding)) {
            composable<Route.Connect> {
                val vm = containerViewModel(container) { ConnectViewModel(settingsStore, repository) }
                ConnectScreen(vm, onOpenContainers = { nav.navigate(Route.Containers) })
            }
            composable<Route.Containers> {
                val vm = containerViewModel(container) { ContainersViewModel(repository) }
                ContainersScreen(
                    vm,
                    onOpen = { nav.navigate(Route.ContainerDetail(it)) },
                    onCreate = { nav.navigate(Route.CreateContainer) },
                )
            }
            composable<Route.CreateContainer> {
                val vm = containerViewModel(container) { CreateContainerViewModel(repository) }
                CreateContainerScreen(
                    vm,
                    onBack = { nav.popBackStack() },
                    onCreated = { id ->
                        nav.navigate(Route.ContainerDetail(id)) { popUpTo<Route.Containers>() }
                    },
                )
            }
            composable<Route.ContainerDetail> { entry ->
                val id = entry.toRoute<Route.ContainerDetail>().id
                ContainerDetailScreen(container, id, onBack = { nav.popBackStack() })
            }
            composable<Route.GpuJobs> {
                val vm = containerViewModel(container) { GpuJobViewModel(repository) }
                GpuJobScreen(vm)
            }
        }
    }
}
