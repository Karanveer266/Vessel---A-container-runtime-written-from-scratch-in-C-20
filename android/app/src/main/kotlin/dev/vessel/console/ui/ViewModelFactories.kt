package dev.vessel.console.ui

import androidx.compose.runtime.Composable
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import dev.vessel.console.di.AppContainer

/**
 * Creates a ViewModel scoped to the current NavBackStackEntry using a lambda
 * over the [AppContainer] — the manual-DI equivalent of `hiltViewModel()`.
 * [key] distinguishes multiple instances of one class on the same entry
 * (e.g. per-container tabs).
 */
@Composable
inline fun <reified VM : ViewModel> containerViewModel(
    container: AppContainer,
    key: String? = null,
    crossinline create: AppContainer.() -> VM,
): VM = viewModel(
    key = key,
    factory = viewModelFactory { initializer { container.create() } },
)
