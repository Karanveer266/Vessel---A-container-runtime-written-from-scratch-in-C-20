package dev.vessel.console.testing

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher

/**
 * Simulates the UI's `collectAsStateWithLifecycle()`: a subscriber that lives
 * until the returned job is cancelled ("app went to background") or the test ends.
 *
 * Note: ViewModels here run infinite tickers/pollers, so tests drive virtual
 * time with advanceTimeBy/runCurrent — never advanceUntilIdle.
 */
@OptIn(ExperimentalCoroutinesApi::class)
fun TestScope.subscribe(flow: Flow<*>): Job =
    backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) { flow.collect {} }
