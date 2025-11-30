"""Generic pipelined compilation and benchmarking infrastructure.

This module provides reusable components for implementing pipelined autotuning
with stream processing, where configs are generated, compiled, and benchmarked
out of order to maximize hardware utilization.

The stream processing approach enables:
- Configs generated on-demand without fixed batches
- Immediate compilation when config is generated
- Immediate benchmarking when compilation completes
- Immediate result reporting when benchmarking completes
- Concurrent execution with separate controls for exploration and compilation parallelism

This implementation uses asyncio for true asynchronous execution:
- Configs are generated asynchronously up to max_configs_ahead limit (exploration control)
- Compilation happens in parallel using thread pools (limited by autotune_precompile_jobs)
- Benchmarking runs asynchronously (sequential per-config for accuracy)

Example Usage:
--------------
Any search algorithm inheriting from BaseSearch can use stream processing:

    from helion.autotuner import StreamCallbacks, StreamExecutor

    class MyCustomSearch(BaseSearch):
        def _autotune(self) -> Config:
            # Check device supports pipelining
            if self.settings.autotune_precompile:
                if self.kernel.env.device.type != "cuda":
                    raise exc.InvalidAPIUsage("Pipelining requires CUDA device")

            # Define how to generate next config
            def generate_next(completed: int) -> tuple[object, Config] | None:
                if completed >= self.max_trials:
                    return None
                trial = ...  # Your trial object
                config = ...  # Corresponding config
                return trial, config

            # Define how to report a single result
            def report_result(trial: object, result: BenchmarkResult) -> None:
                # Update your search algorithm state
                self.update_with_result(trial, result)

            # Create callbacks
            callbacks = StreamCallbacks(
                generate_next=generate_next,
                report_result=report_result,
                should_continue=lambda completed: completed < self.max_trials,
            )

            # Run stream execution (blocking call that runs async internally)
            executor = StreamExecutor(self, callbacks, max_configs_ahead=20)
            trials_completed = executor.run()

            return self.get_best_config()
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import Callable
from typing import Protocol

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .base_search import BaseSearch
    from .base_search import BenchmarkResult
    from .base_search import PrecompileFuture


class ProgressTracker(Protocol):
    """Protocol for tracking progress through the pipeline stages."""

    def on_config_generated(self) -> None:
        """Called when a config is generated."""
        ...

    def on_compilation_complete(self) -> None:
        """Called when a config finishes compilation."""
        ...

    def on_benchmark_complete(self) -> None:
        """Called when a config finishes benchmarking."""
        ...


@dataclasses.dataclass
class ConfigInFlight[T]:
    """State for a single config in the processing pipeline.

    This dataclass holds all state needed to track a config through
    the stream pipeline: trial object, configuration, compilation future,
    and compilation result.

    Type Parameters:
        T: Type of trial/request objects (e.g., optuna.Trial)

    Attributes:
        trial: Trial or request object for this config.
        config: Configuration to be benchmarked.
        fn: Compiled kernel function (set after compilation).
        future: PrecompileFuture tracking compilation progress (set after compilation starts).
        compile_task: Asyncio task for compilation.
    """

    trial: T
    config: Config
    fn: object | None = None
    future: PrecompileFuture | None = None
    compile_task: asyncio.Task | None = None


@dataclasses.dataclass
class StreamCallbacks[T]:
    """Callbacks for customizing the stream execution.

    These callbacks allow different search algorithms to integrate with
    the generic stream processing infrastructure.

    Type Parameters:
        T: Type of trial/request objects

    Attributes:
        generate_next: Generate next trial and config.
            Called when capacity is available to start a new config.
            Arguments: number of trials completed so far.
            Returns None when no more configs are available.

        report_result: Report a single result back to the search algorithm.
            Called immediately after benchmarking completes for a config.
            Arguments: trial and its corresponding result.

        should_continue: Check if pipeline should continue.
            Called to check for stopping conditions (e.g., timeout, trial limit).
            Arguments: trials completed so far.
            Returns False to stop the pipeline.
    """

    generate_next: Callable[[int], tuple[T, Config] | None]
    report_result: Callable[[T, BenchmarkResult], None]
    should_continue: Callable[[int], bool]


class StreamExecutor[T]:
    """Executor for stream-based compilation and benchmarking.

    This class implements stream processing where configs are generated, compiled,
    and benchmarked out of order to maximize hardware utilization. Configs flow
    through the pipeline with separate controls for exploration and compilation parallelism.

    The stream works as follows:
    1. Generate configs on-demand up to max_configs_ahead (controls exploration vs learning)
    2. Each config is compiled immediately upon generation (parallel, limited by _jobs)
    3. Each config is benchmarked immediately when compilation completes
    4. Results are reported immediately when benchmarking completes
    5. New configs are generated as previous configs complete benchmarking

    Compilation parallelism is controlled by BaseSearch._jobs (typically CPU count),
    while max_configs_ahead controls how far ahead to explore before learning from results.
    The overall coordination uses asyncio for efficient resource utilization.

    Type Parameters:
        T: Type of trial/request objects

    Example:
        >>> executor = StreamExecutor(search_instance, callbacks, max_configs_ahead=20)
        >>> trials_completed = executor.run()
    """

    def __init__(
        self,
        search: BaseSearch,
        callbacks: StreamCallbacks[T],
        max_configs_ahead: int = 20,
        progress_tracker: ProgressTracker | None = None,
    ) -> None:
        """Initialize the stream executor.

        Args:
            search: BaseSearch instance providing compilation and benchmarking.
            callbacks: Callbacks for config generation, result reporting, etc.
            max_configs_ahead: Maximum number of configs to generate ahead of benchmark completion.
                Controls exploration vs exploitation tradeoff. Higher values explore more configs
                before learning from results. Independent of compilation parallelism.
            progress_tracker: Optional progress tracker for updating progress bars.
        """
        self.search = search
        self.callbacks = callbacks
        self.max_configs_ahead = max_configs_ahead
        self.progress_tracker = progress_tracker
        # Semaphore to limit concurrent precompile subprocesses
        # Uses the same limit as BaseSearch._jobs
        self._precompile_semaphore: asyncio.Semaphore | None = None
        # Track CUDA context corruption from unrecoverable errors
        self._had_unrecoverable_error = False

    def _attempt_cuda_recovery(self) -> None:
        """Attempt to recover from CUDA context corruption.

        This tries various CUDA reset operations that might help recover
        from certain types of errors. However, for severe errors like
        illegal memory access, the CUDA context is often unrecoverable
        without restarting the process.
        """
        try:
            import torch

            if torch.cuda.is_available():
                self.search.log(
                    "Attempting CUDA recovery after unrecoverable error..."
                )
                # Try to synchronize first to clear pending operations
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass  # Expected to fail if context is corrupted

                # Clear memory cache
                torch.cuda.empty_cache()

                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                # Try synchronize again to verify recovery
                try:
                    torch.cuda.synchronize()
                    self.search.log(
                        "CUDA recovery appears successful, continuing search"
                    )
                except Exception:
                    self.search.log(
                        "CUDA recovery failed - context remains corrupted"
                    )
        except Exception as e:
            self.search.log(f"Error during CUDA recovery attempt: {e}")

    async def _compile_and_precompile(
        self, in_flight: ConfigInFlight[T]
    ) -> ConfigInFlight[T]:
        """Compile a config and start precompilation asynchronously.

        Args:
            in_flight: ConfigInFlight object with config to compile.

        Returns:
            Updated ConfigInFlight with fn and future set.
        """
        import functools

        loop = asyncio.get_running_loop()

        # Compile the config (allow_print is keyword-only, so use functools.partial)
        fn = await loop.run_in_executor(
            None,
            functools.partial(
                self.search.kernel.compile_config, allow_print=False
            ),
            in_flight.config,
        )

        # Start precompile future (spawns subprocess)
        future = await loop.run_in_executor(
            None,
            self.search.start_precompile_and_check_for_hangs,
            in_flight.config,
            fn,
        )

        in_flight.fn = fn
        in_flight.future = future
        return in_flight

    async def _wait_for_precompile(self, in_flight: ConfigInFlight[T]) -> bool:
        """Wait for precompilation to complete.

        This method uses process.join() wrapped in asyncio.to_thread() to wait
        asynchronously without polling. The join() call blocks until the process
        completes, but running it in a thread pool allows the event loop to
        handle other tasks concurrently.

        A semaphore limits concurrent precompile subprocesses to avoid overwhelming
        the system (similar to the _jobs cap in the batch wait_for_all).

        Args:
            in_flight: ConfigInFlight with ongoing precompilation.

        Returns:
            True if compilation succeeded, False otherwise.
        """
        future = in_flight.future
        assert future is not None

        # Handle already-completed futures (e.g., skip futures)
        if future.ok is not None:
            return future.ok

        # Acquire semaphore to limit concurrent precompile processes
        assert self._precompile_semaphore is not None
        async with self._precompile_semaphore:
            # Start the process if not started
            if not future.started:
                await asyncio.to_thread(future.start)

            # Wait for process to complete using join() (no polling!)
            # join(timeout) blocks until process exits or timeout expires
            # Running in thread pool makes it async without busy-waiting
            process = future.process
            assert process is not None
            timeout = future.seconds_left()
            await asyncio.to_thread(process.join, timeout)

            # Mark complete and consume result (these do actual work, use thread pool)
            await asyncio.to_thread(future._mark_complete)
            await asyncio.to_thread(lambda: future._consume_result(raise_on_raise=True))

        assert future.ok is not None
        return future.ok

    async def _benchmark_config(
        self, in_flight: ConfigInFlight[T], ok: bool
    ) -> BenchmarkResult:
        """Benchmark a compiled config asynchronously.

        Args:
            in_flight: ConfigInFlight with completed compilation.
            ok: Whether compilation succeeded.

        Returns:
            BenchmarkResult with performance metrics.
        """
        from .base_search import BenchmarkResult

        compile_time = (
            in_flight.future.elapsed
            if in_flight.future.process is not None and in_flight.future.started
            else None
        )

        if ok:
            # Run benchmark in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            perf = await loop.run_in_executor(
                None, self.search.benchmark_function, in_flight.config, in_flight.fn
            )
            # Capture accuracy error immediately after benchmark
            # (must be done before another benchmark overwrites it)
            accuracy_error = self.search.last_accuracy_error
            # Set status based on whether performance is finite
            status = "ok" if math.isfinite(perf) else "error"
        else:
            perf = math.inf
            accuracy_error = None  # No accuracy check for failed compilations
            status = (
                "timeout" if in_flight.future.failure_reason == "timeout" else "error"
            )

        return BenchmarkResult(
            config=in_flight.config,
            fn=in_flight.fn,
            perf=perf,
            status=status,
            compile_time=compile_time,
            accuracy_error=accuracy_error,
        )

    async def _process_config(self, in_flight: ConfigInFlight[T]) -> None:
        """Process a single config through the pipeline.

        This coroutine handles the entire lifecycle:
        1. Compile and start precompilation
        2. Wait for precompilation to complete
        3. Benchmark the config
        4. Report the result

        Args:
            in_flight: ConfigInFlight to process.

        Raises:
            optuna.TrialPruned: If the meta-pruner decides to stop the study.
                This exception is allowed to propagate to signal study termination.
        """
        try:
            # Compile and start precompilation
            in_flight = await self._compile_and_precompile(in_flight)

            # Wait for precompilation
            ok = await self._wait_for_precompile(in_flight)

            # Update progress: compilation complete
            if self.progress_tracker is not None:
                await asyncio.to_thread(self.progress_tracker.on_compilation_complete)

            # Benchmark
            result = await self._benchmark_config(in_flight, ok)

            # Update progress: benchmark complete
            if self.progress_tracker is not None:
                await asyncio.to_thread(self.progress_tracker.on_benchmark_complete)

            # Report result
            await asyncio.to_thread(
                self.callbacks.report_result, in_flight.trial, result
            )

        except Exception as e:
            # Don't catch TrialPruned - it signals the study should stop
            # The trial was already marked as COMPLETE before pruning check
            # Check by class name to avoid importing optuna (keep pipeline generic)
            if e.__class__.__name__ == "TrialPruned":
                raise

            # Check for unrecoverable runtime errors (e.g., CUDA illegal memory access)
            is_unrecoverable = e.__class__.__name__ == "TritonUnrecoverableRuntimeError"

            if is_unrecoverable:
                if self._had_unrecoverable_error:
                    # Second unrecoverable error - CUDA context is definitely corrupted
                    # Abort the search to avoid wasting time on trials that will all fail
                    self.search.log(
                        "Second unrecoverable error detected - CUDA context is corrupted.\n"
                        "Aborting search. Set HELION_AUTOTUNE_PRECOMPILE='spawn' to isolate "
                        "these errors in subprocesses."
                    )
                    # Re-raise to abort the search
                    raise
                else:
                    # First unrecoverable error - try to recover
                    self._had_unrecoverable_error = True
                    self.search.log(
                        "Unrecoverable error detected. Attempting CUDA recovery..."
                    )
                    await asyncio.to_thread(self._attempt_cuda_recovery)

            # Report failure - only for unexpected exceptions
            # (benchmark_function should not raise, it returns inf on failure)
            import traceback

            from .base_search import BenchmarkResult

            # Log the actual exception for debugging
            error_msg = f"{e.__class__.__name__}: {e}"
            self.search.log(f"Pipeline error processing config: {error_msg}")

            # Log full traceback at debug level if verbose
            if hasattr(self.search, 'settings') and getattr(self.search.settings, 'verbose', False):
                tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                self.search.log(f"Full traceback:\n{tb_str}")

            # Still update progress even on failure
            if self.progress_tracker is not None:
                await asyncio.to_thread(self.progress_tracker.on_compilation_complete)
                await asyncio.to_thread(self.progress_tracker.on_benchmark_complete)

            await asyncio.to_thread(
                self.callbacks.report_result,
                in_flight.trial,
                BenchmarkResult(
                    config=in_flight.config,
                    fn=in_flight.fn,
                    perf=math.inf,
                    status="error",
                    compile_time=None,
                    accuracy_error=None,  # No accuracy check for failed configs
                ),
            )

    async def run_async(self) -> int:
        """Run the stream processing asynchronously.

        This is the async implementation that can be called from async contexts.
        For synchronous contexts, use run() instead.

        Returns:
            Total number of trials completed.
        """
        # Initialize semaphore for precompile concurrency control
        # Uses the same limit as BaseSearch._jobs (typically cpu_count)
        # _jobs is always set in BaseSearch.__init__, so no fallback needed
        precompile_limit = self.search._jobs
        self._precompile_semaphore = asyncio.Semaphore(precompile_limit)

        trials_generated = 0
        trials_completed = 0
        in_flight_tasks: set[asyncio.Task] = set()

        while self.callbacks.should_continue(trials_completed):
            # Generate new configs up to max_configs_ahead limit
            # This controls how far ahead we generate configs before learning from results
            while (trials_generated - trials_completed) < self.max_configs_ahead:
                # Try to generate next config
                next_item = await asyncio.to_thread(
                    self.callbacks.generate_next, trials_generated
                )

                if next_item is None:
                    # No more configs to generate
                    break

                trials_generated += 1
                trial, config = next_item

                in_flight = ConfigInFlight(trial=trial, config=config)

                # Update progress: config generated
                if self.progress_tracker is not None:
                    await asyncio.to_thread(self.progress_tracker.on_config_generated)

                # Start processing this config
                task = asyncio.create_task(self._process_config(in_flight))
                in_flight_tasks.add(task)

            # If no tasks in flight, we're done
            if not in_flight_tasks:
                break

            # Wait for at least one task to complete
            done, in_flight_tasks = await asyncio.wait(
                in_flight_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Check if any task raised TrialPruned or TritonUnrecoverableRuntimeError
            # (both signal the study should stop)
            for task in done:
                if task.exception() is not None:
                    exc = task.exception()
                    if exc.__class__.__name__ in ("TrialPruned", "TritonUnrecoverableRuntimeError"):
                        # Study was pruned or CUDA context corrupted - cancel remaining tasks
                        for remaining_task in in_flight_tasks:
                            remaining_task.cancel()
                        raise exc

            # Update completed count
            trials_completed += len(done)

        # Wait for any remaining tasks to complete
        if in_flight_tasks:
            done_final, _ = await asyncio.wait(in_flight_tasks)

            # Check if any remaining task raised TrialPruned or TritonUnrecoverableRuntimeError
            for task in done_final:
                if task.exception() is not None:
                    exc = task.exception()
                    if exc.__class__.__name__ in ("TrialPruned", "TritonUnrecoverableRuntimeError"):
                        raise exc

            trials_completed += len(done_final)

        return trials_completed

    def run(self) -> int:
        """Run the stream processing synchronously.

        This creates a new event loop and runs the async implementation.
        For calling from async contexts, use run_async() directly.

        Returns:
            Total number of trials completed.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the async stream
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()


__all__ = ["ConfigInFlight", "ProgressTracker", "StreamCallbacks", "StreamExecutor"]
