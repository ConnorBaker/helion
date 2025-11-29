"""Generic pipelined compilation and benchmarking infrastructure.

This module provides reusable components for implementing pipelined autotuning,
where batch N+1 compilation (CPU-bound) overlaps with batch N benchmarking
(GPU-bound) to maximize hardware utilization.

The pipelining approach is applicable to any batched search algorithm and can
significantly reduce total autotuning time by keeping both CPU and GPU busy.

This implementation uses asyncio for true asynchronous execution:
- Configs are generated asynchronously
- Compilation happens in parallel using thread pools
- Benchmarking can run asynchronously (though typically sequential for accuracy)

Example Usage:
--------------
Any search algorithm inheriting from BaseSearch can use pipelining:

    from helion.autotuner import PipelineCallbacks, PipelinedBatchExecutor

    class MyCustomSearch(BaseSearch):
        def _autotune(self) -> Config:
            # Check device supports pipelining
            if self.settings.autotune_precompile:
                if self.kernel.env.device.type != "cuda":
                    raise exc.InvalidAPIUsage("Pipelining requires CUDA device")

            # Define how to prepare batches
            def prepare_batch(batch_size: int, completed: int) -> tuple[list, list[Config]] | None:
                if completed >= self.max_trials:
                    return None
                # Generate trials/requests and configs for this batch
                trials = [...]  # Your trial objects
                configs = [...]  # Corresponding configs
                return trials, configs

            # Define how to report results
            def report_results(trials: list, results: Sequence[BenchmarkResult]) -> None:
                for trial, result in zip(trials, results):
                    # Update your search algorithm state
                    self.update_with_result(trial, result)

            # Create callbacks
            callbacks = PipelineCallbacks(
                prepare_batch=prepare_batch,
                report_results=report_results,
                should_continue=lambda completed: completed < self.max_trials,
                get_batch_description=lambda batch_num, completed: f"Batch {batch_num}"
            )

            # Run pipelined execution (blocking call that runs async internally)
            executor = PipelinedBatchExecutor(self, callbacks)
            trials_completed = executor.run(batch_size=10)

            return self.get_best_config()
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from itertools import starmap
import math
from typing import TYPE_CHECKING
from typing import Callable
from typing import Sequence

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .base_search import BaseSearch
    from .base_search import BenchmarkResult
    from .base_search import PrecompileFuture
    from helion.runtime.kernel import CompiledConfig


@dataclasses.dataclass
class CompilationBatch[T]:
    """State for a batch in the compilation/benchmarking pipeline.

    This dataclass holds all the state needed to track a batch through
    the pipeline: the original trial/request objects, configurations,
    compiled functions, and compilation futures.

    Type Parameters:
        T: Type of trial/request objects (e.g., optuna.Trial)

    Attributes:
        trials: Trial or request objects for this batch.
        configs: Configurations to be benchmarked.
        fns: Compiled kernel functions.
        futures: PrecompileFuture objects tracking compilation progress.
    """

    trials: list[T]
    configs: list[Config]
    fns: list[object]
    futures: list[PrecompileFuture]


@dataclasses.dataclass
class PipelineCallbacks[T]:
    """Callbacks for customizing the pipelined execution.

    These callbacks allow different search algorithms to integrate with
    the generic pipelining infrastructure.

    Type Parameters:
        T: Type of trial/request objects

    Attributes:
        prepare_batch: Prepare next batch of trials and configs.
            Called to generate the next batch to process.
            Arguments: requested batch size, trials completed so far.
            Returns None when no more batches are available.

        report_results: Report results back to the search algorithm.
            Called after benchmarking completes for a batch.
            Arguments: trials and their corresponding results.

        should_continue: Check if pipeline should continue.
            Called before each batch to check for stopping conditions
            (e.g., timeout, trial limit).
            Arguments: trials completed so far.
            Returns False to stop the pipeline.

        get_batch_description: Get description for progress bar.
            Optional callback to customize progress bar text.
            Arguments: batch number (1-indexed), trials completed so far.
            Returns description string or None to disable progress bar.
    """

    prepare_batch: Callable[[int, int], tuple[list[T], list[Config]] | None]
    report_results: Callable[[list[T], Sequence[BenchmarkResult]], None]
    should_continue: Callable[[int], bool]
    get_batch_description: Callable[[int, int], str | None] = (
        lambda batch_num, trials_completed: None
    )


class PipelinedBatchExecutor[T]:
    """Executor for pipelined compilation and benchmarking.

    This class implements the core pipelining logic that can be reused across
    different search algorithms. It orchestrates the overlap of compilation
    and benchmarking to maximize CPU and GPU utilization.

    The pipeline works as follows:
    1. Prepares and compiles first batch asynchronously
    2. For each batch:
       - Prepares next batch asynchronously while waiting for current compilation
       - Waits for current batch compilation to finish
       - Immediately starts next batch compilation
       - Benchmarks current batch asynchronously while next compiles
       - Reports results to the search algorithm

    All compilation happens in parallel using thread pools, and the overall
    coordination uses asyncio for efficient resource utilization.

    Type Parameters:
        T: Type of trial/request objects

    Example:
        >>> executor = PipelinedBatchExecutor(search_instance, callbacks)
        >>> trials_completed = executor.run(batch_size=10)
    """

    def __init__(
        self,
        search: BaseSearch,
        callbacks: PipelineCallbacks[T],
        max_compile_workers: int | None = None,
    ) -> None:
        """Initialize the pipeline executor.

        Args:
            search: BaseSearch instance providing compilation and benchmarking.
            callbacks: Callbacks for batch preparation, result reporting, etc.
            max_compile_workers: Maximum number of parallel compilation workers.
                If None, defaults to min(32, (cpu_count or 1) + 4).
        """
        self.search = search
        self.callbacks = callbacks
        self.max_compile_workers = max_compile_workers

    async def _compile_config_async(self, config: Config) -> object:
        """Compile a single config asynchronously.

        Args:
            config: Configuration to compile.

        Returns:
            Compiled kernel function.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,  # Use default executor
            self.search.kernel.compile_config,
            config,
            False,  # allow_print=False
        )

    async def _start_compilation(
        self, trials: list[T], configs: list[Config]
    ) -> CompilationBatch[T]:
        """Start compiling a batch of configs in parallel.

        Args:
            trials: Trial/request objects for this batch.
            configs: Configurations to compile.

        Returns:
            CompilationBatch with compilation in progress.
        """
        # Compile all configs in parallel
        fns: list[CompiledConfig] = await asyncio.gather(
            *[self._compile_config_async(cfg) for cfg in configs]
        )

        # Start precompile futures (these spawn subprocesses)
        futures = list(
            starmap(
                self.search.start_precompile_and_check_for_hangs,
                zip(configs, fns, strict=True),
            )
        )

        return CompilationBatch(trials, list(configs), list(fns), futures)

    async def _wait_for_compilation(
        self, batch: CompilationBatch[T], desc: str | None
    ) -> list[bool]:
        """Wait for a batch compilation to complete.

        Args:
            batch: CompilationBatch to wait for.
            desc: Description for progress bar.

        Returns:
            List of booleans indicating which compilations succeeded.
        """
        from .base_search import PrecompileFuture

        loop = asyncio.get_running_loop()

        # Run the blocking wait_for_all in a thread pool
        return await loop.run_in_executor(
            None, PrecompileFuture.wait_for_all, batch.futures, desc
        )

    async def _benchmark_single(
        self, config: Config, fn: object, ok: bool, future: PrecompileFuture
    ) -> BenchmarkResult:
        """Benchmark a single config asynchronously.

        Args:
            config: Configuration being benchmarked.
            fn: Compiled kernel function.
            ok: Whether compilation succeeded.
            future: PrecompileFuture for this config.

        Returns:
            BenchmarkResult with performance metrics.
        """
        from .base_search import BenchmarkResult

        compile_time = (
            future.elapsed if future.process is not None and future.started else None
        )

        if ok:
            # Run benchmark in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            perf = await loop.run_in_executor(
                None, self.search.benchmark_function, config, fn
            )
            status = "ok" if math.isfinite(perf) else "error"
        else:
            perf = math.inf
            status = "timeout" if future.failure_reason == "timeout" else "error"

        return BenchmarkResult(
            config=config,
            fn=fn,
            perf=perf,
            status=status,
            compile_time=compile_time,
        )

    async def _benchmark_batch(
        self, batch: CompilationBatch[T], is_working: list[bool]
    ) -> list[BenchmarkResult]:
        """Benchmark a compiled batch.

        Benchmarks are run sequentially to avoid noisy results from concurrent
        GPU usage, but each benchmark runs asynchronously.

        Args:
            batch: CompilationBatch with completed compilation.
            is_working: List indicating which compilations succeeded.

        Returns:
            List of BenchmarkResult objects.
        """
        from .base_search import BenchmarkResult

        results: list[BenchmarkResult] = []
        for config, fn, ok, future in zip(
            batch.configs, batch.fns, is_working, batch.futures, strict=True
        ):
            result = await self._benchmark_single(config, fn, ok, future)
            results.append(result)
        return results

    async def _run_async(self, batch_size: int) -> int:
        """Run the pipelined compilation and benchmarking asynchronously.

        Args:
            batch_size: Number of trials per batch.

        Returns:
            Total number of trials completed.
        """
        trials_completed = 0

        # Prepare and start compilation of first batch
        first_batch = self.callbacks.prepare_batch(batch_size, trials_completed)
        if not first_batch:
            return 0

        pending = await self._start_compilation(*first_batch)

        while self.callbacks.should_continue(trials_completed):
            # Prepare next batch asynchronously
            next_batch_size = batch_size
            next_trials_completed = trials_completed + len(pending.configs)
            prepare_task = asyncio.create_task(
                asyncio.to_thread(
                    self.callbacks.prepare_batch, next_batch_size, next_trials_completed
                )
            )

            # Wait for current batch compilation
            batch_num = trials_completed // batch_size + 1
            desc = self.callbacks.get_batch_description(batch_num, trials_completed)
            is_working = await self._wait_for_compilation(pending, desc)

            # Get the next batch we prepared (await the task)
            next_batch = await prepare_task

            # Start next batch compilation NOW (overlaps with benchmarking)
            if next_batch:
                compile_task = asyncio.create_task(self._start_compilation(*next_batch))
            else:
                compile_task = None

            # Benchmark current batch while next compiles
            results = await self._benchmark_batch(pending, is_working)

            # Report results
            await asyncio.to_thread(
                self.callbacks.report_results, pending.trials, results
            )

            trials_completed += len(pending.configs)

            # Wait for next batch compilation to finish
            if compile_task is not None:
                pending = await compile_task
            else:
                break

        return trials_completed

    def run(self, batch_size: int) -> int:
        """Run the pipelined compilation and benchmarking.

        This is a synchronous wrapper around the async implementation.

        Args:
            batch_size: Number of trials per batch.

        Returns:
            Total number of trials completed.
        """
        # Set up thread pool for compilation
        if self.max_compile_workers is not None:
            executor = ThreadPoolExecutor(max_workers=self.max_compile_workers)
        else:
            executor = None

        try:
            # Set the default executor for run_in_executor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            if executor is not None:
                loop.set_default_executor(executor)

            # Run the async pipeline
            return loop.run_until_complete(self._run_async(batch_size))
        finally:
            loop.close()
            if executor is not None:
                executor.shutdown(wait=True)


__all__ = ["CompilationBatch", "PipelineCallbacks", "PipelinedBatchExecutor"]
