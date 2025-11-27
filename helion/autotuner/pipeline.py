"""Generic pipelined compilation and benchmarking infrastructure.

This module provides reusable components for implementing pipelined autotuning,
where batch N+1 compilation (CPU-bound) overlaps with batch N benchmarking
(GPU-bound) to maximize hardware utilization.

The pipelining approach is applicable to any batched search algorithm and can
significantly reduce total autotuning time by keeping both CPU and GPU busy.

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

            # Run pipelined execution
            executor = PipelinedBatchExecutor(self, callbacks)
            trials_completed = executor.run(batch_size=10)

            return self.get_best_config()
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING
from typing import Callable
from typing import Generic
from typing import Sequence
from typing import TypeVar

if TYPE_CHECKING:
    from .base_search import BaseSearch
    from .base_search import BenchmarkResult
    from .base_search import PrecompileFuture
    from .config import Config


# Generic type for trial/request objects (e.g., optuna.Trial, or any custom type)
T = TypeVar("T")


@dataclasses.dataclass
class CompilationBatch(Generic[T]):
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
class PipelineCallbacks(Generic[T]):
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


class PipelinedBatchExecutor(Generic[T]):
    """Executor for pipelined compilation and benchmarking.

    This class implements the core pipelining logic that can be reused across
    different search algorithms. It orchestrates the overlap of compilation
    and benchmarking to maximize CPU and GPU utilization.

    The pipeline works as follows:
    1. Compiles first batch immediately
    2. For each batch:
       - Waits for current batch compilation to finish
       - Immediately starts next batch compilation
       - Benchmarks current batch while next compiles in background
       - Reports results to the search algorithm

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
    ) -> None:
        """Initialize the pipeline executor.

        Args:
            search: BaseSearch instance providing compilation and benchmarking.
            callbacks: Callbacks for batch preparation, result reporting, etc.
        """
        self.search = search
        self.callbacks = callbacks

    def _start_compilation(
        self, trials: list[T], configs: list[Config]
    ) -> CompilationBatch[T]:
        """Start compiling a batch of configs.

        Args:
            trials: Trial/request objects for this batch.
            configs: Configurations to compile.

        Returns:
            CompilationBatch with compilation in progress.
        """
        from itertools import starmap

        fns = [
            self.search.kernel.compile_config(cfg, allow_print=False) for cfg in configs
        ]
        futures = list(
            starmap(
                self.search.start_precompile_and_check_for_hangs,
                zip(configs, fns, strict=True),
            )
        )
        return CompilationBatch(trials, configs, fns, futures)

    def _benchmark_batch(
        self, batch: CompilationBatch[T], is_working: list[bool]
    ) -> list[BenchmarkResult]:
        """Benchmark a compiled batch.

        Args:
            batch: CompilationBatch with completed compilation.
            is_working: List indicating which compilations succeeded.

        Returns:
            List of BenchmarkResult objects.
        """
        from .base_search import BenchmarkResult

        results = []
        for idx, (fn, ok, future) in enumerate(
            zip(batch.fns, is_working, batch.futures, strict=True)
        ):
            config = batch.configs[idx]
            compile_time = (
                future.elapsed
                if future.process is not None and future.started
                else None
            )

            if ok:
                perf = self.search.benchmark_function(config, fn)
                status = "ok" if math.isfinite(perf) else "error"
            else:
                perf = math.inf
                status = "timeout" if future.failure_reason == "timeout" else "error"

            results.append(
                BenchmarkResult(
                    config=config,
                    fn=fn,
                    perf=perf,
                    status=status,
                    compile_time=compile_time,
                )
            )
        return results

    def run(self, batch_size: int) -> int:
        """Run the pipelined compilation and benchmarking.

        Args:
            batch_size: Number of trials per batch.

        Returns:
            Total number of trials completed.
        """
        from .base_search import PrecompileFuture

        trials_completed = 0

        # Start pipeline with first batch
        first_batch = self.callbacks.prepare_batch(batch_size, trials_completed)
        if not first_batch:
            return 0

        pending = self._start_compilation(*first_batch)

        while self.callbacks.should_continue(trials_completed):
            # Prepare next batch
            next_batch = self.callbacks.prepare_batch(batch_size, trials_completed)

            # Wait for current batch compilation
            batch_num = trials_completed // batch_size + 1
            desc = self.callbacks.get_batch_description(batch_num, trials_completed)
            is_working = PrecompileFuture.wait_for_all(pending.futures, desc=desc)

            # Start next batch compilation NOW (overlaps with benchmarking)
            next_pending = self._start_compilation(*next_batch) if next_batch else None

            # Benchmark current batch while next compiles
            results = self._benchmark_batch(pending, is_working)
            self.callbacks.report_results(pending.trials, results)

            trials_completed += len(pending.configs)

            if next_pending is None:
                break
            pending = next_pending

        return trials_completed


__all__ = ["CompilationBatch", "PipelineCallbacks", "PipelinedBatchExecutor"]
