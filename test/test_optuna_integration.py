"""Integration tests for OptunaSearch autotuner."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfCpu
from helion._testing import skipIfRocm
from helion.autotuner import OptunaSearch
from helion.autotuner import OptunaSearchParams
import helion.language as hl


class TestOptunaSearch(RefEagerTestDisabled, TestCase):
    """Test suite for OptunaSearch autotuner."""

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_import(self) -> None:
        """Test that OptunaSearch can be imported."""
        # Just verifying the import worked (done at module level)
        self.assertIsNotNone(OptunaSearch)
        self.assertIsNotNone(OptunaSearchParams)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_basic_kernel_env_var(self) -> None:
        """Test OptunaSearch via HELION_AUTOTUNER environment variable."""

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([128, 128], device=DEVICE)
        b = torch.randn([128, 128], device=DEVICE)

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "OptunaSearch",
                "HELION_AUTOTUNE_OPTUNA_TRIALS": "5",
            },
            clear=False,
        ):
            result = add(a, b)
            expected = a + b
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_direct_instantiation(self) -> None:
        """Test OptunaSearch with direct instantiation."""

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([128, 128], device=DEVICE)
        b = torch.randn([128, 128], device=DEVICE)

        bound = add.bind((a, b))

        # Create OptunaSearch directly
        params = OptunaSearchParams(
            n_trials=5,
            sampler_name="tpe",
            study_name="test_direct_instantiation",
            load_if_exists=False,
            show_progress_bar=False,
        )
        autotuner = OptunaSearch(bound, (a, b), params=params)

        # Run autotuning
        best_config = autotuner.autotune()
        self.assertIsNotNone(best_config)

        # Verify the study is accessible
        study = autotuner.get_study()
        self.assertIsNotNone(study)
        self.assertEqual(len(study.trials), 5)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_different_samplers(self) -> None:
        """Test that different Optuna samplers work correctly."""

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([128, 128], device=DEVICE)
        b = torch.randn([128, 128], device=DEVICE)
        bound = add.bind((a, b))

        for sampler_name in ["random", "tpe"]:
            with self.subTest(sampler=sampler_name):
                params = OptunaSearchParams(
                    n_trials=3,
                    sampler_name=sampler_name,
                    study_name=f"test_sampler_{sampler_name}",
                    load_if_exists=False,
                    show_progress_bar=False,
                )
                autotuner = OptunaSearch(bound, (a, b), params=params)
                best_config = autotuner.autotune()
                self.assertIsNotNone(best_config)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_persistence(self) -> None:
        """Test that Optuna study persistence works with JournalStorage."""

        @helion.kernel(autotune_log_level=0)
        def mul(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] * b[tile]
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64, 64], device=DEVICE)
        bound = mul.bind((a, b))

        # Create temporary journal file for JournalStorage (better for parallel execution)
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            journal_path = f.name

        try:
            # Using file path (not URL) triggers JournalStorage
            storage = journal_path

            # First run: create study
            params1 = OptunaSearchParams(
                n_trials=3,
                storage=storage,
                study_name="test_persistence",
                show_progress_bar=False,
            )
            autotuner1 = OptunaSearch(bound, (a, b), params=params1)
            autotuner1.autotune()
            study1 = autotuner1.get_study()
            n_trials_1 = len(study1.trials)

            # Clean up storage explicitly
            autotuner1.close()

            # Second run: resume study
            params2 = OptunaSearchParams(
                n_trials=2,
                storage=storage,
                study_name="test_persistence",
                load_if_exists=True,
                show_progress_bar=False,
            )
            autotuner2 = OptunaSearch(bound, (a, b), params=params2)
            autotuner2.autotune()
            study2 = autotuner2.get_study()
            n_trials_2 = len(study2.trials)

            # Verify trials were added to existing study
            self.assertEqual(n_trials_2, n_trials_1 + 2)

            # Clean up storage explicitly
            autotuner2.close()

        finally:
            if os.path.exists(journal_path):
                os.unlink(journal_path)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_max_configs_ahead_parameter(self) -> None:
        """Test that max_configs_ahead parameter is respected."""

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64, 64], device=DEVICE)
        bound = add.bind((a, b))

        # Test with different max_configs_ahead values
        for max_configs_ahead in [5, 10, 20]:
            with self.subTest(max_configs_ahead=max_configs_ahead):
                params = OptunaSearchParams(
                    n_trials=10,
                    sampler_name="random",
                    study_name=f"test_max_configs_ahead_{max_configs_ahead}",
                    load_if_exists=False,
                    show_progress_bar=False,
                    max_configs_ahead=max_configs_ahead,
                )
                autotuner = OptunaSearch(bound, (a, b), params=params)
                self.assertEqual(autotuner.params.max_configs_ahead, max_configs_ahead)
                best_config = autotuner.autotune()
                self.assertIsNotNone(best_config)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_env_var_overrides(self) -> None:
        """Test that environment variables override default parameters."""

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64, 64], device=DEVICE)
        bound = add.bind((a, b))

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_OPTUNA_TRIALS": "7",
                "HELION_AUTOTUNE_OPTUNA_SAMPLER": "random",
                "HELION_AUTOTUNE_OPTUNA_MAX_CONFIGS_AHEAD": "15",
            },
            clear=False,
        ):
            params = OptunaSearchParams(
                study_name="test_env_overrides",
                load_if_exists=False,
                show_progress_bar=False,
            )
            autotuner = OptunaSearch(bound, (a, b), params=params)

            # Verify environment variables were applied
            self.assertEqual(autotuner.params.n_trials, 7)
            self.assertEqual(autotuner.params.sampler_name, "random")
            self.assertEqual(autotuner.params.max_configs_ahead, 15)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_without_precompile(self) -> None:
        """Test OptunaSearch with autotune_precompile disabled."""

        @helion.kernel(autotune_precompile=None, autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64, 64], device=DEVICE)
        bound = add.bind((a, b))

        params = OptunaSearchParams(
            n_trials=3,
            sampler_name="random",
            study_name="test_no_precompile",
            load_if_exists=False,
            show_progress_bar=False,
        )
        autotuner = OptunaSearch(bound, (a, b), params=params)
        best_config = autotuner.autotune()
        self.assertIsNotNone(best_config)

    @skipIfCpu("OptunaSearch requires CUDA for pipelined mode")
    @skipIfRocm("too slow on rocm")
    def test_optuna_multivariate_parallel_sampling(self) -> None:
        """Test that multivariate TPE works in parallel after sequential search space setup.

        This test verifies that:
        1. A sequential trial is run first to establish the search space
        2. Subsequent parallel trials use multivariate TPE (not independent sampling)
        """
        import optuna

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([128, 128], device=DEVICE)
        b = torch.randn([128, 128], device=DEVICE)
        bound = add.bind((a, b))

        # Use TPE sampler with multivariate enabled
        params = OptunaSearchParams(
            n_trials=15,  # 1 sequential + 14 parallel
            sampler_name="tpe",
            sampler_kwargs={"multivariate": True, "n_startup_trials": 10},
            study_name="test_multivariate_parallel",
            load_if_exists=False,
            show_progress_bar=False,
        )
        autotuner = OptunaSearch(bound, (a, b), params=params)
        best_config = autotuner.autotune()
        self.assertIsNotNone(best_config)

        # Verify all trials completed
        study = autotuner.get_study()
        self.assertIsNotNone(study)
        completed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        self.assertEqual(completed, 15)


class TestOptunaMultiStudy(RefEagerTestDisabled, TestCase):
    """Test suite for OptunaSearch multi-study optimization mode."""

    @skipIfCpu("Multi-study OptunaSearch requires CUDA")
    @skipIfRocm("too slow on rocm")
    def test_multi_study_basic(self) -> None:
        """Test basic multi-study optimization with MedianPruner."""

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64, 64], device=DEVICE)
        bound = add.bind((a, b))

        params = OptunaSearchParams(
            n_trials=5,
            n_studies=3,
            meta_pruner_name="median",
            sampler_name="random",  # Use random for faster test
            show_progress_bar=False,
        )
        autotuner = OptunaSearch(bound, (a, b), params=params)
        best_config = autotuner.autotune()

        self.assertIsNotNone(best_config)
        self.assertIsNotNone(autotuner.best_config)
        self.assertTrue(autotuner.best_perf_so_far < float("inf"))

        # Check meta-study has trials
        meta_study = autotuner.get_meta_study()
        self.assertIsNotNone(meta_study)
        self.assertEqual(len(meta_study.trials), 3)

    @skipIfCpu("Multi-study OptunaSearch requires CUDA")
    @skipIfRocm("too slow on rocm")
    def test_multi_study_with_pruning(self) -> None:
        """Test that meta-study pruning works (underperforming studies are stopped)."""
        import optuna

        @helion.kernel(autotune_log_level=0)
        def mul(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] * b[tile]
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64, 64], device=DEVICE)
        bound = mul.bind((a, b))

        # Use MedianPruner with aggressive settings to trigger pruning
        params = OptunaSearchParams(
            n_trials=10,
            n_studies=5,
            meta_pruner_name="median",
            meta_pruner_kwargs={"n_startup_trials": 2, "n_warmup_steps": 2},
            sampler_name="random",
            show_progress_bar=False,
            report_interval=1,
        )
        autotuner = OptunaSearch(bound, (a, b), params=params)
        best_config = autotuner.autotune()

        self.assertIsNotNone(best_config)

        # Check that at least one study may have been pruned (not guaranteed)
        meta_study = autotuner.get_meta_study()
        self.assertIsNotNone(meta_study)

        # Count trial states
        completed = sum(
            1
            for t in meta_study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        pruned = sum(
            1
            for t in meta_study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        )

        # Should have some completed or pruned trials
        self.assertGreater(completed + pruned, 0)

    @skipIfCpu("Multi-study OptunaSearch requires CUDA")
    @skipIfRocm("too slow on rocm")
    def test_multi_study_tpe_sampler(self) -> None:
        """Test multi-study optimization with TPE sampler for inner studies."""

        @helion.kernel(autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn([64, 64], device=DEVICE)
        b = torch.randn([64, 64], device=DEVICE)
        bound = add.bind((a, b))

        params = OptunaSearchParams(
            n_trials=5,
            n_studies=2,
            meta_pruner_name=None,  # No pruning
            sampler_name="tpe",
            sampler_kwargs={"multivariate": True},
            show_progress_bar=False,
        )
        autotuner = OptunaSearch(bound, (a, b), params=params)
        best_config = autotuner.autotune()

        self.assertIsNotNone(best_config)


if __name__ == "__main__":
    unittest.main()
