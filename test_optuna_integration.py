#!/usr/bin/env python3
"""Simple integration test for OptunaSearch autotuner.

This script tests that OptunaSearch can be imported, instantiated, and run
on a simple kernel. It doesn't require pytest and can be run standalone.

Usage:
    python test_optuna_integration.py
"""

from __future__ import annotations

import os
import sys

import torch

import helion
import helion.language as hl
from helion.autotuner import OptunaSearch
from helion.autotuner import OptunaSearchParams


def test_optuna_import():
    """Test that OptunaSearch can be imported."""
    print("✓ OptunaSearch imported successfully")


def test_optuna_basic_kernel():
    """Test OptunaSearch on a simple vector addition kernel."""
    print("\nTesting OptunaSearch on vector addition kernel...")

    # Define a simple kernel
    @helion.kernel()
    def add(a, b):
        out = torch.empty_like(a)
        for tile in hl.tile(out.size()):
            out[tile] = a[tile] + b[tile]
        return out

    # Create test inputs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠ Warning: CUDA not available, using CPU (may be slow)")

    a = torch.randn([128, 128], device=device)
    b = torch.randn([128, 128], device=device)

    # Test with environment variable
    print("\n1. Testing via HELION_AUTOTUNER environment variable...")
    os.environ["HELION_AUTOTUNER"] = "OptunaSearch"
    os.environ["HELION_AUTOTUNE_OPTUNA_TRIALS"] = "5"  # Small number for quick test

    result = add(a, b)
    expected = a + b
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ OptunaSearch via env var works correctly")

    # Test direct instantiation
    print("\n2. Testing direct instantiation...")
    del os.environ["HELION_AUTOTUNER"]

    # Bind the kernel
    bound = add.bind((a, b))

    # Create OptunaSearch directly
    params = OptunaSearchParams(
        n_trials=5,
        sampler="tpe",
        show_progress_bar=False,  # Don't clutter output
    )
    autotuner = OptunaSearch(bound, (a, b), params=params)

    # Run autotuning
    best_config = autotuner.autotune()
    print(f"✓ Best config found: {best_config}")

    # Verify the study is accessible
    study = autotuner.get_study()
    assert study is not None, "Study should not be None after autotuning"
    assert len(study.trials) == 5, f"Expected 5 trials, got {len(study.trials)}"
    print(f"✓ Study contains {len(study.trials)} trials")

    # Test different samplers
    print("\n3. Testing different samplers...")
    for sampler_name in ["random", "tpe"]:
        print(f"   Testing {sampler_name} sampler...")
        params = OptunaSearchParams(
            n_trials=3,
            sampler=sampler_name,
            show_progress_bar=False,
        )
        autotuner = OptunaSearch(bound, (a, b), params=params)
        config = autotuner.autotune()
        print(f"   ✓ {sampler_name} sampler works")

    print("\n✓ All OptunaSearch tests passed!")


def test_optuna_persistence():
    """Test that Optuna study persistence works."""
    print("\n4. Testing study persistence...")

    import tempfile

    # Create a simple kernel
    @helion.kernel()
    def mul(a, b):
        out = torch.empty_like(a)
        for tile in hl.tile(out.size()):
            out[tile] = a[tile] * b[tile]
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn([64, 64], device=device)
    b = torch.randn([64, 64], device=device)

    bound = mul.bind((a, b))

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        storage = f"sqlite:///{db_path}"

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
        print(f"   First run: {n_trials_1} trials")

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
        print(f"   Second run: {n_trials_2} trials total")

        assert n_trials_2 == n_trials_1 + 2, f"Expected {n_trials_1 + 2} trials, got {n_trials_2}"
        print("✓ Study persistence works correctly")

    finally:
        # Cleanup
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    """Run all tests."""
    print("=" * 60)
    print("OptunaSearch Integration Tests")
    print("=" * 60)

    try:
        test_optuna_import()
        test_optuna_basic_kernel()
        test_optuna_persistence()
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
