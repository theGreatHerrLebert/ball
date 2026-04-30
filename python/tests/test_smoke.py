"""Smoke tests for the BALL Python bindings.

These tests assume the `ball` extension module is on `sys.path`. The
parent `python/CMakeLists.txt` arranges that via `add_test` — when run
manually, set `PYTHONPATH=<build>/python` first.

The tests verify shape and sanity, not numerical agreement: we want to
catch a build that imports zero/NaN/inf, not pin specific kJ/mol values
here. Numerical pinning lives in proteon's
tests/oracle/test_ball_energy.py.
"""

from __future__ import annotations

import math
import os
import pathlib

import pytest

import ball


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _resolve_crambin_pdb() -> pathlib.Path:
    """Locate a crambin (1crn) PDB usable as a fixture.

    Prefer the proteon repo's test-pdbs/1crn.pdb when this checkout
    lives next to a proteon clone (the staging-parent layout on
    /scratch/TMAlign/). Fall back to BALL's own test data if the fork
    ships one. Skip the test rather than fail when no fixture is
    findable — environments without proteon should not block CI.
    """
    candidates = [
        pathlib.Path("/scratch/TMAlign/proteon/test-pdbs/1crn.pdb"),
        REPO_ROOT / "test" / "data" / "1crn.pdb",
        REPO_ROOT / "data" / "1crn.pdb",
    ]
    env_path = os.environ.get("BALL_TEST_CRAMBIN")
    if env_path:
        candidates.insert(0, pathlib.Path(env_path))
    for p in candidates:
        if p.is_file():
            return p
    pytest.skip(
        "No 1crn.pdb fixture located; set BALL_TEST_CRAMBIN to override"
    )


def test_module_loads():
    """The C extension imports and exposes the expected symbols."""
    assert hasattr(ball, "amber_energy")
    assert hasattr(ball, "charmm_energy")
    assert callable(ball.amber_energy)
    assert callable(ball.charmm_energy)


def test_amber_energy_returns_finite_components():
    """AMBER on crambin produces finite, non-zero energies for every component."""
    pdb = _resolve_crambin_pdb()
    e = ball.amber_energy(str(pdb))

    expected_keys = {
        "bond_stretch", "angle_bend", "torsion",
        "nonbonded", "total", "n_atoms",
    }
    assert expected_keys.issubset(e.keys()), (
        f"missing keys: {expected_keys - set(e.keys())}"
    )

    assert e["n_atoms"] > 0, "PDB load reported zero atoms"

    for k in ("bond_stretch", "angle_bend", "torsion", "nonbonded", "total"):
        v = e[k]
        assert isinstance(v, float), f"{k}: expected float, got {type(v)}"
        assert math.isfinite(v), f"{k}: non-finite value {v!r}"


def test_charmm_eef1_returns_finite_components():
    """CHARMM+EEF1 on crambin produces a finite solvation term."""
    pdb = _resolve_crambin_pdb()
    e = ball.charmm_energy(str(pdb), use_eef1=True)

    expected_keys = {
        "bond_stretch", "angle_bend", "torsion", "improper_torsion",
        "vdw", "nonbonded", "solvation", "total", "n_atoms", "use_eef1",
    }
    assert expected_keys.issubset(e.keys()), (
        f"missing keys: {expected_keys - set(e.keys())}"
    )

    assert e["use_eef1"] is True
    assert math.isfinite(e["solvation"])
    # EEF1 solvation on a folded protein should be non-zero — a zero
    # would mean the EEF1 toggle silently failed to engage.
    assert e["solvation"] != 0.0, (
        "EEF1 solvation reported as 0.0 — option may not have applied"
    )


def test_charmm_no_eef1_zeroes_solvation():
    """Disabling EEF1 produces zero solvation; bonded terms unaffected."""
    pdb = _resolve_crambin_pdb()
    with_eef1 = ball.charmm_energy(str(pdb), use_eef1=True)
    no_eef1 = ball.charmm_energy(str(pdb), use_eef1=False)

    assert no_eef1["use_eef1"] is False
    assert no_eef1["solvation"] == pytest.approx(0.0, abs=1e-6)

    # Bonded components should agree to floating-point precision —
    # EEF1 only adds a solvation term, doesn't alter bonded math.
    for k in ("bond_stretch", "angle_bend", "torsion", "improper_torsion"):
        assert no_eef1[k] == pytest.approx(with_eef1[k], rel=1e-9), (
            f"{k} drifted between EEF1 on/off; EEF1 should not affect bonded"
        )


def test_load_failure_raises():
    """A nonexistent PDB path raises rather than returning empty energies."""
    with pytest.raises(RuntimeError, match="cannot open|failed to parse"):
        ball.amber_energy("/nonexistent/path/to/no.pdb")
