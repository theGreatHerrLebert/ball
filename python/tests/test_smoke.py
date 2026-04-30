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
    """AMBER on crambin produces finite, non-zero energies for every component.

    The non-zero check on bend and torsion is load-bearing: BALL's
    force-field components silently zero out terms when atom triplets/
    quadruplets cannot match the parameter table. A regression that
    skips name normalisation, bond construction, or hydrogen placement
    on the input PDB shows up exactly as bend = 0 / torsion = 0 here.
    """
    pdb = _resolve_crambin_pdb()
    e = ball.amber_energy(str(pdb))

    expected_keys = {
        "bond_stretch", "angle_bend", "torsion",
        "nonbonded", "total", "n_atoms",
    }
    assert expected_keys.issubset(e.keys()), (
        f"missing keys: {expected_keys - set(e.keys())}"
    )

    # n_atoms after add_hydrogens=True is the heavy-atom count plus
    # placed hydrogens; on crambin (327 heavy atoms) the all-atom
    # AMBER FF lands around 600+. A zero or near-zero count signals
    # the PDB load itself failed.
    assert e["n_atoms"] >= 327, "PDB load lost atoms"

    for k in ("bond_stretch", "angle_bend", "torsion", "nonbonded", "total"):
        v = e[k]
        assert isinstance(v, float), f"{k}: expected float, got {type(v)}"
        assert math.isfinite(v), f"{k}: non-finite value {v!r}"

    for k in ("bond_stretch", "angle_bend", "torsion"):
        assert abs(e[k]) > 1.0, (
            f"{k} is near-zero ({e[k]} kJ/mol) — preprocessing chain may "
            f"have skipped name normalisation, bond build, or H placement"
        )


def test_charmm_eef1_returns_finite_components():
    """CHARMM+EEF1 on crambin: every component populates non-trivially.

    Splits proper from improper torsion (BALL's CharmmFF::getTorsionEnergy
    returns the sum, which previously masked proper torsion always being 0
    on heavy-atom-only inputs).
    """
    pdb = _resolve_crambin_pdb()
    e = ball.charmm_energy(str(pdb), use_eef1=True)

    expected_keys = {
        "bond_stretch", "angle_bend",
        "proper_torsion", "improper_torsion", "torsion",
        "vdw", "nonbonded", "solvation", "total", "n_atoms", "use_eef1",
    }
    assert expected_keys.issubset(e.keys()), (
        f"missing keys: {expected_keys - set(e.keys())}"
    )

    assert e["use_eef1"] is True
    for k in ("bond_stretch", "angle_bend", "proper_torsion",
              "improper_torsion", "vdw", "nonbonded", "solvation", "total"):
        assert math.isfinite(e[k]), f"{k}: non-finite value {e[k]!r}"

    # The torsion key is the documented sum of proper + improper.
    assert e["torsion"] == pytest.approx(
        e["proper_torsion"] + e["improper_torsion"], rel=1e-9
    ), "torsion key drifted from proper+improper sum"

    # EEF1 solvation on a folded protein should be non-zero AND
    # negative (favorable solvation in a folded state). A zero would
    # mean the EEF1 toggle silently failed to engage; a positive
    # value would mean the FF saw an unfolded conformation, which
    # is wrong for crambin.
    assert e["solvation"] < 0.0, (
        f"EEF1 solvation reported as {e['solvation']} — should be negative on a folded protein"
    )

    # Both proper and improper torsion should be non-trivial on a
    # protein. Improper specifically catches the previous bug where
    # only impropers fired and they masked as the total torsion.
    assert abs(e["proper_torsion"]) > 1.0, (
        "proper_torsion is near-zero — preprocessing chain may have skipped H placement"
    )
    assert abs(e["improper_torsion"]) > 1.0, (
        "improper_torsion is near-zero — CharmmImproperTorsion setup may have failed"
    )

    # Folded crambin under CHARMM19+EEF1 should report a negative total
    # energy (favorable). If this flips positive, something is very
    # wrong with the FF assignment.
    assert e["total"] < 0.0, (
        f"CHARMM+EEF1 total energy on folded crambin should be negative, got {e['total']}"
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
    for k in ("bond_stretch", "angle_bend",
              "proper_torsion", "improper_torsion"):
        assert no_eef1[k] == pytest.approx(with_eef1[k], rel=1e-9), (
            f"{k} drifted between EEF1 on/off; EEF1 should not affect bonded"
        )


def test_disable_preprocessing_surfaces_zero_bend():
    """Without preprocessing, bend and torsion zero out — the previous bug.

    This is a regression guard: if a future BALL change starts auto-running
    name normalisation / bond building / H placement during setup(), this
    test will start passing the inverted assertion (bend != 0) and we
    should re-evaluate whether the preprocessing flags still belong on the
    binding surface or can become no-ops.
    """
    pdb = _resolve_crambin_pdb()
    e = ball.amber_energy(
        str(pdb),
        normalize_names=False,
        build_bonds=False,
        add_hydrogens=False,
    )
    # On the heavy-atom-only crambin, no preprocessing -> no bonds
    # built -> CharmmBend / AmberBend can't find triplets -> bend = 0.
    assert e["angle_bend"] == pytest.approx(0.0, abs=1e-9), (
        "angle_bend non-zero without preprocessing — BALL's setup may have "
        "started doing the preprocessing itself, in which case the "
        "preprocessing flags on the binding may no longer be needed"
    )


def test_load_failure_raises():
    """A nonexistent PDB path raises rather than returning empty energies."""
    with pytest.raises(RuntimeError, match="cannot open|failed to parse"):
        ball.amber_energy("/nonexistent/path/to/no.pdb")


# ---------------------------------------------------------------------------
# Analysis surface — hbonds, secondary_structure, add_hydrogens_to_pdb
# Added in v0.1.0a1. Smoke-level only: shape and order-of-magnitude sanity.
# Numerical pinning lives downstream in proteon's oracle layer.
# ---------------------------------------------------------------------------

def test_hbonds_shape():
    """hbonds returns a list of {donor, acceptor, length} dicts."""
    pdb = _resolve_crambin_pdb()
    bonds = ball.hbonds(str(pdb))
    assert isinstance(bonds, list)
    # Crambin (46 residues) has roughly 25–30 backbone hydrogen bonds —
    # an order-of-magnitude check guards against the processor not
    # actually running.
    assert 10 <= len(bonds) <= 60, f"unexpected H-bond count {len(bonds)}"
    sample = bonds[0]
    assert {"donor", "acceptor", "length"}.issubset(sample.keys())
    # atom_key format: "<chain>:<residue><id>:<atom>"
    assert ":" in sample["donor"]
    assert ":" in sample["acceptor"]
    assert 2.0 < sample["length"] < 5.0, (
        f"H-bond length {sample['length']} outside plausible range"
    )


def test_secondary_structure_shape():
    """secondary_structure returns one assignment per residue with valid SS letters."""
    pdb = _resolve_crambin_pdb()
    result = ball.secondary_structure(str(pdb))
    assert "assignments" in result
    assert "n_residues" in result
    assert result["n_residues"] == len(result["assignments"])
    # Crambin has 46 residues; allow some tolerance for terminus
    # handling differences.
    assert 40 <= result["n_residues"] <= 50

    valid_letters = {"H", "E", "T", "C", "-"}
    for entry in result["assignments"]:
        assert {"residue", "ss"} == entry.keys()
        assert entry["ss"] in valid_letters, (
            f"unexpected SS letter {entry['ss']!r} for {entry['residue']}"
        )

    # Crambin has both helix and strand content; assert that the
    # processor produced a non-trivial mix rather than all-coil.
    counts = {letter: 0 for letter in valid_letters}
    for entry in result["assignments"]:
        counts[entry["ss"]] += 1
    assert counts["H"] > 0, "no helix assigned — processor may not have run"
    assert counts["E"] > 0, "no strand assigned — processor may not have run"


def test_add_hydrogens_writes_pdb(tmp_path):
    """add_hydrogens_to_pdb writes a hydrogenated PDB and reports counts."""
    pdb = _resolve_crambin_pdb()
    out_path = tmp_path / "crambin_h.pdb"
    result = ball.add_hydrogens_to_pdb(str(pdb), str(out_path))

    assert out_path.is_file(), "output PDB was not written"
    assert result["pdb_out"] == str(out_path)
    assert result["n_atoms_in"] > 0
    assert result["n_hydrogens_added"] > 0
    assert result["n_atoms_out"] == result["n_atoms_in"] + result["n_hydrogens_added"]
    # Crambin has 327 heavy atoms; expect ~300 hydrogens for an
    # all-atom force field.
    assert 200 <= result["n_hydrogens_added"] <= 400, (
        f"unexpected H count {result['n_hydrogens_added']}"
    )
