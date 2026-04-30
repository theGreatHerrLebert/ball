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


def test_ball_data_path_auto_discovery():
    """AmberFF setup succeeds without the user pre-setting BALL_DATA_PATH.

    Regression guard for the in-binding BALL_DATA_PATH discovery: at
    PYBIND11_MODULE init time, the binding dladdrs its own .so to find
    the wheel-bundled share/BALL/ tree and setenv()s BALL_DATA_PATH if
    the user has not already set it. Without that, every data-dependent
    BALL call (FragmentDB construction, AmberFF::setup, RMSDMinimizer,
    ...) fails with the opaque "BALL: std::exception" because libBALL
    cannot find Fragments.db / parameter INIs.

    Note: Python's os.environ is a snapshot at process start and
    does NOT reflect the C-side setenv. The proof that the env was
    set is that amber_energy actually succeeds — without
    BALL_DATA_PATH, AmberFF::setup throws and the call here would
    raise RuntimeError("BALL: std::exception") before producing a
    finite total.
    """
    pdb = _resolve_crambin_pdb()
    e = ball.amber_energy(str(pdb))
    assert math.isfinite(e["total"]), (
        "amber_energy returned non-finite total — AmberFF::setup probably "
        "could not find BALL data dir; check the dladdr-based discovery "
        "in module.cpp"
    )
    assert e["n_atoms"] >= 327, "PDB load lost atoms"


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


def test_mmff94_energy_either_runs_or_fails_cleanly():
    """MMFF94 on crambin: either finite components, or a clean RuntimeError.

    MMFF94 is a small-molecule force field; its atom typer is known to
    struggle with polypeptide backbones. Accept both outcomes so the
    smoke test passes regardless of how BALL's MMFF94 implementation
    happens to type proteins on this fixture — the binding's job is
    to expose the function and translate exceptions, not to guarantee
    coverage of a use case the FF wasn't designed for.

    Override BALL_TEST_LIGAND with a small-molecule PDB to exercise the
    happy path on a real ligand fixture; otherwise we use crambin and
    relax the success path.
    """
    ligand_env = os.environ.get("BALL_TEST_LIGAND")
    if ligand_env and pathlib.Path(ligand_env).is_file():
        pdb = pathlib.Path(ligand_env)
        # On a real ligand, MMFF94 must succeed.
        e = ball.mmff94_energy(str(pdb))
        for k in ("stretch", "bend", "stretch_bend", "torsion", "plane",
                  "vdw", "electrostatic", "nonbonded", "total"):
            assert math.isfinite(e[k]), f"{k}: non-finite value {e[k]!r}"
        # nonbonded should equal vdw + electrostatic to floating-point
        # precision (BALL's split-vs-sum invariant).
        assert e["nonbonded"] == pytest.approx(
            e["vdw"] + e["electrostatic"], rel=1e-6, abs=1e-6
        )
        return

    pdb = _resolve_crambin_pdb()
    try:
        e = ball.mmff94_energy(str(pdb))
    except RuntimeError as exc:
        # The documented failure mode — surface the error string so a
        # silent regression doesn't hide behind it.
        assert "MMFF94" in str(exc), (
            f"unexpected RuntimeError text: {exc}"
        )
        return
    # If MMFF94 *did* type the protein, the components must be finite.
    for k in ("stretch", "bend", "stretch_bend", "torsion", "plane",
              "vdw", "electrostatic", "nonbonded", "total"):
        assert math.isfinite(e[k]), f"{k}: non-finite value {e[k]!r}"


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


# ---------------------------------------------------------------------------
# Analysis surface — sasa, build_bonds, atom_typer, minimize_energy,
# system_info. Added in v0.1.0a2.
# ---------------------------------------------------------------------------

def test_system_info_shape():
    """system_info enumerates atoms / residues / chains."""
    pdb = _resolve_crambin_pdb()
    info = ball.system_info(str(pdb))
    assert info["n_atoms"] > 0
    assert info["n_residues"] >= 40  # crambin = 46 residues
    assert info["n_chains"] >= 1
    assert isinstance(info["chain_ids"], list)
    assert all(isinstance(c, str) and c for c in info["chain_ids"])


def test_build_bonds_counts():
    """build_bonds infers bonds geometrically and reports the count."""
    pdb = _resolve_crambin_pdb()
    info = ball.build_bonds(str(pdb))
    assert info["n_atoms"] > 0
    # On crambin's heavy-atom-only set (327 atoms, ~46 residues),
    # geometric inference yields ~330 bonds (backbone + sidechain).
    # The exact number depends on the bond-length cutoff but should
    # be in the same ballpark as the atom count.
    assert info["n_bonds_built"] > 100
    assert info["n_bonds_built"] < 1500


def test_sasa_total_and_per_atom():
    """sasa produces non-zero total area and a per-atom dict."""
    pdb = _resolve_crambin_pdb()
    s = ball.sasa(str(pdb))
    assert s["total_area"] > 0.0, "SASA returned 0 — radii assignment may have failed"
    assert s["total_volume"] > 0.0
    assert s["n_atoms"] > 0
    # Crambin total SASA is ~3000 A^2 with PARSE radii.
    assert 2000.0 < s["total_area"] < 5000.0, (
        f"SASA total {s['total_area']} A^2 outside plausible range for crambin"
    )
    assert isinstance(s["per_atom"], dict)
    assert len(s["per_atom"]) > 0
    # Every per-atom area is non-negative; at least some are non-zero.
    nonzero_count = sum(1 for v in s["per_atom"].values() if v > 0)
    assert nonzero_count > 50, "almost no atoms have non-zero SASA"


def test_atom_typer_assigns_charges():
    """atom_typer returns one entry per atom with type + charge + element."""
    pdb = _resolve_crambin_pdb()
    types = ball.atom_typer(str(pdb))
    assert isinstance(types, list)
    assert len(types) > 0
    for entry in types[:5]:
        assert {"atom", "type_name", "charge", "element"} == entry.keys()
        assert ":" in entry["atom"]
        assert isinstance(entry["charge"], float)
        assert isinstance(entry["element"], str) and entry["element"]
    # Most atoms should have non-empty type names; allow a small fraction
    # of misses for terminus residues.
    typed_count = sum(1 for t in types if t["type_name"])
    assert typed_count >= 0.8 * len(types), (
        f"only {typed_count}/{len(types)} atoms got AMBER type names"
    )


def test_rmsd_self_is_zero():
    """RMSD(crambin, crambin) ≈ 0 across all pairing modes.

    Both with and without superposition, comparing a structure to itself
    must yield zero (modulo numerical noise from the Kabsch eigenvalue
    solve). A non-zero RMSD here would indicate either AtomBijection
    paired wrong atoms (e.g. shifted by one) or RMSDMinimizer's transform
    solver is mis-applying.
    """
    pdb = _resolve_crambin_pdb()
    for mode in ("ca", "backbone", "name", "all"):
        for superpose in (True, False):
            r = ball.rmsd(str(pdb), str(pdb), atoms=mode, superpose=superpose)
            assert r["rmsd"] == pytest.approx(0.0, abs=1e-4), (
                f"RMSD self-comparison non-zero in mode={mode}, "
                f"superpose={superpose}: {r['rmsd']}"
            )
            assert r["atoms"] == mode
            assert r["superpose"] is superpose
            assert r["n_matched"] >= 3, (
                f"only {r['n_matched']} atoms paired in mode={mode}"
            )
            assert r["n_atoms_a"] == r["n_atoms_b"]


def test_rmsd_after_minimization_is_nonzero(tmp_path):
    """RMSD(crambin, minimized_crambin) is positive but small.

    Minimizing crambin moves atoms by < ~3 A on average — large enough
    to be unmistakably non-zero but small enough that any double-digit
    result indicates the bijection paired wrong atoms or the structure
    was silently corrupted.
    """
    pdb = _resolve_crambin_pdb()
    minimized = tmp_path / "crambin_min.pdb"
    ball.minimize_energy(str(pdb), str(minimized), max_iter=20)

    # CA-RMSD is the canonical metric; superpose=True removes any
    # rigid-body drift introduced by the minimizer's coordinate frame.
    r = ball.rmsd(str(pdb), str(minimized), atoms="ca", superpose=True)
    assert r["rmsd"] > 0.0, "RMSD between original and minimized is exactly 0"
    assert r["rmsd"] < 5.0, (
        f"CA-RMSD after 20 minimization steps is {r['rmsd']} A — far larger "
        f"than expected, suggests either atom-pairing failure or structural "
        f"corruption"
    )

    # Without superposition the RMSD must be >= the superposed value
    # (Kabsch is a minimum over all rigid motions).
    r_noalign = ball.rmsd(str(pdb), str(minimized), atoms="ca", superpose=False)
    assert r_noalign["rmsd"] >= r["rmsd"] - 1e-9, (
        f"non-superposed RMSD ({r_noalign['rmsd']}) is smaller than "
        f"Kabsch-aligned RMSD ({r['rmsd']}) — Kabsch should be the minimum"
    )


def test_rmsd_unknown_mode_raises():
    """Unknown atoms mode is rejected with a clear message."""
    pdb = _resolve_crambin_pdb()
    with pytest.raises(RuntimeError, match="unknown atoms mode"):
        ball.rmsd(str(pdb), str(pdb), atoms="xyz")


def test_minimize_energy_drops_energy(tmp_path):
    """minimize_energy actually iterates and produces an energy drop."""
    pdb = _resolve_crambin_pdb()
    out_path = tmp_path / "crambin_min.pdb"
    result = ball.minimize_energy(str(pdb), str(out_path), max_iter=50)
    expected_keys = {
        "ff", "method", "max_iter", "iterations", "converged",
        "initial_energy", "final_energy", "energy_drop",
        "pdb_out", "n_atoms",
    }
    assert expected_keys.issubset(result.keys())

    # The minimizer must have actually iterated. Defaulting to 0
    # iterations was the earlier silent-failure mode.
    assert result["iterations"] > 0, (
        "minimizer ran zero iterations — likely a setMaxNumberOfIterations "
        "vs minimize(steps=) bug"
    )
    # Energy must drop on a raw clashy PDB (initial energy is huge).
    assert result["energy_drop"] > 0, (
        f"energy did not drop: initial={result['initial_energy']}, "
        f"final={result['final_energy']}"
    )
    # PDB written.
    assert out_path.is_file()
