# BALL Python bindings (pybind11)

Minimal Python access to BALL's force-field evaluation surface. Wraps
just enough to compute single-point AMBER and CHARMM-EEF1 energies on
a PDB and return per-component breakdowns.

Originating need: serve as an oracle source for the
[proteon](https://github.com/theGreatHerrLebert/proteon) EVIDENT
manifest, replacing the frozen `BALL_CRAMBIN_RAW` reference values in
`tests/oracle/test_ball_energy.py` with live regeneration, and
unblocking the open `forcefield_charmm19_internal` claim by enabling a
CHARMM-EEF1 component oracle.

## Surface

```python
import ball

amber = ball.amber_energy("crambin.pdb")
# -> {bond_stretch, angle_bend, torsion, nonbonded, total, n_atoms}

charmm = ball.charmm_energy("crambin.pdb", use_eef1=True)
# -> {bond_stretch, angle_bend, torsion, improper_torsion,
#     vdw, nonbonded, solvation, total, n_atoms, use_eef1}
```

`nonbonded_cutoff` defaults to `1e6` (effectively NoCutoff), matching
the convention proteon uses on its own side of the comparison.

## Build

```bash
# from the repo root, after Qt6 / Boost / Eigen3 / FFTW3 dev libs are
# installed (libqt6base-dev libqt6core5compat6-dev libboost-all-dev
# libeigen3-dev libfftw3-dev on Ubuntu 22.04 / Debian bookworm):

mkdir build && cd build
cmake -DBALL_BUILD_PYTHON=ON ..
cmake --build . -j

# ball.cpython-3X-x86_64-linux-gnu.so lands in build/python/.
# At runtime the bindings need three env vars set:
#   PYTHONPATH       — find the .so
#   LD_LIBRARY_PATH  — find libBALL.so
#   BALL_DATA_PATH   — find the force-field parameter files
#                      (Amber/, CHARMM/, MMFF94/, ... — these live
#                       at the repo's `data/` directory)
PYTHONPATH=build/python \
LD_LIBRARY_PATH=build/lib \
BALL_DATA_PATH=$(pwd)/../data \
    python -c 'import ball; print(ball.amber_energy.__doc__)'
```

The wheel build (`pip install .` via scikit-build-core, see
[`pyproject.toml`](../pyproject.toml)) bundles the data files into
the wheel and resolves `BALL_DATA_PATH` at install time, so end
users of the published wheel do not set env vars manually.

The pybind11 dependency is fetched via CMake `FetchContent` at
configure time (pinned to `v2.13.6`); no system pybind11 install
required.

## Test

```bash
cd build
PYTHONPATH=python ctest -R python_smoke --output-on-failure
# or directly:
PYTHONPATH=python python -m pytest ../python/tests -v
```

The smoke tests look for a `1crn.pdb` fixture in this order:

1. `$BALL_TEST_CRAMBIN` env var (if set)
2. `/scratch/TMAlign/proteon/test-pdbs/1crn.pdb` (the proteon staging-parent layout)
3. `<repo>/test/data/1crn.pdb`
4. `<repo>/data/1crn.pdb`

Tests skip rather than fail when no fixture is locatable.

## Known issues

These are component-level mapping bugs in the current bindings, not
pipeline issues. The Python-side smoke test catches them via
`pytest.approx`-style assertions in proteon's oracle test layer; here
they are documented for whoever picks up the next iteration:

- **`angle_bend = 0` on both AMBER and CHARMM.** Real bend terms on
  crambin should be tens of kJ/mol. Likely an option flag at
  `setup()` that gates bend-term registration; the binding currently
  passes only the cutoff options.
- **CHARMM `torsion == improper_torsion`.** Both report 39.7 kJ/mol
  on crambin in current testing — they should differ. Either
  `getTorsionEnergy()` already includes impropers (in which case the
  dict key is misleading) or one of the getters is reading the wrong
  component sum. Needs a quick read of BALL's CHARMM force-field
  initialisation to confirm.

## Out of scope (for now)

- Minimization, dynamics, Monte Carlo
- NMR, QSAR, docking
- Trajectory I/O (DCD, XTC)
- Force-field parameter introspection
- Class-level bindings (e.g. `ball.System`, `ball.Atom`); the current
  binding is functional only

These can land later if a downstream consumer needs them. The current
binding is sized to one job — sourcing oracle energies for proteon —
and stays narrow on purpose. New bindings ship through the same wheel
pipeline (`pyproject.toml` → scikit-build-core → cibuildwheel CI) so
adding them is a small per-PR cost.

## Why pybind11 and not the original SIP bindings

The original SIP bindings under `source/PYTHON/EXTENSIONS/` were
removed in the zomball modernization line (commits 1–5 on this fork).
They were unmaintained, did not build with modern Qt/Boost, and
covered far more surface than any current consumer needs. Restarting
on pybind11 against the modernized C++ tree is cheaper than reviving
SIP and produces a binding that matches today's Python packaging
conventions.
