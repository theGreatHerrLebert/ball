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
PYTHONPATH=build/python python -c 'import ball; print(ball.amber_energy.__doc__)'
```

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

## Out of scope (for now)

- Minimization, dynamics, Monte Carlo
- NMR, QSAR, docking
- Trajectory I/O (DCD, XTC)
- Force-field parameter introspection
- pip-installable wheel build (`pyproject.toml` + scikit-build-core)

These can land later if a downstream consumer needs them. The current
binding is sized to one job — sourcing oracle energies for proteon —
and stays narrow on purpose.

## Why pybind11 and not the original SIP bindings

The original SIP bindings under `source/PYTHON/EXTENSIONS/` were
removed in the zomball modernization line (commits 1–5 on this fork).
They were unmaintained, did not build with modern Qt/Boost, and
covered far more surface than any current consumer needs. Restarting
on pybind11 against the modernized C++ tree is cheaper than reviving
SIP and produces a binding that matches today's Python packaging
conventions.
