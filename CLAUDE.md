# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this fork is

This is `theGreatHerrLebert/ball` — a fork of BALL (the Biochemical
ALgorithms Library) whose purpose is the **`ball-py` Python package**:
a minimal pybind11 binding layer over BALL's force-field, format, and
structure surface, published to PyPI as `ball-py`.

The fork is **replanted on upstream `BALL-Project/ball` v1.6.0**. The
upstream v1.6 line is the canonical, modernized C++/CMake/Qt codebase
— do not re-modernize it here. All fork-specific work lives under
`python/` plus the wheel/CI glue; the C++ library is upstream's.

To pull future upstream releases, rebase the `python/*` commits onto
the new tag rather than merging — the binding commits are a thin,
portable layer and a merge would re-introduce conflicts.

## Layout of fork-specific work

- `python/src/module.cpp` — the pybind11 module. Exposes ~12
  functions: `amber_energy`, `charmm_energy`, `mmff94_energy`,
  `sasa`, `rmsd`, `minimize_energy`, `build_bonds`, `atom_typer`,
  `hbonds`, `secondary_structure`, `add_hydrogens_to_pdb`,
  `system_info`.
- `python/CMakeLists.txt` — opt-in subdirectory, gated by the
  top-level `-DBALL_BUILD_PYTHON=ON` option. Fetches pybind11
  v2.13.6 via FetchContent.
- `python/tests/test_smoke.py` — pytest smoke tests, also wired as
  the `python_smoke` CTest target.
- `pyproject.toml` + `.github/workflows/` — scikit-build-core wheel
  build and CI.

## Build (bindings only, no Qt GUI)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBALL_HAS_VIEW=OFF \        # skip the Qt/OpenGL GUI library
  -DBALL_PYTHON_SUPPORT=OFF \  # skip the legacy SIP bindings
  -DBALL_BUILD_PYTHON=ON \     # build python/ pybind11 bindings
  -DCMAKE_CXX_STANDARD=17
cmake --build build --target ball_py -j"$(nproc)"
```

Produces `build/lib/libBALL.so.1.6` and
`build/python/ball.cpython-*.so`.

## Test

```bash
LD_LIBRARY_PATH=build/lib \
PYTHONPATH=build/python \
BALL_DATA_PATH=data \
python3 -m pytest python/tests -v
```

## Notes on BALL as an oracle

`ball-py` exists so proteon can validate force-field energies against
an independent reference. BALL has its own numerical quirks — e.g.
`RMSDMinimizer` leaves a ~5.4e-3 Å residual on identical input (its
Kabsch self-floor). Such properties are documented inline in the
tests; do not write tolerances tighter than BALL's real behavior.
