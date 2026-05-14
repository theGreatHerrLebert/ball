# Requirements: BALLView 1.6 Modernization

**Defined:** 2026-05-14
**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies.

## v1 Requirements

Requirements for the v1.6 release. Derived from `/Users/kohlbach/Claude/BALL/ROADMAP-1.6.md`.

### Build Baseline

- [ ] **BUILD-01**: BALL/VIEW/BALLView configure and build on macOS (Apple Silicon) from Homebrew dependencies with a documented command
- [ ] **BUILD-02**: The 8 modern-toolchain patches are committed to the repo with clear messages
- [ ] **BUILD-03**: Project version is bumped to 1.6.0-dev in the CMake project declaration
- [ ] **BUILD-04**: A `BUILD-macos.md` documents the dependency install and build/run flow

### Language Modernization

- [ ] **LANG-01**: The codebase compiles under C++17 (the C++14 bridge flag is removed)
- [ ] **LANG-02**: All uses of C++17-removed constructs (`std::unary_function`, `binary_function`, `bind2nd`, `bind1st`, `ptr_fun`, `auto_ptr`, `mem_fun`) are replaced across the 7 known files
- [ ] **LANG-03**: C++ standard is set via `CMAKE_CXX_STANDARD`/`target_compile_features`, not raw `-std=` flags

### Dependency System

- [ ] **DEPS-01**: `ball_contrib` is removed from the build path; CMake finds dependencies via system/Homebrew packages
- [ ] **DEPS-02**: A vcpkg manifest provides the Windows dependency set
- [ ] **DEPS-03**: Stale bundled `Find*.cmake` modules (Boost, TBB, Eigen3, OpenBabel) are replaced with config-mode `find_package` where upstream provides it
- [ ] **DEPS-04**: Minimum dependency versions are pinned and documented

### Rendering — Phase 4a (IMMEDIATE PRIORITY)

- [ ] **RENDER-01**: `GLRenderWindow` derives from `QOpenGLWidget` (not the removed-in-Qt6 `QGLWidget`); `QGLFormat` is replaced by `QSurfaceFormat` requesting a compatibility profile + depth/stencil/double buffer
- [ ] **RENDER-02**: BALLView displays a molecule in the embedded 3D scene on macOS (Apple Silicon) — the scene widget renders inside the main window, not a detached native window
- [ ] **RENDER-03**: The interactive GL renderer runs on the GUI thread via `paintGL()`; manual `swapBuffers()`/`setAutoBufferSwap` are removed
- [ ] **RENDER-04**: The raytracer renderer continues to work via its CPU pixel-buffer path, blitted to the `QOpenGLWidget` as a texture
- [ ] **RENDER-05**: On-screen text rendering (formerly `QGLWidget::renderText`) is reimplemented via a `QPainter` overlay
- [ ] **RENDER-06**: Mouse/keyboard interaction (rotate, zoom, pick, selection) works in the ported scene widget
- [ ] **RENDER-07**: `RenderSetup`, `scene.C`, `glOffscreenTarget.C`, `glRenderer.C` are updated to remove all `QGLWidget`/`QGLContext`/`QGLFormat` references and compile cleanly
- [ ] **RENDER-08**: BALLView builds, launches, and renders a molecule on Linux and Windows (platform-independence verified, no regressions vs. macOS)

### Qt 6 + Pipeline

- [ ] **QT6-01**: BALLView builds against Qt 6, with `QGLWidget`-era APIs fully removed
- [ ] **QT6-02**: Qt-deprecated APIs in VIEW (`QRegExp`, `QDesktopWidget`) are replaced with Qt 6 equivalents
- [ ] **QT6-03**: The rendering pipeline is modernized off fixed-function GL (GL core profile or QRhi — decided in Phase 4b)

### Python Bindings

- [ ] **PY-01**: BALL's Python bindings build against a supported Python (3.12+) using a modern binding generator (SIP 6 or pybind11/nanobind)

### Networking

- [ ] **NET-01**: `TCPServer`/`TCPServerThread` are reworked onto the modern Boost.Asio model (or Qt networking) and covered by a unit test

### Packaging

- [ ] **PKG-01**: `BALLView.app` embeds its `data/` directory into `Contents/Resources` and launches by double-click without environment variables set
- [ ] **PKG-02**: The macOS build produces a `macdeployqt`-processed, notarizable universal (arm64 + x86_64) bundle

### CI & Tests

- [ ] **CI-01**: A GitHub Actions matrix builds BALL/VIEW/BALLView on macOS-arm64, Linux, and Windows
- [ ] **CI-02**: The `test/` tree is wired into the build and `ctest` runs in CI

## v2 Requirements

Deferred beyond v1.6.

### Pipeline

- **PIPE-01**: Full programmable-pipeline rewrite of `glRenderer.C` off fixed-function GL

## Out of Scope

| Feature | Reason |
|---------|--------|
| Reviving `ball_contrib` source build | Dead end on modern toolchains; replaced by system/Homebrew/vcpkg deps |
| New molecular-modelling features/algorithms | This milestone is modernization only |
| RTfact raytracer revival | Windows-only contrib, not load-bearing |
| Programmable-pipeline GL rewrite | Large; Phase 4a keeps fixed-function via compat profile, full rewrite is v2 |

## Traceability

Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUILD-01..04 | TBD | Pending |
| LANG-01..03 | TBD | Pending |
| DEPS-01..04 | TBD | Pending |
| RENDER-01..08 | TBD | Pending |
| QT6-01..03 | TBD | Pending |
| PY-01 | TBD | Pending |
| NET-01 | TBD | Pending |
| PKG-01..02 | TBD | Pending |
| CI-01..02 | TBD | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 27 ⚠️

---
*Requirements defined: 2026-05-14*
*Last updated: 2026-05-14 after initial definition*
