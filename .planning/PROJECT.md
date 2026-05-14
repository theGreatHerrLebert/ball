# BALLView 1.6 Modernization

## What This Is

BALL (Biochemical ALgorithms Library) is a C++ molecular modelling framework, and
BALLView is its OpenGL-based molecular visualization GUI application. This project
modernizes the BALL/VIEW/BALLView stack — frozen at a 2022 commit against ~2016-era
dependencies — so it builds, runs, and renders on current macOS (Apple Silicon),
Linux, and Windows toolchains, culminating in a shippable v1.6 release.

## Core Value

BALLView must **build and visibly render molecules** on macOS, Linux, and Windows
from current, supported dependencies — the 3D scene working cross-platform is the
non-negotiable outcome.

## Requirements

### Validated

<!-- Confirmed working as of this review (2026-05-14), built locally on macOS Tahoe arm64. -->

- ✓ Core `libBALL` builds against modern toolchain (Boost 1.90, Eigen 5, Clang/C++14) — existing+patched
- ✓ `libVIEW` builds against Qt 5.15 — existing+patched
- ✓ `BALLView.app` builds and launches on macOS Tahoe (arm64), runs FragmentDB + AMBER demo — existing+patched

### Active

<!-- v1.6 scope. Mirrors /Users/kohlbach/Claude/BALL/ROADMAP-1.6.md -->

- [ ] **Phase 1** — Land the modern-toolchain build baseline (8 patches already applied locally; needs committing, version bump, BUILD-macos.md)
- [ ] **Phase 2** — Modernize language baseline: move off C++14 bridge to C++17, remove `std::unary_function`/`bind2nd`/etc. (7 known files)
- [ ] **Phase 3** — Dependency system overhaul: delete `ball_contrib`, adopt Homebrew/system + vcpkg, rewrite stale `Find*.cmake` as config-mode
- [ ] **Phase 4a** — **(IMMEDIATE PRIORITY)** Port `GLRenderWindow` from the removed-in-Qt6 `QGLWidget` to `QOpenGLWidget`; hybrid threading model (interactive GL on GUI thread, raytracer stays a CPU-buffer worker thread); keep fixed-function GL via a compatibility profile. Restores the 3D scene on Linux/macOS/Windows.
- [ ] **Phase 4b** — Qt 6 migration + rendering pipeline modernization (GL core profile or QRhi)
- [ ] **Phase 5** — Replace SIP 4 Python bindings (SIP 6 or pybind11/nanobind)
- [ ] **Phase 6** — Rework `TCPServer`/Boost.Asio networking onto the modern Asio model, add tests
- [ ] **Phase 7** — macOS packaging: embed `data/` into the `.app` bundle, wire `macdeployqt`, notarizable universal binary
- [ ] **Phase 8** — CI (GitHub Actions matrix: macOS-arm64/Linux/Windows) + wire up the `test/` tree

### Out of Scope

- Programmable-pipeline GL rewrite of `glRenderer.C` — deferred to post-1.6; Phase 4a keeps fixed-function via a compat profile
- Reviving the `ball_contrib` source-tarball build system — dead end on modern toolchains; replaced by system/Homebrew deps
- New molecular-modelling features or algorithms — this milestone is modernization only
- RTfact raytracer revival (Windows-only contrib) — not load-bearing

## Context

- **Origin:** Full code review performed 2026-05-14. Findings + roadmap at `/Users/kohlbach/Claude/BALL/ROADMAP-1.6.md`.
- **Repos:** `BALL-Project/ball` (last commit 2022-05-24) and `BALL-Project/ball_contrib` (dep snapshot ~2016-2018), cloned under `/Users/kohlbach/Claude/BALL/`.
- **Build approach adopted:** `ball_contrib` abandoned; build against Homebrew packages (`qt@5 boost eigen fftw tbb glew open-babel lp_solve libsvm flex bison`).
- **8 patches already applied locally** (uncommitted, shallow clone): CMake policy/version modernization, Boost config-mode, Eigen 5 version-header fix, C++11→C++14, Boost.Asio API breakage in `networking.{h,C}`, a latent `contourSurface.h` bug.
- **The blank-scene blocker (Phase 4a target):** `Scene` embeds `GLRenderWindow : QGLWidget`. macOS Tahoe force-enables layer-backing on every NSView; `QGLWidget` (NSOpenGLContext-based, not layer-backed) gets promoted to a detached 100×30 native window. Renderer draws into an invisible context. Confirmed via Qt `-l` platform logging.
- **Rendering architecture:** `RenderSetup` is a `QThread` (one per renderer). Worker threads `makeCurrent()` + render; GUI thread swaps buffers. `GLRenderer` uses ~100 fixed-function GL calls. The raytracer path already renders to a CPU pixel buffer that gets blitted as a texture (`glRenderWindow.C::refresh()`).
- **Build/run commands** that work today are documented in `ROADMAP-1.6.md`.

## Constraints

- **Compatibility**: Must build and render on macOS (Apple Silicon), Linux, and Windows — platform independence is a hard requirement, no per-OS graphics code if avoidable.
- **Tech stack**: C++ / CMake / Qt. Phase 4a stays on Qt 5.15 + fixed-function GL (compat profile); Qt 6 is Phase 4b.
- **Dependencies**: System/Homebrew packages on macOS/Linux, vcpkg on Windows. No reliance on the dead `ball_contrib`.
- **Risk**: Phase 4a touches the threaded renderer — the highest-risk area. Threading rework must be incremental and verifiable against a running GUI.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Abandon `ball_contrib`, use Homebrew/system deps | 2016-era source tarballs (Qt 5.10, Boost 1.60…) don't build on macOS Tahoe / Apple Silicon | ✓ Good — full stack built |
| Bridge to C++14 (not C++17) for the build baseline | BALL's legacy code uses `std::unary_function`/`bind2nd` removed in C++17; C++14 still has them | — Pending — Phase 2 removes the bridge |
| Phase 4a: `QOpenGLWidget` + compat profile, defer Qt6/pipeline to 4b | Smallest change that restores rendering on all 3 OSes; decouples the macOS blocker from the large pipeline rewrite | — Pending |
| Phase 4a threading: hybrid (interactive GL on GUI thread, raytracer on CPU-buffer worker thread) | `QOpenGLWidget` requires default-FBO rendering on the GUI thread; the raytracer already works via a CPU buffer, so only the GL renderer needs to move | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-14 after initialization*
