# Roadmap: BALLView 1.6 Modernization

**Created:** 2026-05-14
**Granularity:** standard
**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies.

This roadmap mirrors the human-authored `/Users/kohlbach/Claude/BALL/ROADMAP-1.6.md`. Phase 4a (rendering port) is broken out as its own dedicated GSD phase because it is the immediate execution priority â restoring the blank 3D scene. It is sequenced right after the build baseline because it only needs the working build that already exists, not the C++17 or dependency-overhaul work.

## Phases

- [x] **Phase 1: Build Baseline** - Commit the 8 modern-toolchain patches, bump version, document the macOS build flow
- [x] **Phase 2: Rendering Port (4a)** - Port `GLRenderWindow` from `QGLWidget` to `QOpenGLWidget` so the embedded 3D scene renders on all 3 OSes *(complete — human-verified on macOS; RENDER-08 Linux/Windows render-check stays open pending Phase 4/9)*
- [ ] **Phase 02.1: Renderer boundary extraction** - Extract a Qt-GL-free `RenderSurface`/`RendererFactory` boundary so Phase 5 is a contained backend swap *(inserted; depends on Phase 2, blocks Phase 5)*
- [ ] **Phase 3: Language Modernization** - Move the codebase to C++17, remove C++17-removed constructs, set the standard via CMake
- [ ] **Phase 4: Dependency System Overhaul** - Delete `ball_contrib`, adopt Homebrew/system + vcpkg, rewrite stale `Find*.cmake` as config-mode
- [ ] **Phase 5: Qt 6 + Pipeline (4b)** - Build against Qt 6 and replace deprecated VIEW APIs *(NOTE: oversized — flagged for a split into Qt6-migration + renderer-backend in the next roadmap revision)*
- [ ] **Phase 6: Python Bindings** - Replace the SIP 4 binding generator with SIP 6 or pybind11/nanobind against Python 3.12+
- [ ] ~~**Phase 7: Networking Rework**~~ - **Deferred to backlog 999.3** — not core value, the Asio code already compiles (Phase 1); the proper rework + test is 1.6.x polish
- [ ] **Phase 8: macOS Packaging** - Embed `data/` into the `.app` bundle, wire `macdeployqt`, produce a notarizable universal binary
- [ ] **Phase 9: CI & Tests** - GitHub Actions matrix (macOS-arm64/Linux/Windows) and wire the `test/` tree into `ctest`

## Phase Details

### Phase 1: Build Baseline
**Goal**: The modern-toolchain build is reproducible, committed, versioned, and documented â a clean starting point for all later work.
**Depends on**: Nothing (first phase)
**Requirements**: BUILD-01, BUILD-02, BUILD-03, BUILD-04
**Success Criteria** (what must be TRUE):
  1. A fresh checkout configures and builds BALL/VIEW/BALLView on macOS (Apple Silicon) from Homebrew dependencies using a single documented command
  2. The 8 modern-toolchain patches are committed to the repo with clear, descriptive messages
  3. The CMake `PROJECT` version declaration reads `1.6.0-dev`
  4. `BUILD-macos.md` exists and walks a new contributor through dependency install and the build/run flow
**Plans**: 1 plan
- [x] 01-01-PLAN.md — Commit the 8 toolchain patches, bump version to 1.6.0-dev, write BUILD-macos.md, verify rebuild

### Phase 2: Rendering Port (4a)
**Goal**: BALLView's embedded 3D scene renders molecules again on all three platforms by moving off the removed-in-Qt6 `QGLWidget`. This is the immediate execution priority and the highest-risk area.
**Depends on**: Phase 1 (needs the working build baseline; does NOT depend on C++17 or the dependency overhaul)
**Requirements**: RENDER-01, RENDER-02, RENDER-03, RENDER-04, RENDER-05, RENDER-06, RENDER-07, RENDER-08
**Success Criteria** (what must be TRUE):
  1. BALLView displays a molecule in the embedded 3D scene on macOS, Linux, and Windows
  2. The scene widget renders inside the main window â no detached/mis-sized native window
  3. The user can rotate, zoom, pick, and select in the scene with mouse and keyboard
  4. The raytracer renderer still produces output via its CPU pixel-buffer path, and on-screen text appears via a `QPainter` overlay
  5. `RenderSetup`, `scene.C`, `glOffscreenTarget.C`, `glRenderer.C`, and `glRenderWindow.{h,C}` compile with zero `QGLWidget`/`QGLContext`/`QGLFormat` references
**Plans**: 4 plans
- [x] 02-01-PLAN.md — Wave 0: grep symbol gate + resolve raytracer-GL open question A1
- [x] 02-02-PLAN.md — Core port: GLRenderWindow base-class swap, QSurfaceFormat compat profile, paintGL, QPainter text overlay, main.C context sharing
- [x] 02-03-PLAN.md — Renderer subsystem cleanup: renderSetup.C, glRenderer.C, glOffscreenTarget.C
- [x] 02-04-PLAN.md — scene.C port + stereo/multi-display guard-defer + human smoke check
**UI hint**: yes

### Phase 02.1: Renderer boundary extraction (INSERTED)

**Goal**: Extract a clean, Qt-GL-free renderer/surface boundary so the Phase 5 pipeline modernization is a contained ~2-file backend swap behind a flag, not a re-touch of scene.C. Pure refactor — no behaviour change.
**Depends on**: Phase 2 (the QOpenGLWidget port must land first so the extracted `RenderSurface` wraps the real post-port widget)
**Blocks**: Phase 5 (Qt 6 + Pipeline) — the contained-swap boundary is its prerequisite
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):
  1. A `RenderSurface` interface owns the context-lifecycle verbs (`beginFrame`/`endFrame`); `RenderSetup::makeCurrent()`'s GL-specific body moves behind it
  2. A `RendererFactory` constructs renderers and surfaces by enum; `scene.C` contains zero `new GLRenderWindow` and zero `dynamic_cast<GLRenderWindow>`/`dynamic_cast<GLRenderer>` sites
  3. The `Renderer` interface gains a batched `renderRepresentations_()` + `capabilities()` entry point; existing immediate-mode renderers are untouched (default fan-out preserves behaviour)
  4. BALLView builds and renders identically to post-Phase-2 (pure refactor — same pixels, verified by the Phase 2 smoke checklist)
**UI hint**: no
**Reference**: `.planning/RENDERER-INTERFACE-BOUNDARY.md` (full design)
**Plans:** 4/4 plans complete

Plans:
- [ ] TBD (run /gsd-plan-phase 02.1 to break down)

### Phase 3: Language Modernization
**Goal**: The whole project compiles under C++17 with the standard set the modern CMake way, removing the load-bearing C++14 bridge.
**Depends on**: Phase 1 (build baseline). Independent of Phase 2.
**Requirements**: LANG-01, LANG-02, LANG-03
**Success Criteria** (what must be TRUE):
  1. The codebase compiles cleanly with C++17 and the C++14 bridge flag is gone
  2. No occurrences of `std::unary_function`, `binary_function`, `bind2nd`, `bind1st`, `ptr_fun`, `auto_ptr`, or `mem_fun` remain across the 7 known files
  3. The C++ standard is configured via `CMAKE_CXX_STANDARD`/`target_compile_features`, with no raw `-std=` flags
**Plans**: TBD

### Phase 4: Dependency System Overhaul
**Goal**: All dependencies come from current, supported sources (Homebrew/system on macOS/Linux, vcpkg on Windows) with `ball_contrib` fully removed from the build path.
**Depends on**: Phase 1 (build baseline). Benefits from Phase 3 (C++17) since modern dependency headers may require it.
**Requirements**: DEPS-01, DEPS-02, DEPS-03, DEPS-04
**Success Criteria** (what must be TRUE):
  1. CMake resolves every dependency via system/Homebrew packages with `ball_contrib` no longer on the build path
  2. A `vcpkg.json` manifest provides the complete Windows dependency set
  3. Stale bundled `Find*.cmake` modules (Boost, TBB, Eigen3, OpenBabel) are replaced with config-mode `find_package` wherever upstream ships a config
  4. Minimum dependency versions are pinned in CMake and documented
**Plans**: TBD

### Phase 5: Qt 6 + Pipeline (4b)
**Goal**: BALLView builds and runs on Qt 6 with deprecated VIEW APIs removed. The compatibility-profile fixed-function GL path is *kept working under Qt 6* as the known-good backend; the full pipeline rewrite is explicitly NOT in this phase (see PIPE-01, v2).
**Depends on**: Phase 02.1 (the renderer boundary), Phase 2 (the `QOpenGLWidget` port), Phase 4 (Qt 6 from the modern dependency system)
**Requirements**: QT6-01, QT6-02
**Success Criteria** (what must be TRUE):
  1. BALLView builds and launches against Qt 6 with all `QGLWidget`-era APIs removed
  2. The user-facing GUI behaves correctly with `QRegExp` and `QDesktopWidget` replaced by Qt 6 equivalents
  3. The 3D scene still renders correctly under Qt 6 via the compatibility-profile fixed-function path (no pixel regression vs. Phase 2)
**Plans**: TBD
**UI hint**: yes

> **NOTE (Codex review, to apply in the next roadmap revision):** This phase is
> oversized and should be split — Qt 6 migration / renderer-backend decision spike /
> new GL-core-or-QRhi backend. `QT6-03` ("pipeline modernized off fixed-function
> GL") was **removed** because it contradicted `PIPE-01` (full pipeline rewrite =
> v2/out-of-scope). The actual pipeline rewrite is `PIPE-01`, a separate future phase.

### Phase 6: Python Bindings
**Goal**: BALL's Python bindings build and import against a supported Python using a modern binding generator.
**Depends on**: Phase 3 (C++17 codebase), Phase 4 (modern dependency system)
**Requirements**: PY-01
**Success Criteria** (what must be TRUE):
  1. BALL's Python bindings build against Python 3.12+ using SIP 6 or pybind11/nanobind
  2. The generated module imports and exercises core BALL classes from a Python interpreter
**Plans**: TBD

### Phase 7: Networking Rework — DEFERRED TO BACKLOG 999.3
**Status**: Removed from the v1.6 active roadmap per the Codex review. Networking is not core value, and the Boost.Asio code already *compiles* (the API breakage was fixed in Phase 1). The proper `TCPServer` rework + unit test is 1.6.x polish, tracked as backlog **999.3**. `NET-01` moved to REQUIREMENTS.md "Deferred (1.6.x)".
**Plans**: TBD

### Phase 8: macOS Packaging
**Goal**: `BALLView.app` is a self-contained, double-clickable, notarizable universal bundle.
**Depends on**: Phase 5 (Qt 6 build is what gets packaged)
**Requirements**: PKG-01, PKG-02
**Success Criteria** (what must be TRUE):
  1. `BALLView.app` launches by double-click with no environment variables set, finding its `data/` in `Contents/Resources`
  2. The macOS build produces a `macdeployqt`-processed, notarizable universal (arm64 + x86_64) bundle
**Plans**: TBD

### Phase 9: CI & Tests
**Goal**: Every push is proven to build on all three platforms and the test suite runs automatically.
**Depends on**: Phase 1 (build baseline). Most valuable once Phases 2-5 land, but can be stood up incrementally.
**Requirements**: CI-01, CI-02
**Success Criteria** (what must be TRUE):
  1. A GitHub Actions matrix builds BALL/VIEW/BALLView on macOS-arm64, Linux, and Windows
  2. The `test/` tree is wired into the build and `ctest` runs in CI
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Build Baseline | 1/1 | Complete | 2026-05-14 |
| 2. Rendering Port (4a) | 4/4 | In Progress — awaiting human visual recheck (HiDPI fix landed) | - |
| 02.1 Renderer boundary extraction | 0/0 | Not planned | - |
| 3. Language Modernization | 0/0 | Not started | - |
| 4. Dependency System Overhaul | 0/0 | Not started | - |
| 5. Qt 6 + Pipeline (4b) | 0/0 | Not started | - |
| 6. Python Bindings | 0/0 | Not started | - |
| 7. Networking Rework | — | Deferred to backlog 999.3 | - |
| 8. macOS Packaging | 0/0 | Not started | - |
| 9. CI & Tests | 0/0 | Not started | - |

---

## Backlog

### Phase 999.1: BALLView UI maintainer open-questions (BACKLOG)

**Goal:** Get BALL maintainer decisions on 4 UI questions from the Claude Design Handover package — long lead time, raise before Milestone 2 ("BALLView Refresh", SEED-001) reaches its Inspector phase.
**Questions:**
  1. macOS menu bar — keep inline, or use the native global menubar via `QAction::setMenuRole`?
  2. Legacy 5-dock "Classic" workspace — keep as a long-term opt-in preset, or retire after one release?
  3. Theme picker — ship one neutral theme, or Light/Dark/Follow-System? (handover recommends Follow-System)
  4. Translation churn — the menu re-org invalidates ~40% of `BALLView-de_DE.ts`; plan a community translation round.
**Requirements:** TBD
**Plans:** 0 plans
**Reference:** `.planning/DESIGN-HANDOVER-INTEGRATION.md`, `.planning/seeds/SEED-001-ballview-refresh-ui-milestone.md`

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.2: Ninja build generator (BACKLOG)

**Goal:** Switch the default CMake generator from Make/MSBuild to **Ninja** across all three platforms.
**Why:** Faster incremental builds and consistent parallelism everywhere; on Windows it replaces slow MSBuild and makes the legacy `ball_contrib` "do not use -j" hazard moot (contrib is already obsolete on macOS/Linux). Pure build-tooling change — **zero source changes**, just `cmake -G Ninja`.
**Cross-platform impact sketch:**
  - **macOS:** `brew install ninja`; `cmake -G Ninja`. Faster rebuilds, better core utilization than Make. Update `BUILD-macos.md`. Low risk.
  - **Linux:** `apt/dnf install ninja-build`; `cmake -G Ninja`. Same benefits. Low risk.
  - **Windows:** Biggest win — replaces MSBuild (`cmake -G Ninja` in a VS dev shell, or `Ninja Multi-Config`). Eliminates the `ball_contrib` `/maxcpucount`/`-j` undefined-behaviour warning entirely. Slightly more setup (Ninja must be on PATH).
  - **CI (Phase 9):** the GH Actions matrix should standardize on `-G Ninja` on all three runners — simpler, faster, uniform.
  - **Risk:** Low. CMake fully supports Ninja; the only watch-items are non-standard custom commands / `add_custom_command` ordering and any code that shells out assuming Makefile targets. BALL's CMake is fairly standard. Best sequenced *after* Phase 4 (dependency overhaul) and *with* Phase 9 (CI) so it lands once, matrix-wide.
**Requirements:** TBD
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

### Phase 999.3: Networking rework (BACKLOG)

**Goal:** Rework `TCPServer`/`TCPServerThread` onto the modern Boost.Asio acceptor/socket model and cover it with a unit test.
**Why backlog, not v1.6:** Networking is not the milestone's core value, and the Boost.Asio API breakage was already fixed in Phase 1 — the code compiles and links. A proper rework + test is genuine polish but does not gate "build and render on 3 OSes". Deferred to a 1.6.x release. (Moved out of the active roadmap per the Codex adversarial review.)
**Requirements:** NET-01
**Plans:** 0 plans

Plans:
- [ ] TBD (promote with /gsd-review-backlog when ready)

---
*Roadmap created: 2026-05-14*
*Mirrors `/Users/kohlbach/Claude/BALL/ROADMAP-1.6.md` (phases 1, 2, 3, 4a, 4b, 5, 6, 7, 8). Revised 2026-05-14 after Codex adversarial review — cheap fixes applied; structural changes (early CI phase, Phase 5 split, diagnostics requirement, feature matrix) pending a deliberate roadmap revision.*
