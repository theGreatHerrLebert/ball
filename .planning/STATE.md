---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: milestone
status: executing
stopped_at: Completed 03-02-PLAN.md (LANG-03 CMake C++17 configuration)
last_updated: "2026-05-14T13:18:12.773Z"
progress:
  total_phases: 17
  completed_phases: 4
  total_plans: 13
  completed_plans: 12
  percent: 92
---

# STATE: BALLView 1.6 Modernization

## Project Reference

**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies — the 3D scene working cross-platform is the non-negotiable outcome.

**Current Focus:** Phase 03 — language-modernization

## Current Position

Phase: 03 (language-modernization) — EXECUTING
Plan: 1 of 3
**Phase:** 3 — Language Modernization
**Plan:** Not started
**Status:** Executing Phase 03
**Progress:** [█████████░] 92%

```
Phase 1     [x]  Build Baseline
Phase 2     [x]  Rendering Port (4a)        — human-verified on macOS
Phase 02.1  [x]  Renderer boundary extraction
Phase 02.2  [x]  CI and build-smoke matrix  — CI green on all 4 jobs
Phase 3     [ ]  Language Modernization     <- NEXT
Phase 4     [ ]  Dependency System Overhaul
Phase 5     [ ]  Qt 6 Migration (4b)
Phase 05.1  [ ]  Renderer backend decision spike  (depends on Phase 5)
Phase 6     [ ]  Python Bindings
Phase 8     [ ]  Packaging & Distribution
Phase 9     [ ]  Test Suite Triage
(Phase 7 Networking → backlog 999.3)
```

NOTE: gsd-tools `phase complete 02.2` reported `next_phase: 05.1` — that is the
tool's decimal-phase bug; 05.1 depends on Phase 5. The real next phase is **Phase 3**.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 4 (1, 2, 02.1, 02.2) of 11 active (excl. backlog) |
| Plans complete | 10 |
| Requirements | 37 active + NET-01 deferred |
| Milestone | v1.6 |
| Phase 01-build-baseline P01 | 3min | 3 tasks | 9 files |
| Phase 02-rendering-port-4a P01-04 | ~4.5h (incl. 3 debug rounds + human verify) | 9 tasks | 16 files |
| Phase 02.1-renderer-boundary-extraction P01-03 | ~20min + human verify | 7 tasks | ~12 files |
| Phase 02.2-ci-and-build-smoke-matrix P01 | 25min | 2 tasks | 3 files |
| Phase 02.2-ci-and-build-smoke-matrix P02 | ~3h (incl. 2 CI bring-up iterations) | 3 tasks | 4 files |
| Phase 03-language-modernization P01 | 83 | 3 tasks | 7 files |
| Phase 03 P02 | 5min | 3 tasks | 2 files |

## Accumulated Context

### Decisions

- Abandon `ball_contrib`; build against Homebrew/system deps (macOS/Linux) and vcpkg (Windows). Already proven on macOS Tahoe.
- Phase 4a (GSD Phase 2): `QOpenGLWidget` + compatibility profile, deferring Qt 6 and the pipeline rewrite to Phase 4b (GSD Phase 5). Smallest change that restores rendering on all 3 OSes.
- Phase 4a threading: hybrid model — interactive GL on the GUI thread via `paintGL()`, raytracer stays a CPU-buffer worker thread.
- C++14 is a load-bearing bridge until GSD Phase 3 (Language Modernization) removes it.
- GSD Phase 2 (Rendering) is sequenced right after the build baseline because it only needs the working build, not the C++17 or dependency-overhaul phases.
- [Phase 01-build-baseline]: CMake VERSION kept numeric (1.6.0); -dev pre-release marker carried as an inline comment
- [Phase 02-rendering-port-4a]: A1 CONFIRMED: raytracer/BufferedRenderer worker path issues no GL — Plans 03/04 threading scope stays minimal (no QOffscreenSurface/shared context)
- [Phase 02-rendering-port-4a]: GLRenderWindow rebased on QOpenGLWidget: compat-profile QSurfaceFormat, GL work in initializeGL/resizeGL/paintGL, manual swap deleted, QPainter text overlay, global context sharing in main()
- [Phase 02-rendering-port-4a]: Plan 02-03: renderer-side files (renderSetup.C, glRenderer.C, glOffscreenTarget.{h,C}) ported to QOpenGLWidget API — QOpenGLContext, grabFramebuffer, Format_RGBA8888+mirrored label upload, QOpenGLFramebufferObject offscreen target; scene.C is the sole remaining VIEW build failure (Plan 04)
- [Phase 02-rendering-port-4a]: COMPLETE — human-verified on macOS. scene.C ported; the mechanical port then needed 3 debugger rounds for structural QGLWidget→QOpenGLWidget mismatches: lazy context creation (`29aa3d2` defer GL init to initializeGL), HiDPI device-pixel viewport (`81d1145`), and — the big one — `GLRenderWindow::paintGL()` was never being called (Scene::eventFilter swallowed Paint events + ignoreEvents forced); fixed by rendering the GL scene inside paintGL (`5ca7a47`). Two benign startup warnings silenced (`207b1b9`). RENDER-08 (Linux/Windows render) is a documented carry-forward — unverifiable before Phase 4/9.
- [Phase 02.1-renderer-boundary-extraction]: Plan 02.1-01: RenderSurface interface owns beginFrame/endFrame/nativeHandle; RenderSetup::makeCurrent() delegates via dynamic_cast<RenderSurface*>. endFrame() is a deliberate GL-backend no-op (gains meaning for QRhi in Phase 5). GLOffscreenTarget also adopted the interface (it is a RenderWindow subclass too).
- [Phase 02.1]: Plan 02.1-02: Renderer base gains batched renderRepresentations_(const RepresentationList&) + capabilities() Caps query (ARCH-03). capabilities() made NON-pure (deviation from boundary doc's = 0 sketch) to keep the change additive — pure would force-break all ~7 existing subclasses. Default renderRepresentations_() fans out to renderOneRepresentation(); not yet wired into RenderSetup/Scene (Phase 5 scope).
- [Phase 02.1]: Plan 02.1-03 Tasks 1-2 (ARCH-02): RendererFactory namespace (enum Kind + makeRenderer/makeSurface) centralises construction; scene.C's non-deferred paths routed through it. gl_renderer_/main_display_ kept concrete-typed with one static_cast at the construction site only (fewest ripples). setDownsamplingFactor added to RenderSurface, setFogIntensity to Renderer base (no-op defaults) so the casts drop out. RTTI::isKindOf<GLRenderer> guards routed through RenderSetup::getRendererType(). dynamic_cast<GLRenderWindow> replaced with dynamic_cast<RenderSurface>+nativeHandle(). ~9 deferred-stereo 3-arg-ctor new GLRenderWindow + 8 new GLRenderer sites remain as expected guard-deferred residuals (all in stereo/multi-display methods, deferred to Phase 5). Build clean. PAUSED at Task 3 human-verify checkpoint (ARCH-04 identical render).
- [Phase 02.2]: DIAG-01: BALLVIEW_GL_DIAG single-line stdout diagnostic emitted from GLRenderWindow::initializeGL(); fbo_size is pre-layout so the smoke check uses the line's presence + live gl_version as the GL-context oracle, not exact dimensions
- [Phase 02.2]: Render smoke check uses the D-09 fallback (fixed PDB data/structures/bpti.pdb) + a minimal -export-png main.C flag; the auto-demo peptide is restored ~/.BALLView session state, not a deterministic CI input
- [Phase 02.2]: Plan 02.2-02: `.github/workflows/ci.yml` — single `build` job, strategy.matrix.include per OS (extensible without rewrite, D-11); macOS mirrors BUILD-macos.md verbatim, Linux apt+xvfb/software-Mesa, Windows non-blocking via matrix-driven `continue-on-error: ${{ !matrix.blocking }}` (D-03). Standalone blocking `lint` job runs the legacy-GL grep gate.
- [Phase 02.2]: The `lint`-will-be-red finding was RESOLVED during CI bring-up: `glDisplayList.h` ported off `QtOpenGL/qgl.h` → `QtGui/qopengl.h` (`da043c1`), and the grep gate now skips comment-only lines (`4c91600`). The legacy-GL lint job is genuinely green.
- [Phase 02.2]: CI bring-up took 2 iterations after the workflow landed. (1) macOS build red — C++ standard was a late raw `-std=` flag, so AppleClang 15 ran feature-detection sub-C++14 and Eigen rejected the build → fixed with `CMAKE_CXX_STANDARD 14` set early in CMakeLists.txt (`3ac3f24`, a LANG-03 down payment). (2) Linux link red — Ubuntu's `liblpsolve55.a` is non-PIC, can't link into shared libBALL → dropped lp_solve on Linux + `-DUSE_LPSOLVE=OFF` (`1959d9b`); lp_solve is optional, macOS keeps it.
- [Phase 02.2]: COMPLETE — CI run 25859952862 (`1959d9b`) all 4 jobs green: build macos/linux/windows + lint. Render-smoke ran & passed on macOS AND Linux (BALLView headless-rendered a non-blank PNG on each) — real cross-platform render validation, substantially de-risks RENDER-08. Even Windows built clean (non-blocking; closer to "required" than expected).
- [Phase 03-language-modernization]: D-01: Dropped unary_function/binary_function base inheritance entirely — no typedef hand-rolling per LANG-02 D-01
- [Phase 03-language-modernization]: D-02: Rewrote all adapter call sites (bind2nd/mem_fun/mem_fun_ref/not1) as lambdas in 3 files; LANG-02 grep gate passes
- [Phase 03]: D-03/D-04/D-05/D-06: CMAKE_CXX_STANDARD 17 global; both raw -std= lines and stale C++14-bridge comment deleted; CMAKE_CXX_EXTENSIONS OFF retained; blanket -Wno-deprecated-declarations removed

### Roadmap Evolution

- Phase 02.1 inserted after Phase 2: Renderer boundary extraction — pure refactor that makes Phase 5 a contained backend swap. Depends on Phase 2, blocks Phase 5. Design: `.planning/RENDERER-INTERFACE-BOUNDARY.md`.
- Design Handover package analyzed (`.planning/DESIGN-HANDOVER-INTEGRATION.md`): a separate UI/UX modernization milestone (~8 phases) that depends on this milestone's Phase 5 (Qt 6). Planted as SEED-001 (Milestone 2 "BALLView Refresh", target 1.7); not folded into the current roadmap.
- Codex adversarial review run on the roadmap. Cheap fixes AND structural changes applied: Phase 02.2 (early CI matrix) + Phase 05.1 (renderer backend spike) inserted; Phase 5 split to Qt6-only; Phase 6 restructured (decision+slice); Phase 8 scope clarified; DEPS-05/FEAT-01/DIAG-01/SPIKE/PY-02/PKG-03 added; feature matrix added.
- Backlog: 999.1 (UI maintainer open-questions), 999.2 (Ninja generator), 999.3 (networking rework).
- Phase 02.1 planned: 3 plans, 2 waves (01+02 parallel Wave 1, 03 Wave 2). Plan-checker passed after one revision (build-file path fix: BALL uses per-directory `sources.cmake`, not `source/VIEW/CMakeLists.txt`).

### Todos

- (none open)

### Blockers

- None.

## Session Continuity

**Last action:** Phase 02.2 (CI and build-smoke matrix) marked COMPLETE — gsd-verifier passed 10/10, `phase complete 02.2` run. CI run 25859952862 (`1959d9b`) is fully green (build macos/linux/windows + lint); render-smoke ran & passed on macOS and Linux. Pushed branch `v1.6-modernization` is at `1959d9b`.

**Stopped at:** Completed 03-02-PLAN.md (LANG-03 CMake C++17 configuration)

**Next action:** Phase 3 — Language Modernization (`/gsd-discuss-phase 3` or `/gsd-plan-phase 3`). Move the codebase to C++17, remove C++17-removed constructs (`std::unary_function`/`bind2nd`/etc. across the 7 known files), and bump `CMAKE_CXX_STANDARD` 14→17 (the mechanism is already in CMakeLists.txt from the Phase 02.2 CI fix — Phase 3 just bumps the value and removes the legacy raw `-std=` lines from `BALLCompilerSpecific.cmake`). CI (Phase 02.2) is now the regression net for this work.

**Notes:**

- CI ownership: I trigger + `gh run watch` CI myself after build-relevant pushes, diagnose failures from `gh`-fetched logs, fix, re-push, iterate to green. (See memory: feedback_ci_supervision.md.)
- gsd-tools has a recurring decimal-phase bug — `phase complete` mis-marks/mis-reports decimal phases (it spuriously flipped 05.1 once; reported `next_phase: 05.1` wrongly). Hand-verify roadmap/STATE after gsd-tools phase ops.
- Build files: BALL uses per-directory `sources.cmake`, NOT `source/VIEW/CMakeLists.txt`. Headers are implicit, not listed.
- Renderer boundary (Phase 02.1) is in place: `RenderSurface`/`RendererFactory` + additive `Renderer::renderRepresentations_()`/`capabilities()`. Phase 5/05.1 backend swap is now contained — no scene.C edits.
- A1 CONFIRMED (Phase 2): raytracer worker issues no GL. `TilingRenderer`'s GL path is GUI-thread-only.
- Build/run: `BUILD-macos.md`. CI mirrors it. `~/.BALLView` was deleted (stale element-color cache shadowed compiled defaults — backlog 999.4).

---
*State initialized: 2026-05-14*
