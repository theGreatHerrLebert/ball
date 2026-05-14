---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: milestone
status: executing
stopped_at: Completed 02.1-02-PLAN.md
last_updated: "2026-05-14T10:11:35.249Z"
progress:
  total_phases: 15
  completed_phases: 2
  total_plans: 8
  completed_plans: 7
  percent: 88
---

# STATE: BALLView 1.6 Modernization

## Project Reference

**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies — the 3D scene working cross-platform is the non-negotiable outcome.

**Current Focus:** Phase 02.1 — Renderer boundary extraction

## Current Position

Phase: 02.1 (Renderer boundary extraction) — EXECUTING
Plan: 3 of 3
**Phase:** 02.1 — Renderer boundary extraction
**Plan:** Plans 01+02 complete (Wave 1: RenderSurface interface + Renderer batched boundary); Plan 03 next (Wave 2)
**Status:** Executing Phase 02.1
**Progress:** [█████████░] 88%

```
Phase 1     [x]  Build Baseline
Phase 2     [x]  Rendering Port (4a)  — human-verified on macOS
Phase 02.1  [ ]  Renderer boundary extraction  <- NEXT
Phase 3     [ ]  Language Modernization
Phase 4     [ ]  Dependency System Overhaul
Phase 5     [ ]  Qt 6 (4b)  (flagged oversized — split pending)
Phase 6     [ ]  Python Bindings
Phase 8     [ ]  macOS Packaging
Phase 9     [ ]  CI & Tests
(Phase 7 Networking → backlog 999.3)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 2/11 (incl. 02.1; excl. backlog) |
| Plans complete | 6 |
| Requirements | 30 active + NET-01 deferred |
| Milestone | v1.6 |
| Phase 01-build-baseline P01 | 3min | 3 tasks | 9 files |
| Phase 02-rendering-port-4a P01 | 3min | 2 tasks | 2 files |
| Phase 02-rendering-port-4a P02 | 12min | 2 tasks | 3 files |
| Phase 02-rendering-port-4a P03 | 9min | 2 tasks | 4 files |
| Phase 02-rendering-port-4a P04 | ~4h (incl. 3 debug rounds + human verify) | 3 tasks | 7 files |
| Phase 02.1-renderer-boundary-extraction P01 | 6min | 2 tasks | 7 files |
| Phase 02.1 P02 | 8min | 2 tasks | 2 files |

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

**Last action:** Phase 02.1 (Renderer boundary extraction) planned — 3 plans verified by the plan-checker (passed after one revision). Codex structural roadmap changes also applied (Phase 02.2 + 05.1 inserted, Phase 5 split, feature matrix, etc.).

**Stopped at:** Completed 02.1-02-PLAN.md

**Next action:** `/gsd-execute-phase 02.1` — pure refactor extracting `RenderSurface` (Plan 01) + additive `Renderer::renderRepresentations_()`/`capabilities()` (Plan 02) in parallel Wave 1, then `RendererFactory` + `scene.C` concrete-type removal + human-verify identical render (Plan 03) in Wave 2. Alternatively Phase 02.2 (CI matrix) is independent and could go first.

**Notes:**

- Phase 02.1 plans: 01 → `RenderSurface` interface + `makeCurrent` body behind `beginFrame/endFrame` (ARCH-01); 02 → additive batched `renderRepresentations_()`/`capabilities()` non-pure default fan-out (ARCH-03); 03 → `RendererFactory` + kill `new GLRenderWindow`/`dynamic_cast<GLRenderWindow|GLRenderer>` in scene.C + human-verify (ARCH-02, ARCH-04). The ~10 deferred-stereo `new GLRenderWindow` 3-arg sites are EXPECTED residuals (guard-deferred to Phase 5).
- Build files: BALL uses per-directory `sources.cmake` (e.g. `source/VIEW/RENDERING/sources.cmake`), NOT `source/VIEW/CMakeLists.txt`. Headers are implicit, not listed.
- A1 CONFIRMED (Phase 2): raytracer worker issues no GL — threading scope stays minimal. `TilingRenderer`'s GL path is GUI-thread-only.

---
*State initialized: 2026-05-14*
