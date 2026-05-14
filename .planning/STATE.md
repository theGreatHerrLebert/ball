---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: 02-04-PLAN.md Tasks 1-2 complete; paused at Task 3 human-verify checkpoint
last_updated: "2026-05-14T08:00:00.000Z"
progress:
  total_phases: 9
  completed_phases: 1
  total_plans: 5
  completed_plans: 4
  percent: 80
---

# STATE: BALLView 1.6 Modernization

## Project Reference

**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies — the 3D scene working cross-platform is the non-negotiable outcome.

**Current Focus:** Phase 2 — Rendering Port (4a)

## Current Position

Phase: 2 (Rendering Port (4a)) — EXECUTING
Plan: 4 of 4
**Phase:** 2
**Plan:** 04 in progress — Tasks 1-2 complete and committed; paused at Task 3 (human-verify checkpoint)
**Status:** Executing Phase 2 — awaiting human smoke check
**Progress:** [████████░░] 80%

```
Phase 1  [x]  Build Baseline
Phase 2  [ ]  Rendering Port (4a)  <- immediate execution priority
Phase 3  [ ]  Language Modernization
Phase 4  [ ]  Dependency System Overhaul
Phase 5  [ ]  Qt 6 + Pipeline (4b)
Phase 6  [ ]  Python Bindings
Phase 7  [ ]  Networking Rework
Phase 8  [ ]  macOS Packaging
Phase 9  [ ]  CI & Tests
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases complete | 1/9 |
| Plans complete | 2 |
| Requirements mapped | 27/27 |
| Milestone | v1.6 |
| Phase 01-build-baseline P01 | 3min | 3 tasks | 9 files |
| Phase 02-rendering-port-4a P01 | 3min | 2 tasks | 2 files |
| Phase 02-rendering-port-4a P02 | 12min | 2 tasks | 3 files |
| Phase 02-rendering-port-4a P03 | 9min | 2 tasks | 4 files |

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
- [Phase 02-rendering-port-4a]: Plan 02-04 Tasks 1-2: scene.C ported (QSurfaceFormat+QOpenGLContext stereo probe, swap-sync block -> update(), grabFramebuffer) and the 4 top-level stereo/multi-display paths (addGlWindow, enterStereo, enterDualStereo, enterDualStereoDifferentDisplays) guard-deferred per Pitfall 6. Full clean BALL+VIEW+BALLView build links bin/BALLView.app; grep gate clean across source/VIEW + include/BALL/VIEW except the known-deferred glDisplayList.h. Awaiting human smoke-check (Task 3 checkpoint)

### Roadmap Evolution

- Phase 02.1 inserted after Phase 2: Renderer boundary extraction — pure refactor that makes Phase 5 a contained backend swap. Depends on Phase 2, blocks Phase 5. Design: `.planning/RENDERER-INTERFACE-BOUNDARY.md`.
- Design Handover package analyzed (`.planning/DESIGN-HANDOVER-INTEGRATION.md`): a separate UI/UX modernization milestone (~8 phases) that depends on this milestone's Phase 5 (Qt 6). To be planted as Milestone 2; not folded into the current roadmap.

### Todos

- Phase 2 rendering defect: geometry renders embedded (macOS blocker fixed) but mis-projected — bonds draw as giant cylinders. `Cannot resize window. Size 0 x 0` in the log is the likely cause. Gap to close before Phase 2 verification.

### Blockers

- Phase 2 human smoke-check FAILED visually: scene renders embedded (core macOS blocker resolved) but geometry is mis-scaled/mis-projected and unrecognizable. Phase 2 cannot be marked complete until this rendering gap is fixed.

## Session Continuity

**Last action:** Executing 02-04-PLAN.md — Tasks 1-2 complete and committed (`b4965b1` scene.C port, `f75dbc5` top-level stereo/multi-display guard-defer). Full clean `make BALL VIEW BALLView -j8` succeeds; `bin/BALLView.app` linked. Grep gate clean across `source/VIEW` + `include/BALL/VIEW` except the known-deferred `glDisplayList.h` (`QtOpenGL/qgl.h`, logged in `deferred-items.md`, a Qt6 blocker not a Qt5 build blocker). Paused at Task 3 — the blocking human-verify checkpoint.

**Stopped at:** 02-04-PLAN.md Task 3 — human-verify checkpoint (embedded molecule smoke check on macOS)

**Next action:** Human runs BALLView and verifies the smoke checklist (embedded molecule renders, rotate/zoom/pick, raytracer, text overlay, no GL error flood). On "approved", a continuation agent finalizes 02-04: creates `02-04-SUMMARY.md`, advances the plan counter, updates ROADMAP/REQUIREMENTS, makes the final docs commit.

**Notes:**

- GSD phase numbers are sequential (1-9); they map to ROADMAP-1.6.md phases 1, 4a, 2, 3, 4b, 5, 6, 7, 8 respectively. The reorder reflects that the rendering port (4a) is the immediate priority and only depends on the build baseline.
- A1 CONFIRMED → Plans 03/04 threading scope stays minimal: no `QOffscreenSurface` / shared context needed, just worker-thread `makeCurrent()` removal. Caveat documented in `02-A1-FINDINGS.md`: keep `TilingRenderer`'s GL path GUI-thread-only (it already is).

---
*State initialized: 2026-05-14*
