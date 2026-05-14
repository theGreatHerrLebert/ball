---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 02-02-PLAN.md
last_updated: "2026-05-14T07:40:55.526Z"
progress:
  total_phases: 9
  completed_phases: 1
  total_plans: 5
  completed_plans: 3
  percent: 60
---

# STATE: BALLView 1.6 Modernization

## Project Reference

**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies — the 3D scene working cross-platform is the non-negotiable outcome.

**Current Focus:** Phase 2 — Rendering Port (4a)

## Current Position

Phase: 2 (Rendering Port (4a)) — EXECUTING
Plan: 3 of 4
**Phase:** 2
**Plan:** 02 complete; next is 03
**Status:** Executing Phase 2
**Progress:** [██████░░░░] 60%

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

### Todos

- None yet (roadmap just created).

### Blockers

- None. The known blank-3D-scene blocker is the target of Phase 2, not an obstacle to planning.

## Session Continuity

**Last action:** Completed 02-02-PLAN.md — rebased `GLRenderWindow` from `QGLWidget` onto `QOpenGLWidget`: compat-profile `QSurfaceFormat`, GL work moved into `initializeGL`/`resizeGL`/`paintGL`, manual buffer-swap machinery (`safeBufferSwap`/`setAutoBufferSwap`) deleted, on-screen text reimplemented as a `QPainter` overlay, and global context sharing + default surface format wired into BALLView `main()`. `glRenderWindow.{h,C}` + `main.C` compile clean as TUs; remaining `make VIEW` errors are isolated to the not-yet-ported downstream files (`glOffscreenTarget.C`, `renderSetup.C`) — expected, handed to Plans 03/04.

**Stopped at:** Completed 02-02-PLAN.md

**Next action:** Execute 02-03-PLAN.md.

**Notes:**

- GSD phase numbers are sequential (1-9); they map to ROADMAP-1.6.md phases 1, 4a, 2, 3, 4b, 5, 6, 7, 8 respectively. The reorder reflects that the rendering port (4a) is the immediate priority and only depends on the build baseline.
- A1 CONFIRMED → Plans 03/04 threading scope stays minimal: no `QOffscreenSurface` / shared context needed, just worker-thread `makeCurrent()` removal. Caveat documented in `02-A1-FINDINGS.md`: keep `TilingRenderer`'s GL path GUI-thread-only (it already is).

---
*State initialized: 2026-05-14*
