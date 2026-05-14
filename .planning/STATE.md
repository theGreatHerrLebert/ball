# STATE: BALLView 1.6 Modernization

## Project Reference

**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies — the 3D scene working cross-platform is the non-negotiable outcome.

**Current Focus:** Phase 1 — Build Baseline (then Phase 2, the immediate-priority rendering port).

## Current Position

**Phase:** 1 — Build Baseline
**Plan:** None yet
**Status:** Roadmap created, awaiting phase planning
**Progress:** `[ ]` 0/9 phases complete

```
Phase 1  [ ]  Build Baseline
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
| Phases complete | 0/9 |
| Plans complete | 0 |
| Requirements mapped | 27/27 |
| Milestone | v1.6 |

## Accumulated Context

### Decisions
- Abandon `ball_contrib`; build against Homebrew/system deps (macOS/Linux) and vcpkg (Windows). Already proven on macOS Tahoe.
- Phase 4a (GSD Phase 2): `QOpenGLWidget` + compatibility profile, deferring Qt 6 and the pipeline rewrite to Phase 4b (GSD Phase 5). Smallest change that restores rendering on all 3 OSes.
- Phase 4a threading: hybrid model — interactive GL on the GUI thread via `paintGL()`, raytracer stays a CPU-buffer worker thread.
- C++14 is a load-bearing bridge until GSD Phase 3 (Language Modernization) removes it.
- GSD Phase 2 (Rendering) is sequenced right after the build baseline because it only needs the working build, not the C++17 or dependency-overhaul phases.

### Todos
- None yet (roadmap just created).

### Blockers
- None. The known blank-3D-scene blocker is the target of Phase 2, not an obstacle to planning.

## Session Continuity

**Last action:** Roadmap and STATE initialized from PROJECT.md, REQUIREMENTS.md (v1.6), and the human-authored ROADMAP-1.6.md.

**Next action:** Plan Phase 1 via `/gsd-plan-phase 1`.

**Notes:**
- GSD phase numbers are sequential (1-9); they map to ROADMAP-1.6.md phases 1, 4a, 2, 3, 4b, 5, 6, 7, 8 respectively. The reorder reflects that the rendering port (4a) is the immediate priority and only depends on the build baseline.
- 8 modern-toolchain patches are applied locally but uncommitted — Phase 1 commits them.

---
*State initialized: 2026-05-14*
