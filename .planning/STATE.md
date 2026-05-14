---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: milestone
status: executing
stopped_at: Phase 2 complete; Phase 02.1 ready to plan.
last_updated: "2026-05-14T09:51:39.553Z"
progress:
  total_phases: 15
  completed_phases: 2
  total_plans: 8
  completed_plans: 5
  percent: 63
---

# STATE: BALLView 1.6 Modernization

## Project Reference

**Core Value:** BALLView must build and visibly render molecules on macOS, Linux, and Windows from current, supported dependencies — the 3D scene working cross-platform is the non-negotiable outcome.

**Current Focus:** Phase 02.1 — Renderer boundary extraction (next to plan)

## Current Position

**Phase:** 02.1 — Renderer boundary extraction
**Plan:** Not started
**Status:** Ready to execute
**Progress:** Phases 1 + 2 complete

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
| Plans complete | 5 |
| Requirements | 30 active + NET-01 deferred |
| Milestone | v1.6 |
| Phase 01-build-baseline P01 | 3min | 3 tasks | 9 files |
| Phase 02-rendering-port-4a P01 | 3min | 2 tasks | 2 files |
| Phase 02-rendering-port-4a P02 | 12min | 2 tasks | 3 files |
| Phase 02-rendering-port-4a P03 | 9min | 2 tasks | 4 files |
| Phase 02-rendering-port-4a P04 | ~4h (incl. 3 debug rounds + human verify) | 3 tasks | 7 files |

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

### Roadmap Evolution

- Phase 02.1 inserted after Phase 2: Renderer boundary extraction — pure refactor that makes Phase 5 a contained backend swap. Depends on Phase 2, blocks Phase 5. Design: `.planning/RENDERER-INTERFACE-BOUNDARY.md`.
- Design Handover package analyzed (`.planning/DESIGN-HANDOVER-INTEGRATION.md`): a separate UI/UX modernization milestone (~8 phases) that depends on this milestone's Phase 5 (Qt 6). Planted as SEED-001 (Milestone 2 "BALLView Refresh", target 1.7); not folded into the current roadmap.
- Codex adversarial review run on the roadmap. Cheap fixes applied (numbering normalized, status accuracy, QT6-03/PIPE-01 contradiction removed, networking deferred to backlog 999.3, version policy). Structural changes still pending: insert an early CI phase, split Phase 5, add a graphics-diagnostics requirement + feature matrix.
- Backlog: 999.1 (UI maintainer open-questions), 999.2 (Ninja generator), 999.3 (networking rework).

### Todos

- (none open)

### Blockers

- None.

## Session Continuity

**Last action:** Phase 2 (Rendering Port) marked COMPLETE — human visual verification passed ("Approved"), gsd-verifier passed 5/5, `phase complete 02` run. The QGLWidget→QOpenGLWidget port is functionally done on macOS: molecule renders embedded, ball-and-stick correct, survives resize, rotate/zoom/pick work, clean startup log.

**Stopped at:** Phase 2 complete; Phase 02.1 ready to plan.

**Next action:** `/gsd-plan-phase 02.1` — Renderer boundary extraction (the design is fully drafted in `.planning/RENDERER-INTERFACE-BOUNDARY.md`, so research can be skipped). Alternatively, apply the Codex review's structural roadmap changes first (early CI phase, Phase 5 split) before continuing — user's call.

**Notes:**

- GSD phase numbers are sequential (1-9); they map to ROADMAP-1.6.md phases 1, 4a, 2, 3, 4b, 5, 6, 7, 8 respectively. The reorder reflects that the rendering port (4a) is the immediate priority and only depends on the build baseline.
- A1 CONFIRMED → Plans 03/04 threading scope stays minimal: no `QOffscreenSurface` / shared context needed, just worker-thread `makeCurrent()` removal. Caveat documented in `02-A1-FINDINGS.md`: keep `TilingRenderer`'s GL path GUI-thread-only (it already is).

---
*State initialized: 2026-05-14*
