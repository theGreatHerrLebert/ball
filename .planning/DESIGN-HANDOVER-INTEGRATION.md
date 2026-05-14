# Claude Design Handover — Analysis & Roadmap Fit

**Status:** Analysis · 2026-05-14
**Source package:** `/Users/kohlbach/Claude/BALL/Claude Design Handover/`
(`START-HERE.md`, `BALLView UI Audit.html`, `revitalization/` — 8 phase docs +
cross-platform + migration playbook + mockups)

---

## What the package is

A **UI/UX modernization plan for BALLView's surface** — theming, icons,
dialogs, a unified Inspector, workspace layout, menu re-org + command palette,
onboarding, accessibility. 8 phases, ~6 person-months, with mockups, a QSS
design-token system, a CI matrix, branching model, and a release cadence.

**It is explicitly NOT the renderer work.** The package states up front:

> "The renderer migration (`QGLWidget` → `QOpenGLWidget`, Qt 6 port) is tracked
> separately and is not part of this package. Everything here assumes that
> work happens in parallel or has landed first."

So it is a **complement** to the current GSD milestone, not a competitor —
but it does not slot into the existing 9-phase roadmap as-is. Three things
have to be reconciled first.

## Conflict 1 — the version-numbering collision (must resolve)

Two different definitions of "1.6" are now in play:

| | This GSD milestone | Design Handover |
|--|--------------------|-----------------|
| What "1.6" means | Toolchain + renderer + Qt6 + deps + packaging + CI modernization | The UI Refresh marketing release (`BALL_UI_V2` ON by default) |
| Cadence | One milestone | 1.5.1 → 1.5.2 → 1.5.3 → **1.6.0** → 1.6.1 |

These cannot both own "1.6.0". **Recommendation:** the foundation work
(this milestone) is the legitimate **1.6** release — it is what makes BALL
build and render on modern toolchains at all. The UI Refresh becomes the
**next milestone, targeting 1.7** (or a "2.0" if marketing prefers). The
Handover's internal "1.5.x → 1.6.0" cadence should be re-based to
"1.6.x → 1.7.0" when it is adopted. Phase 1 already set the CMake version to
`1.6.0-dev` for the foundation milestone — that stays.

## Conflict 2 — hard dependency on Qt 6.5

The Handover's **Phase 0** (design-system foundation) depends on
`QStyleHints::colorScheme()`, which is **Qt 6.5+**. Its cross-platform doc
sets a hard "Qt 6.5 LTS floor" for all 8 phases.

This GSD milestone is still on **Qt 5.15** until **Phase 5 (Qt 6 + Pipeline)**.

→ **The entire Design Handover is blocked on this milestone's Phase 5.** It
cannot start until the Qt 6 port lands. This is a clean, hard ordering: it
does not change the current roadmap, it just sequences *after* it.

## Conflict 3 — overlapping cross-cutting machinery

The Handover ships its own CI matrix, packaging notes, and a `BALL_UI_V2`
flag. Several pieces overlap with phases already in this milestone:

| Handover artifact | Overlaps with | Resolution |
|-------------------|---------------|------------|
| CI matrix (`09-cross-platform.md`) — Linux/macOS/Windows × `ui_v2 ON/OFF` | **Phase 9 (CI & Tests)** | Phase 9 builds the CI matrix *once*; the Handover adds the `ui_v2` axis later. Don't build CI twice. |
| macOS notarized `.dmg`, entitlements | **Phase 8 (macOS Packaging)** | Phase 8 owns the packaging pipeline; Handover's packaging notes are a *checklist input* to Phase 8, not separate work. |
| `BALL_UI_V2` CMake option | (new) | Belongs to the UI milestone, introduced in its Phase 0. No conflict — just don't pre-create it. |
| Config-path migration (`~/.BALLView` → XDG/Library/AppData) | (new) | New work, lives in the UI milestone. |
| "renderer migration parallel" assumption | **Phases 2 + 5** | Already satisfied — Phase 2 is the QOpenGLWidget port, Phase 5 is Qt 6. The Handover's assumption is *met by* this milestone. |

## Recommended fit: a second milestone, gated on Phase 5

Do **not** renumber or expand the current 9-phase roadmap. Instead:

1. **Keep the current milestone as-is** ("BALLView 1.6 — toolchain/renderer
   modernization"). It is the foundation and the Handover explicitly depends
   on it.
2. **Register the Design Handover as Milestone 2** ("BALLView Refresh / UI v2",
   target 1.7) — its 8 phases become that milestone's roadmap. Its
   `revitalization/*.md` docs are already in the GSD phase-doc shape (Goal /
   Scope / Steps / Files / Acceptance / Cross-platform / Risks), so they drop
   in with light adaptation.
3. **Hard dependency:** Milestone 2 Phase 0 `depends_on` this milestone's
   **Phase 5** (Qt 6). Surface this as a blocker in `STATE.md` so it can't be
   started early.
4. **De-duplicate the cross-cutting work** before Milestone 2 planning:
   - Phase 9 (CI) here builds the base 3-OS matrix; Milestone 2 adds the
     `ui_v2` axis to the *existing* workflow.
   - Phase 8 (macOS Packaging) here consumes the Handover's packaging
     checklist as input.
5. **Surface the 4 open questions now.** The Handover lists 4 questions that
   "must be resolved with BALL maintainers before Phase 4 starts" (macOS
   native menubar, retire the legacy 5-dock workspace?, theme picker scope,
   translation-churn plan). These need a maintainer decision and have a long
   lead time — capture them as a backlog/seed item now, don't wait.

## Concrete GSD actions (when the user approves this fit)

- `/gsd-plant-seed` — plant the "BALLView Refresh / UI v2 milestone" with
  trigger condition = "Phase 5 (Qt 6) complete". It then surfaces
  automatically at the right moment instead of being forgotten.
- `/gsd-add-backlog` — add the 4 maintainer open-questions as a backlog item
  (999.x) so they're tracked and can be raised with maintainers early.
- At Phase 9 planning: explicitly note "CI matrix must be `ui_v2`-axis-ready"
  so Milestone 2 extends rather than rebuilds it.
- When Phase 5 completes: `/gsd-new-milestone` for "BALLView Refresh",
  importing `revitalization/*.md` as the phase set (the docs are already
  GSD-shaped — `/gsd-import` is the natural tool).

## One genuine cross-milestone risk to record

The Handover's Phase 4 (unified Inspector) **replaces 8+ modal settings
dialogs**. Several of those dialogs touch the *renderer/scene* settings
(material, coloring, stage, clipping). If Phase 5 (this milestone) reworks
the rendering pipeline, the Inspector rebuild in Milestone 2 must target the
*post-Phase-5* renderer API, not today's. → Another reason Milestone 2 phases
strictly follow Phase 5, and a reason the **renderer-interface boundary**
(see `RENDERER-INTERFACE-BOUNDARY.md`) matters: a stable `Renderer` /
`RenderSurface` contract is exactly what the Inspector rebuild should bind to.

## Summary

| Question | Answer |
|----------|--------|
| Does it fit the current roadmap? | Not *inside* it — it is a **separate, larger milestone** that depends on this one. |
| Does it conflict with current work? | Only on **version numbering** (resolve: foundation = 1.6, refresh = 1.7) and **cross-cutting CI/packaging** (resolve: de-dupe, don't rebuild). The renderer work it assumes is already this milestone's Phase 2 + 5. |
| Blocking dependency? | Yes — its Phase 0 needs Qt 6.5, i.e. **this milestone's Phase 5**. |
| Recommended action | Plant it as Milestone 2, gate on Phase 5, surface its 4 open questions to maintainers now. |
