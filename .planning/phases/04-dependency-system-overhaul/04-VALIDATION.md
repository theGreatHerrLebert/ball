---
phase: 4
slug: dependency-system-overhaul
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-14
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | CMake configure + build + CTest; GitHub Actions matrix (macOS-arm64 / Linux / Windows) |
| **Config file** | `CMakePresets.json` (new this phase), `.github/workflows/ci.yml` |
| **Quick run command** | `cmake --preset macos-homebrew && cmake --build --preset macos-homebrew` |
| **Full suite command** | CI matrix run across all 4 jobs (macOS / Linux / Windows / ci) |
| **Estimated runtime** | ~local configure+build minutes; CI matrix ~tens of minutes |

---

## Sampling Rate

- **After every task commit:** Re-run `cmake --preset <platform>` configure and confirm the touched dependency resolves
- **After every plan wave:** Full local build on the development platform (macOS) + push to trigger CI matrix
- **Before `/gsd-verify-work`:** CI matrix green on all jobs — including the Windows job flipped to a required check (DEPS-02)
- **Max feedback latency:** local configure ~seconds; full CI ~tens of minutes

---

## Per-Task Verification Map

> Populated by the planner from the final PLAN.md task breakdown. Each task maps to a requirement and a verifiable command (CMake configure output, build success, or CI job status).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | DEPS-01..05, FEAT-01 | — | N/A | build/CI | TBD by planner | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `CMakePresets.json` — the configure presets that all subsequent verification depends on (D-07)
- [ ] Confirm CI matrix from Phase 02.2 still green as the regression baseline before dependency changes land

*Build-system phase — verification is configure/build/CI success, not a unit-test framework install.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| OpenBabel 3.x integration produces correct chemistry (implicit-H, aromaticity) | DEPS-04 / DEPS-05 | Semantic correctness of the 2.4→3.x port can't be proven by "it compiles" — needs a known-input check | Build with OpenBabel enabled, run `molecularSimilarity` against a reference molecule pair, compare similarity output to a pre-port baseline |
| BALLView still launches and renders after the dependency overhaul | DEPS-01 | GUI smoke — the dependency graph feeds the renderer | Launch `BALLView`, load a molecule, confirm the 3D scene renders (carries the Phase 2 render check forward) |

---

## Validation Sign-Off

- [ ] All tasks have an automated verify (configure/build/CI) or a Wave 0 dependency
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers `CMakePresets.json` (everything downstream configures through it)
- [ ] No watch-mode flags
- [ ] Feedback latency acceptable (local configure fast; CI matrix is the gate)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
