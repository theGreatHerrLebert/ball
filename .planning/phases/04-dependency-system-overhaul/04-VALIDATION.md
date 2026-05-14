---
phase: 4
slug: dependency-system-overhaul
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-14
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | CMake configure + build; GitHub Actions matrix (macOS-arm64 / Linux / Windows) + one standalone OpenBabel chemistry smoke driver |
| **Config file** | `CMakePresets.json` (new this phase), `vcpkg.json` (new this phase), `.github/workflows/ci.yml` |
| **Quick run command** | `cmake --preset macos-homebrew && cmake --build --preset macos-homebrew --target BALL VIEW BALLView` |
| **Full suite command** | CI matrix run across all jobs (build macos / linux / windows + lint), render smoke (macOS/Linux), OpenBabel smoke (macOS/Linux) |
| **Estimated runtime** | local configure+build minutes; CI matrix ~tens of minutes |

---

## Sampling Rate

- **After every task commit:** Re-run `cmake --preset <platform>` configure (and build the touched target) and confirm the touched dependency resolves
- **After every plan wave:** Full local build on the development platform (macOS) + push to trigger the CI matrix
- **Before `/gsd-verify-work`:** CI matrix green on all jobs — including the Windows job flipped to a required check (DEPS-02)
- **Max feedback latency:** local configure ~seconds; full CI ~tens of minutes

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-T1 | 01 | 1 | DEPS-03 | T-04-01, T-04-03 | Config-mode finders carry pinned min versions; no untrusted finder substitution | build | `cmake -S . -B build/04-verify ... && cmake --build build/04-verify --target BALL`; configure log shows `Found TBB ... config mode` | ✅ CMakeLists.txt | ⬜ pending |
| 04-01-T2 | 01 | 1 | DEPS-01, DEPS-04 | T-04-02 | `BALL_CONTRIB_PATH` configure-time injection vector removed; min versions pinned | configure | `grep -rn BALL_CONTRIB_PATH CMakeLists.txt cmake/` clean (bar FindSIP comment) + `cmake -S . -B build/04-verify2` exits 0 | ✅ CMakeLists.txt | ⬜ pending |
| 04-01-T3 | 01 | 1 | DEPS-03 | — | OpenBabel GPL-gate decision recorded, not silently changed | checkpoint | maintainer selects lgpl-ok / gpl-gated | ✅ checkpoint | ⬜ pending |
| 04-02-T1 | 02 | 1 | DEPS-05 | T-04-05, T-04-06 | Presets are version-controlled; `windows-vcpkg` toolchainFile resolution documented | build | `python3 -c "import json; ..."` + `cmake --preset macos-homebrew` exits 0 | ❌ → CMakePresets.json | ⬜ pending |
| 04-02-T2 | 02 | 1 | DEPS-05 | T-04-08 | CI invokes version-controlled presets; resolved flags auditable via `cmake --preset -N` | build/CI | `grep "preset macos-homebrew" BUILD-macos.md` + `grep "preset ci-" ci.yml` + YAML parse | ✅ ci.yml | ⬜ pending |
| 04-02-T3 | 02 | 1 | FEAT-01 | — | N/A (doc) | doc review | `grep -nE "not a vcpkg port\|non-PIC\|auto-disable" REQUIREMENTS.md` | ✅ REQUIREMENTS.md | ⬜ pending |
| 04-03-T1 | 03 | 2 | DEPS-05 (D-05) | T-04-09, T-04-10 | OpenBabel 3.x port; pinned to >=3.0 so 2.x parser CVEs N/A | build | `cmake --build build/macos-homebrew --target BALL` with `BALL_HAS_OPENBABEL=ON` exits 0; no 2.x symbols remain | ✅ molecularSimilarity.C | ⬜ pending |
| 04-03-T2 | 03 | 2 | DEPS-05 (D-05) | T-04-09 | TOOLS ported to 3.x API; no 2.x symbols | build | `cmake --build ... --target MolDepict ProteinProtonator Ligand3DGenerator` exits 0 | ✅ TOOLS *.C | ⬜ pending |
| 04-03-T3 | 03 | 2 | DEPS-05 (D-05) | T-04-10, T-04-11 | Smoke driver catches silent implicit-H/aromaticity corruption | smoke | `bash .planning/phases/04-.../scripts/openbabel-smoke.sh` prints `OPENBABEL_SMOKE_OK` | ❌ → openbabel-smoke.{C,sh} | ⬜ pending |
| 04-03-T4 | 03 | 2 | DEPS-05 (D-05) | T-04-11 | CI exercises the port; smoke step version-controlled, read-only-token runner | CI | `grep -nE "BALL_HAS_OPENBABEL=ON\|openbabel-smoke.sh" ci.yml` + YAML parse + green CI | ✅ ci.yml | ⬜ pending |
| 04-04-T1 | 04 | 2 | DEPS-02 | T-04-13, T-04-15 | `vcpkg.json` baseline pinned to a 40-char SHA; no non-port deps | configure | `python3 -c "..."` asserts pinned baseline + verified port set | ❌ → vcpkg.json | ⬜ pending |
| 04-04-T2 | 04 | 2 | DEPS-02 | T-04-14, T-04-16, T-04-17 | choco pkg name pinned; vcpkg cached; ephemeral read-only runner | build/CI | `grep -nE "winflexbison3\|preset ci-windows" ci.yml` + no `\|\| true` + Windows CI builds green | ✅ ci.yml | ⬜ pending |
| 04-04-T3 | 04 | 2 | DEPS-02 | — | Windows green on 2+ runs before flipped to required | checkpoint | maintainer confirms Windows job green 2x + `blocking: true` committed + green | ✅ checkpoint | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

> "Wave 0" in 04-RESEARCH.md maps to **Wave 1** in the final plan numbering (Plans 01 + 02).

- [ ] `CMakePresets.json` (Plan 02 / Wave 1) — the configure presets that all subsequent verification depends on (D-07)
- [ ] The 3 config-mode finder migrations + `ball_contrib` removal + version pins (Plan 01 / Wave 1) — the CMake foundation Plans 03 + 04 build on
- [ ] Confirm CI matrix from Phase 02.2 still green as the regression baseline before dependency changes land
- [ ] CI preset/configure must enable `BALL_HAS_OPENBABEL=ON` on macOS/Linux (Plan 03 / Wave 2) so the D-05 port is genuinely exercised — the one real verification gap the research flagged

*Build-system phase — verification is configure/build/CI success plus one OpenBabel chemistry smoke driver, not a unit-test framework install.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| OpenBabel 3.x integration produces correct chemistry (implicit-H, aromaticity) | DEPS-05 (D-05) | Semantic correctness of the 2.4→3.x port can't be fully proven by "it compiles" — partly automated now by the Plan 03 Task 3 smoke driver (non-empty + deterministic + aromaticity-correct SMILES); a deeper reference-molecule comparison stays manual | Build with OpenBabel enabled, run `molecularSimilarity` against a reference molecule pair, compare similarity output to a pre-port baseline |
| BALLView still launches and renders after the dependency overhaul | DEPS-01 | GUI smoke — the dependency graph feeds the renderer; the Phase 02.2 render smoke check carries this on macOS/Linux but a human launch confirms no regression | Launch `BALLView`, load a molecule, confirm the 3D scene renders |
| Windows build is genuinely, repeatably green before being made required | DEPS-02 | "Windows green" is only verifiable via CI; a one-off green can be a fluke (Pitfall 2) — needs human judgement on repeatability before flipping `blocking: true` | Plan 04 Task 3 checkpoint: confirm the Windows CI job green on 2+ consecutive runs, then flip `blocking: true` |

---

## Validation Sign-Off

- [x] All tasks have an automated verify (configure/build/CI/smoke) or are an explicit checkpoint
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (the 2 checkpoints are bracketed by automated-verify tasks)
- [x] Wave 1 covers `CMakePresets.json` + the finder/contrib foundation (everything downstream configures through them)
- [x] No watch-mode flags
- [x] Feedback latency acceptable (local configure fast; CI matrix is the gate)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planned — ready for `/gsd-execute-phase 4`
