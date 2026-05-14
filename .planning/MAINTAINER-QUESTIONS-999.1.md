# Backlog 999.1 — Maintainer UI/UX Decisions (issue-ready)

**Status:** Write-up ready · 2026-05-14 · awaiting publication to maintainers
**Why this exists:** The 4 questions below need BALL-maintainer decisions and have a
long lead time. They gate the **BALLView Refresh** milestone (target 1.7), which is
itself gated on the 1.6 foundation milestone's Phase 5 (Qt 6). Raising them now.

**Publish as:** a GitHub issue on `BALL-Project/ball`, or a maintainers'-list thread.
(Automated issue creation was blocked as an unauthorized external write — publish
manually, or explicitly authorize `gh issue create`.)

---

## Suggested issue title

`BALLView Refresh (1.7): 4 UI/UX decisions needed from maintainers`

## Suggested issue body

### Context

The **BALLView Refresh** UI/UX modernization milestone (target 1.7, the "Claude
Design Handover" package) is gated on the 1.6 foundation milestone's Qt 6 port
(Phase 5). Four design decisions need maintainer input and have a long lead time —
raising them now so they're settled well before the UI milestone's Inspector and
menu phases start.

Tracked internally as backlog item **999.1** (`.planning/ROADMAP.md`); see
`.planning/DESIGN-HANDOVER-INTEGRATION.md` and
`.planning/seeds/SEED-001-ballview-refresh-ui-milestone.md` for full context.

### Decisions needed

**1. macOS menu bar**
Keep the current inline menu bar, or adopt the native global macOS menubar via
`QAction::setMenuRole`?

**2. "Classic" workspace**
Keep the legacy 5-dock workspace layout as a long-term opt-in preset, or retire it
after one release once the consolidated workspace ships?

**3. Theme picker scope**
Ship a single neutral theme, or expose Light / Dark / Follow-System?
*(Handover recommendation: Follow-System, no custom color picker.)*

**4. Translation churn**
The Phase 6 menu re-org invalidates roughly 40% of `BALLView-de_DE.ts` strings.
Accept this and plan a community translation round — and if so, when?

### Why now

These block the UI milestone's Phase 4 (unified Inspector) and Phase 6 (menu
re-org). They need a deliberate decision rather than a default; translation
coordination especially needs lead time.

---
*When published, record the issue URL here and update backlog 999.1 in `ROADMAP.md`.*
