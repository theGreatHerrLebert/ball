<!-- GSD:project-start source:PROJECT.md -->
## Project

**BALLView 1.6 Modernization**

BALL (Biochemical ALgorithms Library) is a C++ molecular modelling framework, and
BALLView is its OpenGL-based molecular visualization GUI application. This project
modernizes the BALL/VIEW/BALLView stack — frozen at a 2022 commit against ~2016-era
dependencies — so it builds, runs, and renders on current macOS (Apple Silicon),
Linux, and Windows toolchains, culminating in a shippable v1.6 release.

**Core Value:** BALLView must **build and visibly render molecules** on macOS, Linux, and Windows
from current, supported dependencies — the 3D scene working cross-platform is the
non-negotiable outcome.

### Constraints

- **Compatibility**: Must build and render on macOS (Apple Silicon), Linux, and Windows — platform independence is a hard requirement, no per-OS graphics code if avoidable.
- **Tech stack**: C++ / CMake / Qt. Phase 4a stays on Qt 5.15 + fixed-function GL (compat profile); Qt 6 is Phase 4b.
- **Dependencies**: System/Homebrew packages on macOS/Linux, vcpkg on Windows. No reliance on the dead `ball_contrib`.
- **Risk**: Phase 4a touches the threaded renderer — the highest-risk area. Threading rework must be incremental and verifiable against a running GUI.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
