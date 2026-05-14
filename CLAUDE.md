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

C++17 (set via `CMAKE_CXX_STANDARD` since Phase 3), built with CMake 3.5+ (tested to 3.31). Flex + Bison generate the format parsers. Cross-platform: macOS (Apple Silicon + Intel), Linux, Windows.

**Required:** Qt 5 (5.15.x via Homebrew — `Core`, `Network`, `Xml`, `OpenGL`, `PrintSupport`, `Widgets`), Boost (chrono/date_time/iostreams/regex/serialization/thread), Eigen3 (5.x), OpenGL/GLU. **Optional (compile-time `BALL_HAS_*` flags):** GLEW, TBB (oneTBB), FFTW (GPL builds), OpenBabel 2.x, lp_solve, libSVM, MPI, CUDA. **Removed/disabled:** RTfact, QtWebEngine, SIP Python bindings.

Dependencies come from Homebrew/system on macOS/Linux; vcpkg for Windows is pending (Phase 4). The legacy `ball_contrib` tree is deprecated and being removed from the build path. Build config is generated into `include/BALL/CONFIG/config.h` from `cmake/config.h.in`. Build output: `libBALL`, `libVIEW`, and the `BALLView` app; `BALL_DATA_PATH` must point at `data/` at runtime.

Full detail: `.planning/codebase/STACK.md`, `.planning/codebase/INTEGRATIONS.md`.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Headers are `.h`, implementation files are `.C` (matching names). Every file starts with the modeline `// -*- Mode: C++; tab-width: 2; -*-` + `// vi: set ts=2:` — **2-space indentation**, Allman braces. Tests are `{Component}_test.C` / `{Component}_test{N}.C`, co-located in `test/`.

Naming: classes PascalCase with the `BALL_EXPORT` macro for shared-library visibility; methods camelCase with `get*`/`set*`/`has*`/`is*`/`create*`/`destroy*` prefixes; member variables camelCase with a **trailing underscore** (`type_`, `descriptor_matrix_`); constants/macros `UPPER_CASE`, defaults as `BALL_{CLASS}_DEFAULT_*`. Include guards are `BALL_{MODULE}_{FILE}_H`; internal includes are guarded and tab-indented (`#\tinclude <BALL/...>`).

Everything lives in `namespace BALL`, submodules nest (`BALL::QSAR`). Source files typically declare `using namespace BALL;` after includes. Errors throw the `BALL::Exception` hierarchy with `__FILE__, __LINE__, message`; `BALL::Log` handles warnings/errors. Public APIs require Doxygen `/** ... */` with `@param`/`@return`/`@see`.

Full detail: `.planning/codebase/CONVENTIONS.md`.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Layered, modular C++ library. Foundational layers — `COMMON` (exceptions, `LogStream`, hash containers), `CONCEPT` (the core patterns), `DATATYPE`, `MATHS`, `SYSTEM` — sit under `KERNEL` (the molecular hierarchy: `System → Molecule → Residue → Chain → Atom → Bond`). Domain layers (`FORMAT`, `STRUCTURE`, `ENERGY`, `MOLMEC`, `NMR`, `SOLVATION`, `QSAR`, `DOCKING`, `SCORING`, `XRAY`) build on `KERNEL`. `VIEW` is the Qt/OpenGL rendering + GUI layer; `PYTHON` (SIP, currently disabled) sits on top. Code is split `include/BALL/<LAYER>/` + `source/<LAYER>/`.

Two patterns dominate: **Composite** (`CONCEPT/composite.h`) — the structural tree, with `apply()` traversal — and **Processor / `UnaryProcessor<T>`** (`CONCEPT/processor.h`) — the visitor used for structure traversal, model building, and energy calculation, with `start()/operator()/finish()` and `CONTINUE/BREAK/ABORT` control flow.

Rendering goes through a pluggable-backend abstraction: `ModelProcessor`s turn the molecular tree into primitives → `RenderSetup` batches them → `RendererFactory` picks a `Renderer` (`GLRenderer` for interactive, plus POV-Ray/raytracing/etc.) → draws into a `RenderWindow`/`RenderSurface` (the post-Phase-2 `GLRenderWindow` is a `QOpenGLWidget`). The `Renderer`/`RenderSurface` boundary (Phase 02.1, see `.planning/RENDERER-INTERFACE-BOUNDARY.md`) is the stable contract for the Phase 5 backend swap. Entry point: `source/APPLICATIONS/BALLVIEW/main.C` → `MainFrame` → `Scene` widget.

Full detail: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.
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
