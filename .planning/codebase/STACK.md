# Technology Stack

**Analysis Date:** 2026-05-14

## Languages

**Primary:**
- C++ (C++17 standard, configured in `CMakeLists.txt:33`) — Core library and VIEW components
- Python (optional, disabled by default) — Python bindings via SIP (not currently built)

**Build/Configuration:**
- CMake (minimum version 3.5, tested up to 3.31 in `CMakeLists.txt:8`)
- Flex (lexical analyzer generator) — Used for format parsing (`CMakeLists.txt:131`)
- Bison (parser generator) — Used for format parsing (`CMakeLists.txt:121`)

## Runtime

**Environment:**
- Cross-platform: macOS (Apple Silicon + Intel), Linux (various distributions), Windows
- Native OS libraries: POSIX systems (Linux, macOS) and Win32 (Windows)

**Package Manager:**
- CMake with system/Homebrew packages on macOS/Linux
- vcpkg planned for Windows (currently legacy `ball_contrib` is deprecated; ROADMAP-1.6.md indicates vcpkg migration is pending)
- No lockfile needed (all dependencies resolved via CMake find_package)

## Frameworks

**Core:**
- Qt 5 (minimum 5.5, tested with 5.15.18 on Homebrew) — GUI framework for BALLView and VIEW widgets
  - Components: Core, Network, Xml (always), OpenGL, PrintSupport, Test, Widgets (VIEW only), WebEngine (optional but disabled)
  - Configured in `CMakeLists.txt:300-344`

**Molecular Computing:**
- Boost (minimum 1.55, tested with 1.90+) — Asio, threading, serialization, regex
  - Components: chrono, date_time, iostreams, regex, serialization, thread (configured in `cmake/BALLConfigBoost.cmake`)
- Eigen3 (minimum 3.0, tested with 5.0.1) — Linear algebra, vector/matrix math
  - Configured in `CMakeLists.txt:359-361`

**Graphics & Rendering:**
- OpenGL (required for VIEW) — 3D graphics backend, fixed-function pipeline (compat profile)
  - GLU required; configured in `CMakeLists.txt:371-376`
- GLEW (OpenGL Extension Wrangler, 2.3.1+, optional but found) — OpenGL extension loading
  - Configured in `CMakeLists.txt:378-384`

**Parallel/Performance:**
- Intel Threading Building Blocks (TBB) (oneTBB 2023.0.0+, optional) — Parallelization for computationally intensive tasks
  - Enabled by default; configured in `CMakeLists.txt:226-256`

**Development & Build:**
- GNUInstallDirs — Standard CMake install paths; included in `CMakeLists.txt:86`

## Key Dependencies

**Critical (always required):**
- Flex — Lexical scanner for format parsing
- Bison — Parser for format parsing
- Qt5::Core, Qt5::Network, Qt5::Xml — Base Qt libraries
- Eigen3 — Mathematical foundation
- Boost (chrono, date_time, iostreams, regex, serialization, thread)

**For VIEW (visualization):**
- OpenGL/GLU — 3D rendering
- Qt5::OpenGL, Qt5::PrintSupport, Qt5::Test, Qt5::Widgets — Qt GUI components
- GLEW (optional if USE_GLEW=ON) — OpenGL extension handling

**Optional (enabled by default if found):**
- TBB (Intel Threading Building Blocks) — Parallel computation
- FFTW (Fast Fourier Transform) — Available only in GPL builds (`CMakeLists.txt:164-176`)
- OpenBabel 2.x (optional, off by default in current build) — Molecular format conversion
- lpsolve (linear programming solver, optional) — Linear optimization
- libSVM (optional, found and linked if available) — Machine learning support
- MPI (optional, off by default) — Parallel distributed computing
- CUDA (optional, off by default) — GPU acceleration (version ≤ 2.1 only)
- RTfact (optional, off by default) — Ray-tracing backend (Windows/Visual Studio only)

**Infrastructure:**
- XDR (External Data Representation) — Binary data serialization, required for MPI; configured in `cmake/BALLConfigXDR.cmake`
- Python3 (Interpreter + Development) — If `BALL_PYTHON_SUPPORT=ON`; configured in `CMakeLists.txt:422`
- SIP (System Information Provider, 4.9+, optional for Python) — Python binding generator (currently broken/disabled)

## Configuration

**Build Configuration:**
- CMake-driven; all options stored in generated `include/BALL/CONFIG/config.h` (template at `cmake/config.h.in`)
- Features detected at configure time:
  - System headers: `unistd.h`, `process.h`, `time.h`, `dirent.h`, `pwd.h`, `stdint.h` etc. (see `cmake/BALLConfiguration.cmake:106-119`)
  - Endianness (big-endian / little-endian detection)
  - Compiler support for deprecated C++ constructs (e.g., `std::unary_function`, which exist in C++14 but not C++17)
  - Function availability: `kill`, `sysconf`

**CMake Build Types:**
- Debug (default when not specified): generates `BALLd.dll` on MSVC (debug suffix)
- Release (recommended for production)
- MinSizeRel, RelWithDebInfo (supported but less common)

**Generated Configuration Files:**
- `include/BALL/CONFIG/config.h` — Main configuration header, auto-generated from `cmake/config.h.in`
- `include/BALL/PYTHON/BALLPythonConfig.h` — Python-specific config (from `cmake/BALLPythonConfig.h.in`)
- `test/BALLTestConfig.h` — Test suite config (from `cmake/BALLTestConfig.h.in`)
- `source/BENCHMARKS/BALLBenchmarkConfig.h` — Benchmark config (from `cmake/BALLBenchmarkConfig.h.in`)

**Platform-Specific:**
- macOS: `BALL_OS_DARWIN=TRUE`; uses Clang compiler (`BALL_COMPILER_LLVM`)
- Linux: `BALL_OS_LINUX=TRUE`; uses GCC (`BALL_COMPILER_GXX`) or Clang
- Windows: `BALL_OS_WINDOWS=TRUE`, `BALL_PLATFORM_WINDOWS=TRUE`; uses MSVC (`BALL_COMPILER_MSVC`)

**Preprocessor Defines:**
- `BALL_DEBUG` — Set in debug builds
- `BALL_BUILD_DLL` — For MSVC DLL export
- `BALL_VIEW_BUILD_DLL` — For VIEW library MSVC export
- `BALL_HAS_*` — Feature flags (GLEW, VIEW, TBB, FFTW, OPENBABEL, LPSOLVE, LIBSVM, MPI, CUDA, QTWEBENGINE, RTFACT)

## Build Output

**Libraries:**
- `libBALL.so` (shared) or `libBALL.a` (static if `BUILD_SHARED_LIBS=FALSE`) — Main BALL library
  - Linked libraries: Boost, Qt5, Eigen, XDR, FFTW (GPL only), lpsolve, MPI, Python, TBB, libSVM
  - Binary output: `{build_dir}/lib/libBALL.so`
- `libVIEW.so` (shared) — Visualization library (if `BALL_HAS_VIEW=ON`)
  - Linked libraries: BALL, Qt5 (OpenGL, PrintSupport, Test, Widgets), OpenGL, GLEW, RTfact
  - Binary output: `{build_dir}/lib/libVIEW.so`

**Executables:**
- `BALLView` (GUI application) — Molecular visualization application; located at `{build_dir}/bin/BALLView.app/Contents/MacOS/BALLView` on macOS
  - Requires: BALL, VIEW, Qt5, OpenGL
  - Requires runtime data: `BALL_DATA_PATH` environment variable (defaults to `{install_prefix}/share/BALL/data`)

**Test & Benchmark Binaries:**
- Generated from test and benchmark source files; linked with gtest or similar framework

## Platform Requirements

**Development (macOS example from ROADMAP-1.6.md):**
- CMake 3.5+ (tested with 3.31)
- Clang/Apple Clang (C++14 minimum, C++17 optimal)
- Homebrew packages: `qt@5`, `boost`, `eigen`, `fftw`, `tbb`, `glew`, `flex`, `bison`, `lp_solve`, `libsvm`
- Optional: `open-babel` (API 2.x or 3.x requires porting)
- DYLD_LIBRARY_PATH or RPATH linking for shared libraries

**Production (cross-platform targets):**
- macOS: Apple Silicon (arm64) or Intel (x86_64), Monterey or later
- Linux: glibc 2.29+, modern GCC/Clang
- Windows: MSVC 2019+, Visual Studio build tools, vcpkg for dependencies (migration in progress)
- OpenGL 2.1+ support (fixed-function / compatibility profile)
- Sufficient RAM for TBB thread pools during molecular computations

**Data & Runtime:**
- `BALL_DATA_PATH` must point to the `data/` directory (containing FragmentDB, element colors, forcefields)
- Locale support (Qt uses system locale; NMR, JCAMP data parsing may depend on LC_NUMERIC)

---

*Stack analysis: 2026-05-14*
