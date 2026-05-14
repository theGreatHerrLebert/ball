# External Integrations

**Analysis Date:** 2026-05-14

## File Format Support

BALL is primarily a molecular file I/O and computation library. Extensive format support is implemented in `source/FORMAT/` and `include/BALL/FORMAT/`.

**Molecular Structure Formats:**
- **PDB** (Protein Data Bank) — `PDBFile.h`, `PDBFileGeneral.C`, `PDBFileDetails.C`
  - Standard format for protein structures; includes legacy PDB records
  - Handlers: `GenericPDBFile` for flexible parsing, `PDBFile` for strict format
- **MOL/SDF** (Structure Data Format, SDFile) — `MOLFile.h`, `MOLFile.C`, `SDFile.h`, `SDFile.C`
  - Chemical structure with optional data records (SDF)
  - Used for small molecules and chemical databases
- **MOL2** — `MOL2File.h`, `MOL2File.C`
  - Tripos MOL2 format with atom types and bonds
- **XYZ** — `XYZFile.h`, `XYZFile.C`
  - Simple XYZ coordinate format
- **HIN** — `HINFile.h`, `HINFile.C`
  - HyperChem format
- **JCAMP** — `JCAMPFile.h`, `JCAMPFile.C`
  - Joint Committee on Atomic and Molecular Physical Data format
- **CIF** — `CIFFile.h`, `CIFFile.C`
  - Crystallographic Information File (uses Flex/Bison parser in `CIFParserLexer.l`, `CIFParserParser.y`)
- **KCF** — `KCFFile.h`, `KCFFile.C`
  - KEGG Chemical Function format

**Molecular Dynamics / Trajectory Formats:**
- **DCD** — `DCDFile.h`, `DCDFile.C`
  - CHARMM/NAMD trajectory format
- **TRR** — `TRRFile.h`, `TRRFile.C`
  - GROMACS trajectory format

**Computational Chemistry Output:**
- **GAMESS** — `GAMESSLogFile.h`, `GAMESSDatFile.h` (including parsers in `.l` and `.y` files)
  - Log file and data file parsing for quantum chemistry results
- **MOPAC** — `MOPACInputFile.h`, `MOPACOutputFile.h`
  - Semi-empirical quantum chemistry input/output
- **Antechamber** — `antechamberFile.h`, `antechamberFile.C`
  - AMBER force field preparation

**Protein Structure Prediction:**
- **SCWRL Rotamers** — `SCWRLRotamerFile.h`, `SCWRLRotamerFile.C`
  - Rotamer library format

**Density Maps / Electron Density:**
- **CCP4** — `CCP4File.h`, `CCP4File.C`
  - CCP4 electron density map format
- **DSN6** — `DSN6File.h`, `DSN6File.C`
  - DSN6 density map format
- **Amira Mesh** — `amiraMeshFile.h`, `amiraMeshFile.C`
  - Amira volumetric mesh format

**Spectroscopy:**
- **Bruker NMR** — `bruker1DFile.h`, `bruker1DFile.C`, `bruker2DFile.h`, `bruker2DFile.C`
  - Bruker NMR raw data formats
- **NMRStar** — `NMRStarFile.h`, `NMRStarFile.C`
  - BMRB NMR chemical shift format

**Other Formats:**
- **HMO** — `HMOFile.h`, `HMOFile.C`
  - Hückel Molecular Orbital format
- **INI** — `INIFile.h`, `INIFile.C`
  - Generic INI file parser (used for configuration)
- **Parameter Files** — `paramFile.h`, `parameters.h`
  - CHARMM/AMBER force field parameters
- **Docking Results** — `dockResultFile.h`, `dockResultFile.C`
  - Molecular docking output format
- **Trajectory Factory** — `trajectoryFileFactory.h`, `trajectoryFileFactory.C`
  - Automatic format detection for MD trajectories

**Format Factory & Auto-Detection:**
- `molFileFactory.h`, `trajectoryFileFactory.h` — Auto-detect molecular and trajectory file formats based on extension and content

## APIs & External Services

**OpenBabel Integration:**
- SDK/Client: `OpenBabel2` (optional, off by default)
- Availability: Conditional compile flag `BALL_HAS_OPENBABEL` (set in `CMakeLists.txt:186`)
- Status: Version 2.x API only (3.x requires API porting); currently disabled due to incompatibility
- Use case: Molecular format conversion between OpenBabel-supported formats

**Configuration/Online Resources:**
- **PubChem Downloader** — `pubchemDownloader.h` (optional integration)
  - Allows programmatic download of molecular structures from PubChem

## Data Storage

**In-Memory Data Structures:**
- **Kernel Classes** (in-memory molecular representation):
  - `System`, `Protein`, `Chain`, `Molecule`, `Fragment`, `Atom`, `Bond`, `Residue`, `SecondaryStructure`
  - Implemented in `source/KERNEL/` with headers in `include/BALL/KERNEL/`
- **Property Caching** — Atoms and molecules cache computed properties (partial charges, aromaticity, ring membership)

**Trajectory Data:**
- **Frame Buffer** — Trajectories store coordinate frames in memory; support for periodic boundary conditions (MD box)
- **Molecular Dynamics Ensemble** — GROMACS (TRR) and CHARMM (DCD) frame support

**No Database or SQL Integration:**
- No relational database, ORM, or ODBC support
- All data manipulation is in-memory C++ objects
- File I/O is the only persistence mechanism

## File Storage

**Local Filesystem Only:**
- All I/O is local file-based (no remote S3, Azure Blob, etc.)
- No network file system abstractions
- Data directory (molecular fragments, element data, forcefields) expected at `BALL_DATA_PATH`:
  - `data/fragments/` — Fragment database for molecular building
  - `data/elements/` — Element color/property definitions (including pymol-element-colors patch in root)
  - `data/forcefields/` — AMBER, CHARMM, MMFF94 force field parameters

## Caching

**No caching layer:**
- In-memory computation results are cached on individual `Atom` and `Molecule` objects
- No memcached, Redis, or HTTP caching
- User code responsible for caching if needed

## Authentication & Identity

**Not Applicable:**
- BALL is a computational library, not a service
- No user authentication, OAuth, JWT, or session management
- Optional PubChem integration uses public HTTP (no authentication required)

## Monitoring & Observability

**Error Handling:**
- C++ exceptions (`std::exception` and BALL custom exceptions)
- Assertion macros for debugging (enabled in Debug builds, stripped in Release)
- Some file I/O returns bool/status codes for error handling

**Logging:**
- `std::cout`, `std::cerr` for diagnostic output
- No structured logging framework (no spdlog, Boost.Log, etc.)
- DEBUG_LOG macros in some modules (compile-time conditional)
- Some progress reporting via callbacks in long-running operations

**Performance Profiling:**
- Benchmark suite in `source/BENCHMARKS/` with dedicated CMake config
- Timing instrumentation in molecular dynamics and QSAR modules
- No built-in performance metrics or telemetry (external tools like perf/valgrind expected)

## Parallel Computing

**Threading:**
- **Intel TBB (Threading Building Blocks)** — Optional parallelization; CMake option `USE_TBB=ON` (default)
  - Used in force field calculations, molecular dynamics, QSAR feature generation
  - Requires version ≥ 2017 for LGPL builds; version check in `CMakeLists.txt:244`

**MPI (Message Passing Interface):**
- **MPI** — Optional for distributed computing; CMake option `USE_MPI=OFF` (default)
  - Requires XDR for binary serialization (dependency: `cmake/BALLConfigXDR.cmake`)
  - Documented in code but rarely used in modern workflows

**GPU Acceleration:**
- **CUDA** — Optional, legacy support for versions ≤ 2.1 (not modern CUDA)
  - CMake option `USE_CUDA=OFF` (default)
  - Status: Outdated, not recommended; RT-Fact ray-tracing is CUDA-optional

## CI/CD & Deployment

**Version Control Integration:**
- CMake `PROJECT` version: `1.6.0` (from `CMakeLists.txt:23`)
- No built-in version inference from git

**Build System:**
- **CMake** as build meta-generator (no pre-built binaries in repo)
- No CI/CD configured in current codebase (ROADMAP indicates intent to add CI)
- Manual testing and patch application (see ROADMAP-1.6.md for current status)

**Installation:**
- `CMAKE_INSTALL_PREFIX` determines install location (defaults to `/usr/local` on Unix)
- Standard directories managed by GNUInstallDirs CMake module
- Install targets for libraries, headers, data, translations

**macOS App Bundling:**
- `macdeployqt` tool (found at configure time for macOS) bundles Qt libraries into `.app` bundle
- BALLView produces `BALLView.app/Contents/MacOS/BALLView` executable on macOS

**Windows Package Generation:**
- `windeployqt.exe` tool (found at configure time for Windows) handles Qt deployment
- MSVC incremental linking disabled for BALL library due to size

**Qt Translations:**
- Optional translation generation from `.ts` files via `Qt5LinguistTools` (optional at build time)
- CMake option `UPDATE_TRANSLATIONS=OFF` (default); set to `ON` to regenerate translations

## Packaging & Distribution

**Platform-Specific Approaches:**
- **macOS:** Homebrew formula expected; app bundle via `macdeployqt`
- **Linux:** System package manager (deb, rpm); no bundled Homebrew on Linux
- **Windows:** vcpkg manifest (planned, not yet fully implemented; legacy `ball_contrib` being phased out)

## Environment Configuration

**Build-Time Environment Variables (optional):**
- `PATH` — Must include Bison and Flex executables; on Homebrew macOS:
  ```bash
  export PATH="/opt/homebrew/opt/bison/bin:/opt/homebrew/opt/flex/bin:$PATH"
  ```
- `PYTHONPATH` — Set if using `ball_contrib` Python modules (legacy, deprecated)

**Runtime Environment Variables:**
- `BALL_DATA_PATH` — **Critical:** Path to molecular data directory (fragments, forcefields, element colors)
  - Default (if install prefix is `.`): `{project_root}/data`
  - Default (if install prefix is system): `{CMAKE_INSTALL_PREFIX}/share/BALL/data`
  - BALLView requires this to function (FragmentDB will not load without it)
- `DYLD_LIBRARY_PATH` (macOS) / `LD_LIBRARY_PATH` (Linux) — Path to shared libraries
  - Usually handled by CMake install/RPATH, but needed for in-tree builds: `{build_dir}/lib`
- `BALL_PYTHONPATH` (legacy, deprecated) — Set from `ball_contrib` if present

**CMake Configuration Variables:**
- `CMAKE_PREFIX_PATH` — Directories for `find_package()` to search (e.g., Homebrew Qt, Boost, TBB)
- `CMAKE_BUILD_TYPE` — Release (default), Debug, MinSizeRel, RelWithDebInfo
- `BALL_CONTRIB_PATH` — Path to legacy `ball_contrib` installation (Windows only; obsolete on macOS/Linux)
- `BALL_LICENSE` — LGPL (default) or GPL (affects FFTW, OpenBabel availability)
- `QTDIR` — Optional explicit Qt installation path
- `BISON_EXECUTABLE`, `FLEX_EXECUTABLE` — Explicit paths to parser generators (usually auto-found)

## Webhooks & Callbacks

**Not Applicable:**
- BALL is a computational library, not a service with webhooks or HTTP endpoints
- Internal callback mechanisms exist for progress reporting in long-running operations (molecular dynamics, QSAR training)

## Molecular Data Standards & Registries

**No network integrations:**
- PubChem integration (if used) fetches data via HTTP but no callback mechanism
- No live data bindings or subscriptions to biological databases

---

*Integration audit: 2026-05-14*
