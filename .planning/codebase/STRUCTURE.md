# Codebase Structure

**Analysis Date:** 2026-05-14

## Directory Layout

```
./
├── include/BALL/           # Public API headers (C++ templates and interfaces)
│   ├── COMMON/            # Logging, exceptions, macros, version
│   ├── CONCEPT/           # Abstract base classes (Composite, Processor, Iterator, etc.)
│   ├── CONFIG/            # Build-time configuration
│   ├── DATATYPE/          # Generic data structures (HashMap, BitVector, String, etc.)
│   ├── KERNEL/            # Molecular structures (Atom, Bond, Molecule, System, etc.)
│   ├── MATHS/             # Linear algebra and geometry (Vector3, Matrix44, FFT, etc.)
│   ├── SYSTEM/            # OS abstractions (File, Path, Mutex, plugins)
│   ├── FORMAT/            # File format parsers (PDB, MOL, DCD, etc.)
│   ├── STRUCTURE/         # Structure analysis (FragmentDB, RotamerLibrary, SES, etc.)
│   ├── ENERGY/            # Energy calculations (Coulomb, vdW, forcefields)
│   ├── MOLMEC/            # Molecular mechanics (dynamics, integrators)
│   ├── NMR/               # NMR calculations (shifts, tensors)
│   ├── SOLVATION/         # Solvation models (PB, GBSA)
│   ├── QSAR/              # Molecular descriptors and ML
│   ├── DOCKING/           # Docking algorithms
│   ├── SCORING/           # Scoring functions
│   ├── XRAY/              # Crystallography
│   ├── VIEW/              # Visualization and GUI
│   ├── PLUGIN/            # Plugin interface definitions
│   └── PYTHON/            # Python binding declarations
├── source/                # Implementation (.C) files, organized by module
│   ├── COMMON/
│   ├── CONCEPT/
│   ├── DATATYPE/
│   ├── KERNEL/
│   ├── MATHS/
│   ├── SYSTEM/
│   ├── FORMAT/
│   ├── STRUCTURE/
│   ├── ENERGY/
│   ├── MOLMEC/
│   ├── NMR/
│   ├── SOLVATION/
│   ├── QSAR/
│   ├── DOCKING/
│   ├── SCORING/
│   ├── XRAY/
│   ├── VIEW/              # GUI, rendering, models, widgets
│   │   ├── DATATYPE/      # VIEW-specific types (ColorRGBA, etc.)
│   │   ├── KERNEL/
│   │   ├── MODELS/        # Visualization processors (BallAndStick, Cartoon, etc.)
│   │   ├── RENDERING/     # OpenGL, raytracers, framebuffer
│   │   │   └── RENDERERS/ # Pluggable renderers (GL, POV, etc.)
│   │   ├── WIDGETS/       # Qt widgets (Scene, MainFrame)
│   │   ├── DIALOGS/       # Dialog windows
│   │   ├── INPUT/         # Input device handling
│   │   ├── PRIMITIVES/    # Geometric primitives
│   │   └── PLUGIN/        # Plugin handler interfaces
│   ├── PYTHON/
│   │   └── EXTENSIONS/    # Python binding modules
│   ├── APPLICATIONS/      # Standalone tools and GUIs
│   │   ├── BALLVIEW/      # Main GUI application
│   │   ├── AMBER/         # AMBER force field tools
│   │   ├── MMFF94/        # MMFF94 force field tools
│   │   ├── NMRVIEW/       # NMR visualization
│   │   ├── DCD2PNG/       # Trajectory renderer
│   │   ├── PB/            # Poisson-Boltzmann solver
│   │   ├── TOOLS/         # Command-line utilities
│   │   └── UTILITIES/     # Installation and demo utilities
│   ├── EXTENSIONS/        # Optional plugins
│   │   ├── PRESENTABALL/  # Presentation tools
│   │   ├── SPACENAV/      # SpaceNavigator input
│   │   ├── VRPN/          # VRPN tracking
│   │   ├── VRPNHD/        # VRPN HD
│   │   ├── BALLAXY/       # Galaxy workflow
│   │   └── JUPYTER/       # Jupyter integration
│   ├── BENCHMARKS/        # Performance benchmarks
│   └── PLUGIN/            # Plugin source interface
├── test/                  # Test suites
│   ├── python/            # Python tests
│   ├── cmake/             # CMake test configuration
│   └── data/              # Test molecule structures
├── cmake/                 # CMake build configuration
│   ├── BALLMacros.cmake
│   ├── BALLConfiguration.cmake
│   ├── BALLCompilerSpecific.cmake
│   ├── BALLConfigBoost.cmake
│   ├── BALLConfigFFTW.cmake
│   ├── FindXXX.cmake      # Find modules for dependencies
│   └── templates/         # Config file templates
├── data/                  # Runtime data (fragments, parameters)
│   └── BALLView/          # BALLView data (icons, translations)
├── doc/                   # Documentation
├── build/                 # CMake build output (generated)
├── build_03/              # Alternative build (generated)
├── workflow_support/      # Galaxy/CWL workflow definitions
│   ├── galaxy/
│   ├── cwl/
│   └── common/
├── .github/               # GitHub CI/CD workflows
├── .planning/             # Project planning artifacts
├── CMakeLists.txt         # Root CMake configuration
└── BALL version 1.6.0-dev
```

## Directory Purposes

**include/BALL/**
- Purpose: C++ public API; all headers exposed to library users
- Contains: Class declarations, template implementations (.iC files), interface contracts
- Key files: `common.h`, `kernel.h`, `structure.h`, `view.h` (aggregate headers)
- Organized: By module name (KERNEL/, ENERGY/, FORMAT/, etc.), matching source/ structure

**source/KERNEL/**
- Purpose: Core molecular data structure implementations
- Contains: Atom.C, Bond.C, Molecule.C, System.C, Residue.C, Chain.C, Protein.C, NucleicAcid.C
- Size: ~50 .C files, large class implementations
- Compile time: Significant (many template instantiations)

**source/VIEW/RENDERING/**
- Purpose: Graphics pipeline implementation
- Contains: GLRenderWindow.C (Qt5 OpenGL widget), RendererFactory, RenderSetup, VertexBuffer, Camera, Lighting
- Key subdir: RENDERERS/ with GLRenderer.C (main), POVRenderer.C (ray-tracing export), raytracingRenderer.C (CPU raytracer)
- Dependencies: Qt5::OpenGL, OpenGL headers

**source/VIEW/MODELS/**
- Purpose: Visualization model processors
- Contains: Processors that traverse Atoms/Bonds and generate Primitives (spheres, cylinders, meshes)
- Examples: BallAndStickModel, CartoonModel, SurfaceModel, BackboneModel, LabelModel
- Pattern: Each inherits from ModelProcessor, uses UnaryProcessor<Atom> or <Residue>

**source/VIEW/WIDGETS/**
- Purpose: GUI widgets and dialogs
- Contains: Scene (main visualization widget), MainFrame (application window), various dialogs
- Qt integration: Custom Qt widgets, .ui files, qrc resource files

**source/FORMAT/**
- Purpose: Molecular file I/O
- Contains: PDBFile, MOLFile, SDFile, DCD, HIN, XYZ, GAMESS, etc.
- Pattern: Each format subclasses GenericMolFile or LineBasedFile
- Responsibility: Parse/write molecular structures to KERNEL objects

**source/STRUCTURE/**
- Purpose: Structure analysis algorithms
- Contains: FragmentDB (residue template library), RotamerLibrary, surface generators (SES, SAS), structure mappers
- Size: Large; FragmentDB.C is substantial

**source/ENERGY/**
- Purpose: Energy calculations
- Contains: EnergyProcessor, Coulomb, vdW, other potential energy terms
- Dependencies: MOLMEC for forcefield parameters

**source/APPLICATIONS/BALLVIEW/**
- Purpose: Main BALLView GUI application
- Entry point: main.C, calls MainFrame constructor
- UI files: aboutDialog.ui, demoTutorialDialog.ui
- Resources: Icons, welcome screen, translations

**test/**
- Purpose: Test suites
- Organization: Mirrors source/ structure
- Test data: /test/data/ contains sample molecules (PDB files)
- Runners: pytest (Python tests), CMake custom targets for C++ tests

**data/BALLView/**
- Purpose: Runtime application data
- Contains: FragmentDB files (residues.db), parameters, icon files, translations (.ts/.qm)
- Loaded at: Application startup by MainFrame/Scene

**cmake/**
- Purpose: Build system configuration
- Key files: 
  - `BALLConfiguration.cmake`: Feature detection (FFTW, OpenBabel, TBB, Qt5)
  - `BALLMacros.cmake`: BALL_ADD_LIBRARY, BALL_ADD_TEST macros
  - `BALLCompilerSpecific.cmake`: Compiler flags (C++17, warnings)
  - `FindXXX.cmake`: Dependency locators

**build/** and **build_03/**
- Purpose: CMake build artifacts (generated; not committed)
- Location: Build output, object files, binaries, compiled tests
- Generated files: Flex/Bison parsers (expressionParser.C, etc.), config headers

## Key File Locations

**Entry Points:**
- `source/APPLICATIONS/BALLVIEW/main.C`: BALLView application startup
- `source/APPLICATIONS/BALLVIEW/mainframe.C`: GUI window initialization
- `CMakeLists.txt`: Build root, dependencies, project setup

**Configuration:**
- `CMakeLists.txt`: Project version (1.6.0), C++ standard (17), dependency finds
- `cmake/BALLConfiguration.cmake`: Feature flags (BALL_HAS_OPENBABEL, BALL_HAS_TBB, etc.)
- `include/BALL/CONFIG/config.h.in`: Configured at build time with platform defines

**Core Logic:**
- `include/BALL/KERNEL/molecule.h`: Molecule class (inherits AtomContainer → Composite)
- `include/BALL/KERNEL/atom.h`: Atom class (stores element, position, velocity, bonds, properties)
- `include/BALL/CONCEPT/composite.h`: Composite tree pattern (selection, modification tracking, tree traversal)
- `include/BALL/CONCEPT/processor.h`: Visitor pattern for structure algorithms
- `include/BALL/VIEW/RENDERING/glRenderWindow.h`: Qt5 OpenGL widget, main rendering surface
- `source/VIEW/RENDERING/RENDERERS/glRenderer.C`: OpenGL draw calls, shader program activation

**Testing:**
- `test/python/`: Python test modules (imported by pytest)
- `test/data/`: Sample molecule files (PDB, DCD, etc.)
- `CMakeLists.txt` in test/: CMake targets for C++ unit tests

**Data Files:**
- `data/BALLView/`: FragmentDB, parameters, icons, translations
- `data/BALLView/translations/*.ts`: Qt Linguist translation files (auto-generated or edited)

## Naming Conventions

**Files:**
- C++ source: `.C` (implementation, unusual convention; some projects use `.cpp`)
- C++ header: `.h` (declarations)
- Inline implementations: `.iC` (template method bodies, included by .h)
- Qt: `.ui` (XML UI definitions), `.qrc` (resource files), `.ts` (translations)
- CMake: `CMakeLists.txt` (build config per directory), `*.cmake` (included modules)

**Directories:**
- UPPERCASE: Module names (KERNEL, ENERGY, VIEW, RENDERING)
- lowercase: Subdirectory categories (source/VIEW/MODELS/, source/APPLICATIONS/BALLVIEW/)
- Plural: For collections (RENDERERS/, EXTENSIONS/, APPLICATIONS/)

**Classes:**
- CamelCase: Atom, Molecule, System, GLRenderWindow, BallAndStickModel
- Suffixes: `Iterator` (atom iterator), `Processor` (processor), `Model` (visualization), `File` (I/O)
- Acronyms: PTE (periodic table), SES (solvent-excluded surface), SAS (solvent-accessible), RMSD (root mean squared deviation)

**Functions/Methods:**
- camelCase: getAtom(), setPosition(), apply(), start(), finish()
- Prefixes: `get` (accessor), `set` (mutator), `has` (predicate), `is` (state query)

**Variables:**
- camelCase: atomCount, moleculeList, energyValue
- Member: Suffix `_` (atom_, molecule_) in some files (not universal)

**Macros:**
- UPPERCASE_WITH_UNDERSCORES: BALL_EXPORT, BALL_CREATE_DEEP, BALL_HAS_OPENBABEL, MAX_NUMBER_OF_BONDS

## Where to Add New Code

**New Feature (e.g., new scoring function):**
- Primary code: `include/BALL/SCORING/myScoreFunc.h` + `source/SCORING/myScoreFunc.C`
- Tests: `test/python/testMyScore.py` or create C++ test target in `test/CMakeLists.txt`
- Integration: Add to CMakeLists.txt in SCORING/ to compile and link

**New Component/Module (e.g., new format parser):**
- Header: `include/BALL/FORMAT/myFormat.h`
- Implementation: `source/FORMAT/myFormat.C`
- Base class: Inherit from GenericMolFile or LineBasedFile
- Register: Add to `source/FORMAT/molFileFactory.C` for automatic detection by file extension

**New Visualization Model (e.g., new rendering style):**
- Header: `include/BALL/VIEW/MODELS/myVisModel.h`
- Implementation: `source/VIEW/MODELS/myVisModel.C`
- Base class: Inherit from ModelProcessor
- Pattern: Implement `Processor::Result operator()(Atom& a)` to return Primitives (sphere, cylinder, mesh)
- Integration: Register in VIEW::ModelFactory or MainFrame model list

**Utilities (shared helpers):**
- Common: `include/BALL/STRUCTURE/defaultProcessors.h` (general-purpose structure algorithms)
- Math: `include/BALL/MATHS/` for geometric operations (already rich)
- Data: `include/BALL/DATATYPE/` for container/algorithm templates

**Standalone Application:**
- Create: `source/APPLICATIONS/MYAPP/main.C` with CMakeLists.txt
- Link against: BALL library and required dependencies (Qt5, OpenGL if GUI)
- Pattern: Reference BALLVIEW or TOOLS for boilerplate

## Special Directories

**build/ and build_03/:**
- Purpose: CMake build artifacts
- Generated: Object files, executables, test results
- Committed: No (should be in .gitignore)
- Regeneration: `mkdir build && cd build && cmake .. && make`

**cmake/templates/**
- Purpose: Template files for code generation
- Examples: BALLViewInstallFix.cmake.in (substituted with CMake variables at configure time)
- Committed: Yes

**data/BALLView/**
- Purpose: Application runtime data (not C++ code)
- Contents: FragmentDB binary/text, molecular parameter files, UI resources, translations
- Committed: Yes (data is essential for functionality)
- Loaded at: Application startup via SYSTEM::Path or QResources

**workflow_support/galaxy/ and workflow_support/cwl/**
- Purpose: Export BALL tools as Galaxy/CWL workflows for bioinformatics platforms
- Not core to library functionality
- Optional, for integration with external systems

**ball_contrib/** (parent ../ball_contrib)
- Purpose: Bundled dependencies (Boost, Qt, OpenBabel, etc.) — **OBSOLETE**
- Status: Dead; build against system/Homebrew packages instead
- Do not use: CMakeLists.txt explicitly disables it on macOS/Linux

---

*Structure analysis: 2026-05-14*
