# Architecture

**Analysis Date:** 2026-05-14

## Pattern Overview

**Overall:** Layered modular architecture with hierarchical design patterns

**Key Characteristics:**
- Hierarchical composite pattern for molecular structures (System → Molecule → Residue → Atom → Bond)
- Template-based processor pattern for tree traversal and algorithms
- Separation of concerns: KERNEL (data structures), CONCEPT (interfaces), algorithm modules (ENERGY, STRUCTURE, etc.)
- Graphics rendering abstraction with pluggable renderer backends (GL, POV-Ray, raytracing)
- Qt-based GUI framework for BALLView application

## Layers

**COMMON Layer:**
- Purpose: Core utilities, logging, exceptions, memory management, version info
- Location: `include/BALL/COMMON/`, `source/COMMON/`
- Contains: Exception handling, LogStream, HashSet/HashMap, String utilities, version management
- Depends on: Standard C++ libraries
- Used by: All other BALL modules

**CONCEPT Layer:**
- Purpose: Abstract interfaces and design patterns for the entire library
- Location: `include/BALL/CONCEPT/`, `source/CONCEPT/`
- Contains: Composite (tree pattern), Processor (visitor pattern), Functor, Iterator, Persistent objects, Selectable interface
- Depends on: COMMON
- Used by: KERNEL, all algorithm modules

**KERNEL Layer:**
- Purpose: Core molecular data structures and their hierarchies
- Location: `include/BALL/KERNEL/`, `source/KERNEL/`
- Contains: Atom, Bond, Fragment, Residue, Chain, Molecule, System, Protein, NucleicAcid, Element (periodic table)
- Depends on: CONCEPT, COMMON, MATHS
- Used by: All domain modules (STRUCTURE, ENERGY, FORMAT, DOCKING, etc.)

**DATATYPE Layer:**
- Purpose: General-purpose data structures and algorithms
- Location: `include/BALL/DATATYPE/`, `source/DATATYPE/`
- Contains: HashMap, HashGrid, BitVector, RegularData (1D/2D/3D), List, Triple, String, Options
- Depends on: COMMON
- Used by: KERNEL, algorithm modules

**MATHS Layer:**
- Purpose: Mathematical operations and geometric computations
- Location: `include/BALL/MATHS/`, `source/MATHS/`
- Contains: Vector3, Matrix44, Quaternion, Plane3, Sphere3, FFT (via FFTW), numerical integration
- Depends on: COMMON, optionally FFTW library
- Used by: STRUCTURE, VIEW rendering, geometry calculations

**SYSTEM Layer:**
- Purpose: OS-level abstractions (file I/O, paths, mutexes, threading)
- Location: `include/BALL/SYSTEM/`, `source/SYSTEM/`
- Contains: Directory, File, Path, Mutex, RTTI, plugin interface
- Depends on: COMMON
- Used by: FORMAT loaders, plugin system, GUI threading

**FORMAT Layer:**
- Purpose: Molecular file parsers and writers
- Location: `include/BALL/FORMAT/`, `source/FORMAT/`
- Contains: PDB, MOL, SDF, DCD, HIN, XYZ, GAMESS, Antechamber, DSN6, trajectory formats
- Depends on: KERNEL, COMMON, DATATYPE, optionally OpenBabel
- Used by: Applications, VIEW

**STRUCTURE Layer:**
- Purpose: Structure analysis and manipulation algorithms
- Location: `include/BALL/STRUCTURE/`, `source/STRUCTURE/`
- Contains: FragmentDB, RotamerLibrary, ResidueChecker, SES/SAS surface calculations, geometric transformations, structure mapping
- Depends on: KERNEL, MATHS, DATATYPE
- Used by: DOCKING, SCORING, applications

**ENERGY Layer:**
- Purpose: Molecular energy calculations and forcefield integration
- Location: `include/BALL/ENERGY/`, `source/ENERGY/`
- Contains: EnergyProcessor, Coulomb, van der Waals, atomic contact energy, forcefields (AMBER, MMFF94)
- Depends on: KERNEL, MATHS, CONCEPT
- Used by: MOLMEC, applications

**MOLMEC Layer:**
- Purpose: Molecular mechanics and dynamics
- Location: `include/BALL/MOLMEC/`, `source/MOLMEC/`
- Contains: Forcefields (AMBER, MMFF94), molecular dynamics integrators, constraint handling
- Depends on: ENERGY, KERNEL, MATHS
- Used by: Applications, VIEW

**NMR Layer:**
- Purpose: NMR spectroscopy calculations
- Location: `include/BALL/NMR/`, `source/NMR/`
- Contains: Shift calculations, SHIFTX integration, anisotropy tensor
- Depends on: KERNEL, ENERGY, MATHS
- Used by: Specialized applications

**SOLVATION Layer:**
- Purpose: Implicit solvation models
- Location: `include/BALL/SOLVATION/`, `source/SOLVATION/`
- Contains: Poisson-Boltzmann, GBSA, solvation energy calculations
- Depends on: ENERGY, KERNEL, optionally FFTW
- Used by: MOLMEC, energy calculations

**QSAR Layer:**
- Purpose: Quantitative Structure-Activity Relationship tools
- Location: `include/BALL/QSAR/`, `source/QSAR/`
- Contains: Molecular descriptors, kernel methods, models, SVM integration
- Depends on: KERNEL, DATATYPE, optionally libSVM
- Used by: Drug discovery applications

**DOCKING Layer:**
- Purpose: Molecular docking algorithms
- Location: `include/BALL/DOCKING/`, `source/DOCKING/`
- Contains: GeometricFit, genetic algorithms (GENETICDOCK), image-based methods (IMGDOCK)
- Depends on: KERNEL, ENERGY, STRUCTURE, MATHS
- Used by: Docking applications

**SCORING Layer:**
- Purpose: Scoring functions for ranking binding poses
- Location: `include/BALL/SCORING/`, `source/SCORING/`
- Contains: Various scoring function implementations
- Depends on: ENERGY, KERNEL
- Used by: DOCKING, applications

**XRAY Layer:**
- Purpose: X-ray crystallography support
- Location: `include/BALL/XRAY/`, `source/XRAY/`
- Contains: Crystallographic operations and analysis
- Depends on: KERNEL, MATHS
- Used by: Structural biology applications

**VIEW Layer:**
- Purpose: Graphics rendering and visualization GUI
- Location: `include/BALL/VIEW/`, `source/VIEW/`
- Contains: OpenGL rendering pipeline, visualization models (ball & stick, cartoon, surface), GUI widgets, camera, lighting
- Depends on: KERNEL, MATHS, DATATYPE, Qt5, OpenGL
- Used by: BALLView application

**PYTHON Layer:**
- Purpose: Python bindings and scripting interface
- Location: `include/BALL/PYTHON/`, `source/PYTHON/`
- Contains: SIP bindings (currently disabled), Python extension modules
- Depends on: All BALL modules
- Used by: Python scripting, Jupyter integration

## Data Flow

**Molecular Structure Loading Pipeline:**

1. User loads PDB/MOL file via `MainFrame::onOpenStructure()`
2. FORMAT layer deserializes file (e.g., `PDBFile::read()`) → creates KERNEL objects
3. KERNEL builds hierarchical structure: System contains Molecules, Molecules contain Residues, Residues contain Atoms
4. STRUCTURE module optionally processes (FragmentDB lookup, secondary structure detection)
5. Results stored in memory as Composite tree

**Visualization Pipeline:**

1. KERNEL Atoms/Bonds passed to VIEW::ModelProcessor tree
2. Model processors (BallAndStickModel, CartoonModel, SurfaceModel) traverse molecular tree
3. Processors return Primitives (spheres, cylinders, meshes) via VIEW::Geometry
4. RenderSetup batches geometries into VertexBuffers
5. RendererFactory selects active renderer (GLRenderer for interactive, POVRenderer for output)
6. GLRenderer draws to GLRenderWindow (Qt5 QOpenGLWidget); camera/lighting applied via RenderSetup
7. Frame displayed via Qt event loop

**Energy Calculation Pipeline:**

1. EnergyProcessor (or ComposedEnergyProcessor) instantiated with parameters
2. Traverses KERNEL atoms via Composite::apply(processor)
3. Each energy component (Coulomb, van der Waals, etc.) calculates pair-wise or self-energies
4. Energies accumulated and returned; optional gradient calculated for optimization

**State Management:**
- Molecular structures are mutable in-memory representations (Atom positions, velocities, forces)
- Processors apply transformations destructively (e.g., updating coordinates)
- Selection state maintained via Composite::setSelected(), propagated up/down tree
- Modification timestamp tracked (CONCEPT::TimeStamp) for caching invalidation

## Key Abstractions

**Composite:**
- Purpose: Base class for all structural hierarchy (System, Molecule, Residue, Chain, Atom)
- Examples: `include/BALL/CONCEPT/composite.h`, `include/BALL/KERNEL/atom.h`, `include/BALL/KERNEL/molecule.h`
- Pattern: Template method via virtual apply() for tree traversal with predicates
- Exports: Bidirectional tree navigation, depth-first traversal, selection management

**UnaryProcessor<T>:**
- Purpose: Visitor pattern for applying algorithms to structures
- Examples: `include/BALL/CONCEPT/processor.h`
- Pattern: Functor-based; returns Processor::CONTINUE/BREAK/ABORT
- Exports: start(), operator()(T&), finish() lifecycle hooks
- Used for: structure traversal (STRUCTURE processors), visualization (ModelProcessors), energy (EnergyProcessor)

**ConstUnaryProcessor<T>:**
- Purpose: Read-only visitor for const-correctness
- Examples: `include/BALL/CONCEPT/processor.h`
- Pattern: Same as UnaryProcessor but takes const T&
- Used for: analysis, geometric queries

**Renderer Abstract Base:**
- Purpose: Abstraction over rendering backends
- Examples: `include/BALL/VIEW/RENDERING/RENDERERS/glRenderer.h`, POVRenderer, raytracingRenderer
- Subclasses: GLRenderer (OpenGL), POVRenderer (ray-tracing via POV-Ray), XML3DRenderer, STLRenderer
- Pattern: Factory pattern via RendererFactory; plugin-loadable
- Exports: render(Geometry, RenderSetup), targeting RenderTarget or framebuffer

**RenderWindow:**
- Purpose: Abstract graphics output surface
- Examples: `include/BALL/VIEW/RENDERING/glRenderWindow.h`, GLOffscreenTarget
- Pattern: Template method with init(), resize(), refresh() lifecycle; implements RenderSurface
- Exports: OpenGL context lifecycle (beginFrame/endFrame), text rendering, event handling

## Entry Points

**BALLView Application:**
- Location: `source/APPLICATIONS/BALLVIEW/main.C`
- Triggers: User launches BALLView.app / ballview executable
- Responsibilities: Initialize Qt application, create MainFrame, load FragmentDB, install signal handlers, start event loop, handle command-line arguments

**MainFrame:**
- Location: `source/VIEW/WIDGETS/mainframe.C/H` (inferred from import in main.C)
- Triggers: QApplication initialization
- Responsibilities: Create menu bar, toolbars, status bar, initialize Scene widget, manage file dialogs, coordinate application actions

**Scene Widget:**
- Location: `include/BALL/VIEW/WIDGETS/scene.h`
- Triggers: MainFrame::loadStructure()
- Responsibilities: Manage molecular system in memory, delegate rendering to RenderSetup/GLRenderWindow, handle mouse/keyboard for rotation/zoom

**Python Extensions:**
- Location: `source/PYTHON/EXTENSIONS/` (if enabled)
- Triggers: Python import ball
- Responsibilities: Export BALL classes to Python via SIP bindings (currently disabled in v1.6)

## Error Handling

**Strategy:** Exception-based with fallbacks for critical paths

**Patterns:**
- BALL::Exception hierarchy (`include/BALL/COMMON/exception.h`)
- LogStream for warnings/errors (BALL::Log.error(), Log.warn())
- Processor ABORT signal for terminating tree traversal on error
- File I/O: exceptions on parse failure, logged to LogStream; format-specific handlers catch and rethrow

## Cross-Cutting Concerns

**Logging:** 
- Central: `BALL::Log` instance via `include/BALL/COMMON/logStream.h`
- Usage: Log.info() << "message" << std::endl; level-based filtering
- Qt integration: qInstallMessageHandler() in main.C redirects Qt messages to BALL::Log

**Validation:** 
- STRUCTURE::ResidueChecker validates molecule topology
- FORMAT loaders validate file syntax and element symbols via PTE
- KERNEL constructors validate atom/bond constraints

**Authentication:** 
- Not applicable (library, not networked service)

**Threading:** 
- VIEW rendering: GLRenderWindow uses QOpenGLWidget (Qt's thread-safe wrapper)
- TBB integration for parallel algorithms (optional, compile-time toggle USE_TBB)
- MOLMEC: Careful to avoid race conditions in force calculations (mutexes in critical sections)
- Plugin system: Thread-safe plugin loading via SYSTEM::DynamicLibrary

---

*Architecture analysis: 2026-05-14*
