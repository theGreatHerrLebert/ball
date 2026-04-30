// Python bindings for BALL — analysis surface that mirrors the things
// a downstream consumer (e.g. proteon's EVIDENT oracle suite) wants
// to compare BALL against on a per-PDB basis.
//
// Surface today (v0.1.0a4):
//   amber_energy            single-point AMBER force-field energy
//   charmm_energy           single-point CHARMM force-field energy (EEF1 toggle)
//   mmff94_energy           single-point MMFF94 force-field energy (small-molecule FF)
//   hbonds                  hydrogen bond list (HBondProcessor)
//   secondary_structure     per-residue SS assignment (SecondaryStructureProcessor)
//   add_hydrogens_to_pdb    write a hydrogenated PDB (AddHydrogenProcessor)
//   sasa                    solvent-accessible surface area (NumericalSAS)
//   build_bonds             bond inference (BuildBondsProcessor)
//   atom_typer              per-atom AMBER type + charge enumeration
//   minimize_energy         conjugate-gradient or steepest-descent
//   system_info             atom/residue/chain counts and chain ids
//   rmsd                    Kabsch-aligned RMSD between two PDBs
//
// Deferred to v0.3 (each needs more setup than the surface budget):
//   gb_energy               GeneralizedBornModel needs scaling-factor INI
//   peptide_from_sequence   generative fixture builder; structure
//                           construction has its own validation surface
//
// Each binding is a "process and report" function: takes a path,
// returns a structured Python dict/list, raises RuntimeError with the
// underlying BALL message on failure. State-bearing class bindings
// (PySystem, PyAtom, etc.) are deferred — most oracle comparisons
// don't need them.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include <BALL/COMMON/exception.h>
#include <BALL/FORMAT/PDBFile.h>
#include <BALL/KERNEL/system.h>
#include <BALL/KERNEL/atom.h>
#include <BALL/KERNEL/atomIterator.h>
#include <BALL/KERNEL/residue.h>
#include <BALL/KERNEL/residueIterator.h>
#include <BALL/KERNEL/chain.h>
#include <BALL/KERNEL/chainIterator.h>
#include <BALL/KERNEL/secondaryStructure.h>
#include <BALL/KERNEL/PTE.h>
#include <BALL/MOLMEC/AMBER/amber.h>
#include <BALL/MOLMEC/CHARMM/charmm.h>
#include <BALL/MOLMEC/MMFF94/MMFF94.h>
#include <BALL/STRUCTURE/fragmentDB.h>
#include <BALL/STRUCTURE/addHydrogenProcessor.h>
#include <BALL/MOLMEC/COMMON/forceField.h>
#include <BALL/MOLMEC/MINIMIZATION/conjugateGradient.h>
#include <BALL/MOLMEC/MINIMIZATION/steepestDescent.h>
#include <BALL/STRUCTURE/HBondProcessor.h>
#include <BALL/STRUCTURE/buildBondsProcessor.h>
#include <BALL/STRUCTURE/defaultProcessors.h>
#include <BALL/STRUCTURE/numericalSAS.h>
#include <BALL/STRUCTURE/secondaryStructureProcessor.h>
#include <BALL/STRUCTURE/RMSDMinimizer.h>
#include <BALL/STRUCTURE/atomBijection.h>

namespace py = pybind11;
using namespace BALL;

namespace {

// Locate the wheel-bundled BALL data tree and set BALL_DATA_PATH if
// the user has not already set it.
//
// Why this is necessary: scikit-build-core installs BALL's data
// directory (Fragments.db, parameter INI files, atom-typing
// templates, radii sets, ...) under share/BALL/ in the wheel. At
// runtime libBALL.so resolves data files via Path::find(), which
// consults BALL_DATA_PATH first. cibuildwheel-built wheels fix the
// data path to the build container's layout, which does not exist
// on the consumer machine; without overriding BALL_DATA_PATH every
// data-dependent call (AmberFF::setup, FragmentDB construction,
// CharmmFF, RMSDMinimizer, ...) fails with the opaque
// "BALL: std::exception" because libBALL never finds Fragments.db.
//
// Strategy: at module-init time, dladdr the address of an in-module
// symbol to recover this .so's filesystem path; the wheel layout
// puts ball.cpython-*.so at <site-packages>/ and share/BALL/ as a
// sibling at <site-packages>/share/BALL/, so the data dir is one
// directory up from the .so. Probe for the canonical sentinel file
// (fragments/Fragments.db) to confirm the layout before touching
// the env. If the user has pre-set BALL_DATA_PATH, leave it alone.
//
// The helper is a free function rather than a side-effect at static
// init so its execution order is deterministic — it runs as the
// first statement of PYBIND11_MODULE, before any m.def() body
// could plausibly trigger a BALL data lookup. Using a static
// initializer would risk firing before pybind11's own init,
// depending on link order.
void set_ball_data_path_if_unset() {
    if (std::getenv("BALL_DATA_PATH") != nullptr) {
        return;
    }
    // Take the address of THIS function — it is guaranteed to live
    // in the ball.cpython-*.so we want to locate. dladdr fills
    // dli_fname with the .so's filesystem path.
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&set_ball_data_path_if_unset), &info) == 0
        || info.dli_fname == nullptr) {
        return;
    }
    auto so_path = std::filesystem::path(info.dli_fname);
    auto candidate = so_path.parent_path() / "share" / "BALL";
    auto sentinel = candidate / "fragments" / "Fragments.db";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(sentinel, ec)) {
        // Wheel layout not as expected. Could be a development
        // install where share/BALL/ lives elsewhere; user must set
        // BALL_DATA_PATH explicitly in that case.
        return;
    }
    // setenv writes into the process environment; the third arg=0
    // means do not overwrite, but we already returned early above
    // when the env was set, so 1 (overwrite) is equivalent and
    // matches the function's intent.
    setenv("BALL_DATA_PATH", candidate.c_str(), 1);
}

// pybind11's default exception translator turns bare C++ exceptions
// into Python RuntimeError("std::exception") — losing the message.
// Wrap the public entry points to translate BALL's typed exceptions
// (Exception::GeneralException and friends, all inheriting from
// std::exception) into runtime_error with the original what() text.
template <typename Fn>
auto translate_ball_exceptions(Fn&& fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const Exception::GeneralException& e) {
        // GeneralException stores the message body via what(); BALL's
        // own location/file info is appended by the throw site.
        throw std::runtime_error(std::string("BALL: ") + e.what());
    } catch (const std::exception& e) {
        // Non-BALL std exceptions (filesystem errors, etc.).
        throw std::runtime_error(std::string("BALL bindings: ") + e.what());
    }
}

// Read a PDB file into a freshly-allocated System and run the
// standard preprocessing chain that BALL's force-field setup expects:
// fragment-DB name normalization, bond building, and (optionally)
// hydrogen placement.
//
// Why this matters: BALL's force-field components (CharmmBend,
// AmberTorsion, etc.) silently zero out individual terms when atom
// triplets/quadruplets cannot be matched in the parameter table. On a
// raw heavy-atom-only PDB (e.g. crambin from the wwPDB) most bend and
// torsion terms fail lookup because the relevant atom types only
// exist in their hydrogenated form. The result is a force-field
// energy that LOOKS plausible (stretch, vdW, electrostatic populate)
// but reports angle_bend = 0 and torsion = 0. Callers want to know
// when that happens — hence the optional preprocessing flags below
// rather than silently mutating the system.
//
// PDBFile::open returns void (overriding File::open's bool return),
// so success has to be probed via isOpen() rather than a return
// value. read() does return bool.
std::unique_ptr<System> load_pdb(const std::string& path,
                                 bool normalize_names,
                                 bool build_bonds,
                                 bool add_hydrogens) {
    // Pre-check existence so a missing path produces a friendly
    // message before BALL throws its own FileNotFound (which the
    // translator wrapper would also catch, but the explicit check
    // gives a more obvious diagnostic for the common typo case).
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("BALL: cannot open PDB file (no such file): " + path);
    }

    auto sys = std::make_unique<System>();
    PDBFile f;
    f.open(path);
    if (!f.isOpen()) {
        throw std::runtime_error("BALL: cannot open PDB file: " + path);
    }
    if (!f.read(*sys)) {
        f.close();
        throw std::runtime_error("BALL: failed to parse PDB file: " + path);
    }
    f.close();
    if (sys->countAtoms() == 0) {
        throw std::runtime_error("BALL: PDB loaded zero atoms: " + path);
    }

    if (normalize_names || build_bonds) {
        // Default-constructed FragmentDB loads "fragments/Fragments.db"
        // from BALL_DATA_PATH; that file ships in BALL's data tree and
        // covers the standard amino acids. Constructing once here is
        // fine — the DB is read-only and small.
        FragmentDB frag_db("");
        if (normalize_names) {
            sys->apply(frag_db.normalize_names);
        }
        if (build_bonds) {
            sys->apply(frag_db.build_bonds);
        }
    }

    if (add_hydrogens) {
        // BALL places ALL hydrogens (polar + non-polar) here. For
        // CHARMM19 (a polar-H force field) the non-polar Hs are not
        // strictly needed but their presence is harmless because
        // CHARMM19's atom-type table absorbs them into united-carbon
        // types. AMBER96 (all-atom) wants them.
        //
        // Caller-side note: H placement is a documented divergence
        // axis in oracle comparisons — proteon's add_hydrogens and
        // BALL's AddHydrogenProcessor will not place identical Hs.
        // For a clean energy-component oracle, pre-place hydrogens
        // upstream and pass add_hydrogens=False here.
        AddHydrogenProcessor hp;
        sys->apply(hp);
    }

    return sys;
}

// Apply user-supplied non-bonded cutoff to the force-field options
// before setup(). BALL's AmberFF and CharmmFF share the same option
// names for these knobs, so the helper is force-field-agnostic.
template <typename FF>
void apply_cutoff(FF& ff, double nonbonded_cutoff) {
    ff.options[FF::Option::NONBONDED_CUTOFF] = std::to_string(nonbonded_cutoff);
    ff.options[FF::Option::VDW_CUTOFF]       = std::to_string(nonbonded_cutoff);
    ff.options[FF::Option::ELECTROSTATIC_CUTOFF] = std::to_string(nonbonded_cutoff);
}

// ---------------------------------------------------------------------------
// Helpers — atom identity for HBond / SS / SASA outputs
// ---------------------------------------------------------------------------

// Render a stable atom identifier so a downstream consumer can join
// against the same atom in another tool's output. Format:
//   "<chain>:<residue_name><residue_id>:<atom_name>"
// Matches the convention proteon uses for cross-tool joins.
std::string atom_key(const Atom& atom) {
    std::ostringstream out;
    const auto* residue = atom.getResidue();
    if (residue != nullptr) {
        const auto* chain = residue->getChain();
        if (chain != nullptr) {
            out << chain->getName() << ":";
        }
        out << residue->getName() << residue->getID() << ":";
    }
    out << atom.getName();
    return out.str();
}

std::string residue_key(const Residue& residue) {
    std::ostringstream out;
    const auto* chain = residue.getChain();
    if (chain != nullptr) {
        out << chain->getName() << ":";
    }
    out << residue.getName() << residue.getID();
    return out.str();
}

// ---------------------------------------------------------------------------
// hbonds — HBondProcessor wrapper
// ---------------------------------------------------------------------------

py::list hbonds(const std::string& pdb_path,
                bool normalize_names,
                bool build_bonds,
                bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);
        HBondProcessor hbp;
        sys->apply(hbp);
        auto bonds = hbp.getHBonds();

        py::list out;
        for (auto& h : bonds) {
            const Atom* donor = h.getDonor();
            const Atom* acceptor = h.getAcceptor();
            if (donor == nullptr || acceptor == nullptr) continue;
            py::dict entry;
            entry["donor"] = atom_key(*donor);
            entry["acceptor"] = atom_key(*acceptor);
            entry["length"] = h.getLength();
            out.append(entry);
        }
        return out;
    });
}

// ---------------------------------------------------------------------------
// secondary_structure — SecondaryStructureProcessor wrapper
// ---------------------------------------------------------------------------

// Translate BALL's SecondaryStructure::Type enum to a one-letter code
// matching the DSSP convention. BALL's vocabulary is coarser
// (HELIX / STRAND / TURN / COIL / UNKNOWN) than DSSP's 8-class
// alphabet; map to the 3-class reduction (H / E / C / T).
const char* ss_type_letter(SecondaryStructure::Type t) {
    switch (t) {
        case SecondaryStructure::HELIX:   return "H";
        case SecondaryStructure::STRAND:  return "E";
        case SecondaryStructure::TURN:    return "T";
        case SecondaryStructure::COIL:    return "C";
        default:                          return "-";
    }
}

py::dict secondary_structure(const std::string& pdb_path,
                             bool normalize_names,
                             bool build_bonds,
                             bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);
        SecondaryStructureProcessor proc;
        sys->apply(proc);

        // Walk residues in order; emit (residue_key, ss_letter) pairs.
        py::list assignments;
        for (auto rit = sys->beginResidue(); +rit; ++rit) {
            const Residue& r = *rit;
            const SecondaryStructure* ss = r.getSecondaryStructure();
            const char* letter = ss != nullptr
                ? ss_type_letter(ss->getType())
                : "-";
            py::dict entry;
            entry["residue"] = residue_key(r);
            entry["ss"] = std::string(letter);
            assignments.append(entry);
        }

        py::dict d;
        d["assignments"] = assignments;
        d["n_residues"] = py::len(assignments);
        return d;
    });
}

// ---------------------------------------------------------------------------
// add_hydrogens — write a hydrogenated PDB
// ---------------------------------------------------------------------------

py::dict add_hydrogens_to_pdb(const std::string& pdb_in,
                              const std::string& pdb_out,
                              bool normalize_names,
                              bool build_bonds) {
    return translate_ball_exceptions([&] {
        // Load with H-placement deferred so we can run the processor
        // explicitly and report how many hydrogens were placed.
        auto sys = load_pdb(pdb_in, normalize_names, build_bonds, /*add_hydrogens=*/false);
        AddHydrogenProcessor hp;
        sys->apply(hp);

        // Write the result. PDBFile::write is the inverse of open+read;
        // create the parent dir if needed so a fresh dest path works
        // without prep.
        auto out_path = std::filesystem::path(pdb_out);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        PDBFile out;
        out.open(pdb_out, std::ios::out);
        if (!out.isOpen()) {
            throw std::runtime_error("BALL: cannot open output PDB: " + pdb_out);
        }
        if (!out.write(*sys)) {
            out.close();
            throw std::runtime_error("BALL: failed to write PDB: " + pdb_out);
        }
        out.close();

        py::dict d;
        d["pdb_out"] = pdb_out;
        d["n_atoms_in"] = sys->countAtoms() - hp.getNumberOfAddedHydrogens();
        d["n_hydrogens_added"] = hp.getNumberOfAddedHydrogens();
        d["n_atoms_out"] = sys->countAtoms();
        return d;
    });
}

// gb_energy — Generalized-Born solvation
//
// TODO(v0.3): GeneralizedBornModel needs scaling factors via an INI
// file (`setScalingFactorFile`) and explicit solute / solvent
// dielectric constants before `setup(*sys)` will succeed. The
// straight-through pattern that works for AmberFF / CharmmFF is
// not enough here. Defer until either:
//   - a default GB INI ships in BALL's data tree and we can locate
//     it via Path::find(), or
//   - this binding grows a `gb_options=` kwarg taking a dict of
//     scaling factors keyed by atom type.
//
// Workaround for proteon's gb_obc claim: use the existing
// charmm_energy with use_eef1=true. EEF1 is BALL's preferred
// implicit-solvation model and is fully wired through CharmmFF.

// ---------------------------------------------------------------------------
// sasa — NumericalSAS with explicit AssignRadiusProcessor pass
// ---------------------------------------------------------------------------

py::dict sasa(const std::string& pdb_path,
              const std::string& radii_file,
              double probe_radius,
              bool normalize_names,
              bool build_bonds,
              bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);

        // PDB-loaded atoms have radius=0; NumericalSAS reads
        // atom.getRadius() per atom and silently returns 0 area when
        // every radius is 0. AssignRadiusProcessor walks the system,
        // matches each atom against an INI dictionary keyed by element
        // / type, and writes the per-atom radius. Default file is
        // PARSE.siz — the standard radius set used in implicit-solvent
        // and SASA contexts; pass `radii_file=""` or override with
        // amber94.siz for AMBER-consistent radii.
        AssignRadiusProcessor radii(radii_file);
        sys->apply(radii);

        NumericalSAS nsas;
        nsas.options[NumericalSAS::Option::PROBE_RADIUS] =
            std::to_string(probe_radius);
        // operator() is the compute step; getTotalArea() / getAtomAreas()
        // expose the result.
        nsas(*sys);

        py::dict d;
        d["total_area"] = nsas.getTotalArea();
        d["total_volume"] = nsas.getTotalVolume();
        d["n_atoms"] = sys->countAtoms();
        d["radii_file"] = radii_file;
        d["probe_radius"] = probe_radius;

        py::dict per_atom;
        const auto& atom_areas = nsas.getAtomAreas();
        for (auto it = sys->beginAtom(); +it; ++it) {
            const Atom& a = *it;
            auto found = atom_areas.find(&a);
            if (found != atom_areas.end()) {
                per_atom[py::str(atom_key(a))] = found->second;
            }
        }
        d["per_atom"] = per_atom;
        return d;
    });
}

// ---------------------------------------------------------------------------
// build_bonds — BuildBondsProcessor wrapper
// ---------------------------------------------------------------------------

py::dict build_bonds_in_pdb(const std::string& pdb_in,
                            const std::string& pdb_out,
                            bool normalize_names) {
    return translate_ball_exceptions([&] {
        // Don't run FragmentDB::build_bonds during load — this binding's
        // whole job is to compare against BALL's geometric bond inference.
        auto sys = load_pdb(pdb_in,
                            /*normalize_names=*/normalize_names,
                            /*build_bonds=*/false,
                            /*add_hydrogens=*/false);

        // Atoms-only count BEFORE bonding; downstream comparison cares
        // about how many bonds were inferred, not how many atoms exist.
        const std::size_t atoms = sys->countAtoms();

        BuildBondsProcessor bbp;
        sys->apply(bbp);

        const std::size_t bonds_built =
            static_cast<std::size_t>(bbp.getNumberOfBondsBuilt());

        if (!pdb_out.empty()) {
            auto out_path = std::filesystem::path(pdb_out);
            if (out_path.has_parent_path()) {
                std::filesystem::create_directories(out_path.parent_path());
            }
            PDBFile out;
            out.open(pdb_out, std::ios::out);
            if (!out.isOpen()) {
                throw std::runtime_error("BALL: cannot open output PDB: " + pdb_out);
            }
            if (!out.write(*sys)) {
                out.close();
                throw std::runtime_error("BALL: failed to write PDB: " + pdb_out);
            }
            out.close();
        }

        py::dict d;
        d["n_atoms"] = atoms;
        d["n_bonds_built"] = bonds_built;
        d["pdb_out"] = pdb_out;
        return d;
    });
}

// ---------------------------------------------------------------------------
// atom_typer — per-atom AMBER type + charge enumeration
// ---------------------------------------------------------------------------
//
// Returns a list of dicts, one per atom, with the AMBER96 type name
// and assigned partial charge. Useful as a preflight oracle: a
// downstream consumer can compare BALL's typing decisions against
// proteon's, and a divergence here fully explains downstream energy
// gaps. We pull the values off the Atom objects after AmberFF setup
// — those are the assignments the energy actually used.

py::list atom_typer(const std::string& pdb_path,
                    bool normalize_names,
                    bool build_bonds,
                    bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);

        AmberFF ff;
        ff.options[AmberFF::Option::NONBONDED_CUTOFF] = "1e6";
        if (!ff.setup(*sys)) {
            throw std::runtime_error(
                "BALL: AmberFF setup failed during atom typing"
            );
        }

        py::list out;
        for (auto it = sys->beginAtom(); +it; ++it) {
            const Atom& a = *it;
            py::dict entry;
            entry["atom"] = atom_key(a);
            entry["type_name"] = std::string(a.getTypeName());
            entry["charge"] = a.getCharge();
            entry["element"] = std::string(a.getElement().getSymbol());
            out.append(entry);
        }
        return out;
    });
}

// ---------------------------------------------------------------------------
// minimize_energy — wrap ConjugateGradient or SteepestDescent
// ---------------------------------------------------------------------------
//
// The minimizer surface is intentionally narrow: pick FF + algorithm
// + max-iter + (optional) write-out path. Per-step trajectory dumping,
// snapshot managers, and the Lewenstein/StrangLBFGS variants are not
// exposed in v0.1; they lengthen the option list disproportionately
// for the typical "minimize this PDB and tell me the final energy"
// use case.

py::dict minimize_energy(const std::string& pdb_in,
                         const std::string& pdb_out,
                         const std::string& ff,
                         const std::string& method,
                         int max_iter,
                         double nonbonded_cutoff,
                         bool use_eef1,
                         bool normalize_names,
                         bool build_bonds,
                         bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_in, normalize_names, build_bonds, add_hydrogens);

        // Tagged-union over the two FF types — using std::variant or a
        // base-class pointer is overkill for two cases. Each branch
        // sets up its FF, runs the minimizer, then queries final energy
        // through the same EnergyMinimizer accessor.
        std::unique_ptr<ForceField> force_field;
        if (ff == "amber96") {
            auto a = std::make_unique<AmberFF>();
            a->options[AmberFF::Option::NONBONDED_CUTOFF] =
                std::to_string(nonbonded_cutoff);
            if (!a->setup(*sys)) {
                throw std::runtime_error(
                    "BALL: AmberFF setup failed before minimization"
                );
            }
            force_field = std::move(a);
        } else if (ff == "charmm19_eef1") {
            auto c = std::make_unique<CharmmFF>();
            c->options[CharmmFF::Option::USE_EEF1] = use_eef1 ? "true" : "false";
            c->options[CharmmFF::Option::NONBONDED_CUTOFF] =
                std::to_string(nonbonded_cutoff);
            if (!c->setup(*sys)) {
                throw std::runtime_error(
                    "BALL: CharmmFF setup failed before minimization"
                );
            }
            force_field = std::move(c);
        } else {
            throw std::runtime_error(
                "BALL: unknown ff " + ff +
                "; valid: 'amber96', 'charmm19_eef1'"
            );
        }

        const double initial_energy = force_field->updateEnergy();

        std::unique_ptr<EnergyMinimizer> minimizer;
        if (method == "conjugate-gradient") {
            minimizer = std::make_unique<ConjugateGradientMinimizer>(*force_field);
        } else if (method == "steepest-descent") {
            minimizer = std::make_unique<SteepestDescentMinimizer>(*force_field);
        } else {
            throw std::runtime_error(
                "BALL: unknown minimization method " + method +
                "; valid: 'conjugate-gradient', 'steepest-descent'"
            );
        }
        minimizer->setMaxNumberOfIterations(static_cast<Size>(max_iter));

        // Pass `max_iter` explicitly: minimize() defaults its `steps`
        // parameter to 0, which the inner loop interprets as "run 0
        // iterations" — not "run until converged or hit
        // setMaxNumberOfIterations." The setter alone is silent.
        const bool converged =
            minimizer->minimize(static_cast<Size>(max_iter));
        const double final_energy = force_field->updateEnergy();
        const std::size_t iterations =
            static_cast<std::size_t>(minimizer->getNumberOfIterations());

        if (!pdb_out.empty()) {
            auto out_path = std::filesystem::path(pdb_out);
            if (out_path.has_parent_path()) {
                std::filesystem::create_directories(out_path.parent_path());
            }
            PDBFile out;
            out.open(pdb_out, std::ios::out);
            if (!out.isOpen()) {
                throw std::runtime_error("BALL: cannot open output PDB: " + pdb_out);
            }
            if (!out.write(*sys)) {
                out.close();
                throw std::runtime_error("BALL: failed to write PDB: " + pdb_out);
            }
            out.close();
        }

        py::dict d;
        d["ff"] = ff;
        d["method"] = method;
        d["max_iter"] = max_iter;
        d["iterations"] = iterations;
        d["converged"] = converged;
        d["initial_energy"] = initial_energy;
        d["final_energy"] = final_energy;
        d["energy_drop"] = initial_energy - final_energy;
        d["pdb_out"] = pdb_out;
        d["n_atoms"] = sys->countAtoms();
        return d;
    });
}

// ---------------------------------------------------------------------------
// system_info — atom/residue/chain enumeration
// ---------------------------------------------------------------------------
//
// Cheap I/O parity check. A consumer that wants to confirm proteon
// and BALL parsed the same PDB into the same hierarchy can compare
// these counts plus the chain id list. Divergence here is a
// load-side bug, not a force-field one.

py::dict system_info(const std::string& pdb_path,
                     bool normalize_names,
                     bool build_bonds,
                     bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);

        std::size_t n_residues = 0;
        for (auto rit = sys->beginResidue(); +rit; ++rit) {
            ++n_residues;
        }

        py::list chain_ids;
        std::size_t n_chains = 0;
        for (auto cit = sys->beginChain(); +cit; ++cit) {
            const Chain& c = *cit;
            chain_ids.append(std::string(c.getName()));
            ++n_chains;
        }

        py::dict d;
        d["n_atoms"] = sys->countAtoms();
        d["n_residues"] = n_residues;
        d["n_chains"] = n_chains;
        d["chain_ids"] = chain_ids;
        return d;
    });
}

py::dict amber_energy(const std::string& pdb_path,
                      double nonbonded_cutoff,
                      bool normalize_names,
                      bool build_bonds,
                      bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);
        AmberFF ff;
        apply_cutoff(ff, nonbonded_cutoff);
        if (!ff.setup(*sys)) {
            throw std::runtime_error("BALL: AmberFF setup failed (atom typing or charge assignment)");
        }
        ff.updateEnergy();
        py::dict d;
        d["bond_stretch"] = ff.getStretchEnergy();
        d["angle_bend"]   = ff.getBendEnergy();
        d["torsion"]      = ff.getTorsionEnergy();
        d["nonbonded"]    = ff.getNonbondedEnergy();
        d["total"]        = ff.getEnergy();
        d["n_atoms"]      = sys->countAtoms();
        return d;
    });
}

py::dict charmm_energy(const std::string& pdb_path,
                       bool use_eef1,
                       double nonbonded_cutoff,
                       bool normalize_names,
                       bool build_bonds,
                       bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);
        CharmmFF ff;
        ff.options[CharmmFF::Option::USE_EEF1] = use_eef1 ? "true" : "false";
        apply_cutoff(ff, nonbonded_cutoff);
        if (!ff.setup(*sys)) {
            throw std::runtime_error("BALL: CharmmFF setup failed (atom typing or charge assignment)");
        }
        ff.updateEnergy();
        py::dict d;
        d["bond_stretch"]      = ff.getStretchEnergy();
        d["angle_bend"]        = ff.getBendEnergy();
        // CharmmFF::getTorsionEnergy() returns proper + improper summed.
        // Surface them split so consumers can compare per-component.
        // proper_torsion = total torsion - improper_torsion.
        const double torsion_total    = ff.getTorsionEnergy();
        const double improper_torsion = ff.getImproperTorsionEnergy();
        d["proper_torsion"]    = torsion_total - improper_torsion;
        d["improper_torsion"]  = improper_torsion;
        d["torsion"]           = torsion_total;  // kept for backward compat / total
        d["vdw"]               = ff.getVdWEnergy();
        d["nonbonded"]         = ff.getNonbondedEnergy();
        d["solvation"]         = ff.getSolvationEnergy();
        d["total"]             = ff.getEnergy();
        d["n_atoms"]           = sys->countAtoms();
        d["use_eef1"]          = use_eef1;
        return d;
    });
}

// ---------------------------------------------------------------------------
// mmff94_energy — MMFF94 force-field single-point
// ---------------------------------------------------------------------------
//
// MMFF94 is a small-molecule / drug-like organic force field — it was
// designed for ligands, not polypeptides. On protein-only PDBs the
// atom-typer will frequently bail out with TooManyErrors because the
// peptide backbone uses bonding patterns MMFF94's typing rules don't
// recognise. The binding still surfaces the function so the right
// fixture (a parsed ligand: SDF→PDB or a small organic molecule) can
// reach BALL's MMFF94 implementation as an oracle.
//
// Energy components surfaced (matching CharmmFF's split philosophy):
//   stretch / bend / stretch_bend (cross term, MMFF94-specific) /
//   torsion (proper + improper) / plane (out-of-plane bend) /
//   vdw (vdW + H-bond) / electrostatic / nonbonded (vdw + es).
//
// The MMFF94 class itself is named just `MMFF94` rather than
// `Mmff94FF` — confusingly close to the namespace, but it's the BALL
// convention for this force field.

py::dict mmff94_energy(const std::string& pdb_path,
                       double nonbonded_cutoff,
                       bool normalize_names,
                       bool build_bonds,
                       bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        auto sys = load_pdb(pdb_path, normalize_names, build_bonds, add_hydrogens);
        MMFF94 ff;
        // MMFF94 shares the option-name vocabulary with AmberFF / CharmmFF
        // for these three knobs, but it does NOT inherit from them, so
        // the apply_cutoff template has to be specialised per-FF — easier
        // to set the strings directly.
        ff.options[MMFF94::Option::NONBONDED_CUTOFF]      = std::to_string(nonbonded_cutoff);
        ff.options[MMFF94::Option::VDW_CUTOFF]            = std::to_string(nonbonded_cutoff);
        ff.options[MMFF94::Option::ELECTROSTATIC_CUTOFF]  = std::to_string(nonbonded_cutoff);
        if (!ff.setup(*sys)) {
            throw std::runtime_error(
                "BALL: MMFF94 setup failed (atom typing or charge assignment). "
                "MMFF94 is a small-molecule force field; protein-only inputs "
                "frequently fail here. Use AMBER96 or CHARMM19 for polypeptides."
            );
        }
        ff.updateEnergy();
        py::dict d;
        d["stretch"]        = ff.getStretchEnergy();
        d["bend"]           = ff.getBendEnergy();
        d["stretch_bend"]   = ff.getStretchBendEnergy();
        d["torsion"]        = ff.getTorsionEnergy();
        d["plane"]          = ff.getPlaneEnergy();
        d["vdw"]            = ff.getVdWEnergy();
        d["electrostatic"]  = ff.getESEnergy();
        d["nonbonded"]      = ff.getNonbondedEnergy();
        d["total"]          = ff.getEnergy();
        d["n_atoms"]        = sys->countAtoms();
        return d;
    });
}

// ---------------------------------------------------------------------------
// rmsd — Kabsch-aligned RMSD between two PDBs
// ---------------------------------------------------------------------------
//
// Wraps AtomBijection (atom-pairing) + RMSDMinimizer (Kabsch via the
// Coutsalis et al. eigenvalue method). Closes a real gap on the
// proteon side: proteon exposes `rmsd()` in its public Python API but
// has no oracle today. With this binding, `proteon.geometry.rmsd` can
// be cross-checked against `ball.rmsd` on identical fixtures.
//
// Pairing modes mirror BALL's AtomBijection methods:
//   ca         — C-alpha atoms only, in residue order (default; the
//                canonical structural-alignment metric for proteins)
//   backbone   — CA + C + N + H + O atoms across all residues
//   name       — fully-qualified <chain>:<residue>:<id>:<atom_name>
//                match (strict; will pair only atoms present in both
//                with the same name and residue context)
//   all        — assignTrivial: pair atoms in iteration order, stop
//                at the smaller of the two structures. Dangerous when
//                the two PDBs have different atom counts/orders, but
//                useful for "I know these are the same atoms in the
//                same order" cases (e.g. before/after minimization).
//
// superpose toggles whether to apply the RMSD-optimal Kabsch transform
// before measuring deviation. False → calculateRMSD (no superposition,
// useful for "how far did the structure drift from its starting pose"
// in trajectory analysis). True → minimum-RMSD over all rigid motions.

py::dict rmsd(const std::string& pdb_a,
              const std::string& pdb_b,
              const std::string& atoms,
              bool superpose,
              bool normalize_names,
              bool build_bonds,
              bool add_hydrogens) {
    return translate_ball_exceptions([&] {
        // RMSD doesn't need bonds or hydrogens for the typical CA / backbone
        // case, so the defaults at the binding boundary differ from the
        // force-field bindings. Callers who DO want hydrogens included
        // (e.g. when computing all-atom RMSD across two minimised structures)
        // can flip add_hydrogens=true at the call site.
        auto sys_a = load_pdb(pdb_a, normalize_names, build_bonds, add_hydrogens);
        auto sys_b = load_pdb(pdb_b, normalize_names, build_bonds, add_hydrogens);

        AtomBijection bijection;
        Size matched = 0;
        if (atoms == "ca") {
            matched = bijection.assignCAlphaAtoms(*sys_a, *sys_b);
        } else if (atoms == "backbone") {
            matched = bijection.assignBackboneAtoms(*sys_a, *sys_b);
        } else if (atoms == "name") {
            matched = bijection.assignByName(*sys_a, *sys_b);
        } else if (atoms == "all") {
            matched = bijection.assignTrivial(*sys_a, *sys_b);
        } else {
            throw std::runtime_error(
                "BALL: unknown atoms mode '" + atoms +
                "'; valid: 'ca', 'backbone', 'name', 'all'"
            );
        }

        // RMSDMinimizer::computeTransformation requires at least 3
        // points to solve the eigenvalue problem; surface a clearer
        // diagnostic than BALL's TooFewCoordinates exception.
        if (matched < 3) {
            throw std::runtime_error(
                "BALL: only " + std::to_string(matched) +
                " atoms paired between the two PDBs (need >= 3 for RMSD); "
                "try a different atoms mode or check that both structures "
                "represent the same molecule"
            );
        }

        double rmsd_value = 0.0;
        if (superpose) {
            // computeTransformation returns (Matrix4x4 transform, double rmsd)
            // — the rmsd field is the RMSD AFTER applying the optimal
            // Kabsch transform, i.e. the canonical Kabsch RMSD.
            auto result = RMSDMinimizer::computeTransformation(bijection);
            rmsd_value = result.second;
        } else {
            // Direct RMSD over the bijection without rigid-body alignment.
            rmsd_value = bijection.calculateRMSD();
        }

        py::dict d;
        d["rmsd"]        = rmsd_value;
        d["n_matched"]   = static_cast<std::size_t>(matched);
        d["atoms"]       = atoms;
        d["superpose"]   = superpose;
        d["n_atoms_a"]   = sys_a->countAtoms();
        d["n_atoms_b"]   = sys_b->countAtoms();
        return d;
    });
}

}  // namespace

PYBIND11_MODULE(ball, m) {
    // Locate the bundled BALL data tree and set BALL_DATA_PATH if the
    // user has not already done so. Must run before any m.def() body
    // that could trigger a BALL data lookup. See the helper's comment
    // for why this matters and why it is a free function rather than
    // a static initializer.
    set_ball_data_path_if_unset();

    m.doc() =
        "Minimal Python bindings for BALL force-field evaluation.\n"
        "\n"
        "Exposes single-point AMBER and CHARMM-EEF1 energies on a PDB,\n"
        "broken down into bond / angle / torsion / non-bonded components.\n"
        "Designed as an oracle source for proteon's EVIDENT manifest.";

    m.def("amber_energy",
          &amber_energy,
          py::arg("pdb_path"),
          py::arg("nonbonded_cutoff") = 1e6,
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Compute AMBER force-field energy components on the given PDB.\n"
          "\n"
          "Returns a dict with keys: bond_stretch, angle_bend, torsion,\n"
          "nonbonded, total, n_atoms. Energies are in BALL's reporting\n"
          "units (kJ/mol on the production build).\n"
          "\n"
          "nonbonded_cutoff defaults to 1e6 A which is effectively NoCutoff,\n"
          "matching the proteon-side `nonbonded_cutoff=1e6` convention used\n"
          "in tests/oracle/test_ball_energy.py.\n"
          "\n"
          "normalize_names / build_bonds / add_hydrogens (default True) drive\n"
          "BALL's standard PDB preprocessing: residue/atom name normalization\n"
          "via FragmentDB, bond construction, and hydrogen placement via\n"
          "AddHydrogenProcessor. Set add_hydrogens=False when comparing\n"
          "against an external oracle that pre-places its own hydrogens —\n"
          "BALL's H placement is a documented divergence axis.");

    m.def("charmm_energy",
          &charmm_energy,
          py::arg("pdb_path"),
          py::arg("use_eef1") = true,
          py::arg("nonbonded_cutoff") = 1e6,
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Compute CHARMM force-field energy components on the given PDB.\n"
          "\n"
          "Returns a dict with keys: bond_stretch, angle_bend,\n"
          "proper_torsion, improper_torsion, torsion (= proper + improper),\n"
          "vdw, nonbonded, solvation, total, n_atoms, use_eef1.\n"
          "\n"
          "use_eef1 toggles the EEF1 implicit-solvation model. Set to false\n"
          "to compare against external oracles that lack EEF1 (e.g. OpenMM\n"
          "with toppar_c19 — see the v1.1 oracle plan in\n"
          "proteon/evident/claims/forcefield_charmm19_internal.md).\n"
          "\n"
          "normalize_names / build_bonds / add_hydrogens (default True) drive\n"
          "BALL's standard PDB preprocessing: see amber_energy for details.\n"
          "CHARMM19 is a polar-H force field; non-polar Hs are absorbed into\n"
          "united-carbon types and their presence is harmless when present.");

    m.def("mmff94_energy",
          &mmff94_energy,
          py::arg("pdb_path"),
          py::arg("nonbonded_cutoff") = 1e6,
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Compute MMFF94 force-field energy components on the given PDB.\n"
          "\n"
          "MMFF94 is a small-molecule / drug-like organic force field. It\n"
          "is the right oracle for ligand parity claims, NOT for proteins\n"
          "— the atom-typer expects bonding patterns from organic\n"
          "chemistry and frequently fails on polypeptide backbones with\n"
          "'BALL: MMFF94 setup failed (atom typing or charge\n"
          "assignment)'. Use amber_energy / charmm_energy for proteins.\n"
          "\n"
          "Returns a dict with keys: stretch, bend, stretch_bend,\n"
          "torsion (proper + improper), plane (out-of-plane bend),\n"
          "vdw (vdW + H-bond term), electrostatic, nonbonded\n"
          "(vdw + electrostatic), total, n_atoms.\n"
          "\n"
          "stretch_bend is the MMFF94-specific cross term coupling bond\n"
          "stretching and angle bending — absent in AMBER and CHARMM.\n"
          "Surface it separately so consumers can verify it's not zero\n"
          "when comparing component-by-component.");

    m.def("hbonds",
          &hbonds,
          py::arg("pdb_path"),
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "List hydrogen bonds via HBondProcessor's Kabsch-Sander method.\n"
          "\n"
          "Returns a list of dicts, each with: donor, acceptor (atom keys\n"
          "as '<chain>:<residue><id>:<atom_name>'), length (donor-acceptor\n"
          "distance in A).");

    m.def("secondary_structure",
          &secondary_structure,
          py::arg("pdb_path"),
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Assign secondary structure per residue via\n"
          "SecondaryStructureProcessor.\n"
          "\n"
          "Returns a dict with assignments (list of {residue, ss}) and\n"
          "n_residues. SS letters use the DSSP 3-class reduction:\n"
          "H = helix, E = strand, T = turn, C = coil, '-' = unassigned.");

    m.def("add_hydrogens_to_pdb",
          &add_hydrogens_to_pdb,
          py::arg("pdb_in"),
          py::arg("pdb_out"),
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          "Read pdb_in, place hydrogens via AddHydrogenProcessor, write\n"
          "the hydrogenated structure to pdb_out.\n"
          "\n"
          "Returns a dict with: pdb_out, n_atoms_in (heavy-atom count),\n"
          "n_hydrogens_added, n_atoms_out (total).");

    m.def("sasa",
          &sasa,
          py::arg("pdb_path"),
          py::arg("radii_file") = "radii/PARSE.siz",
          py::arg("probe_radius") = 1.5,
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Compute solvent-accessible surface area via NumericalSAS.\n"
          "\n"
          "Returns a dict with: total_area, total_volume, n_atoms,\n"
          "radii_file, probe_radius, per_atom (dict keyed by\n"
          "'<chain>:<residue><id>:<atom_name>').\n"
          "\n"
          "radii_file is resolved via BALL_DATA_PATH; default\n"
          "'radii/PARSE.siz' is the standard implicit-solvent radius\n"
          "set. Use 'radii/amber94.siz' for AMBER-consistent radii.\n"
          "Areas in A^2; volume in A^3.");

    m.def("build_bonds",
          &build_bonds_in_pdb,
          py::arg("pdb_in"),
          py::arg("pdb_out") = "",
          py::arg("normalize_names") = true,
          "Infer bonds geometrically via BuildBondsProcessor.\n"
          "\n"
          "Returns a dict with: n_atoms, n_bonds_built, pdb_out. When\n"
          "pdb_out is non-empty, writes the post-bonding structure to\n"
          "that path (parent directories created as needed).\n"
          "\n"
          "FragmentDB-driven bond building (the load_pdb default) is\n"
          "deliberately disabled here so BuildBondsProcessor's\n"
          "geometric inference is what produces the count — useful as\n"
          "a comparator against proteon's bond-order pipeline.");

    m.def("atom_typer",
          &atom_typer,
          py::arg("pdb_path"),
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Run AmberFF setup and enumerate per-atom typing decisions.\n"
          "\n"
          "Returns a list of dicts, each: atom (key), type_name (AMBER\n"
          "atom type), charge (partial charge), element (symbol).\n"
          "\n"
          "Useful as a preflight oracle: a divergence in BALL vs\n"
          "proteon energy components is often fully explained by a\n"
          "typing or charge mismatch visible here. Compares directly\n"
          "against proteon's atom-typing pipeline.");

    m.def("minimize_energy",
          &minimize_energy,
          py::arg("pdb_in"),
          py::arg("pdb_out") = "",
          py::arg("ff") = "amber96",
          py::arg("method") = "conjugate-gradient",
          py::arg("max_iter") = 500,
          py::arg("nonbonded_cutoff") = 1e6,
          py::arg("use_eef1") = true,
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Energy-minimize a structure with the chosen force field.\n"
          "\n"
          "Returns a dict with: ff, method, max_iter, iterations\n"
          "(actual count), converged (bool), initial_energy,\n"
          "final_energy, energy_drop, n_atoms, pdb_out.\n"
          "\n"
          "ff: 'amber96' or 'charmm19_eef1'.\n"
          "method: 'conjugate-gradient' or 'steepest-descent'.\n"
          "use_eef1 toggles EEF1 implicit solvation when ff is\n"
          "charmm19_eef1; ignored otherwise.\n"
          "\n"
          "When pdb_out is non-empty, the minimized structure is\n"
          "written to that path. Per-step trajectory dumping and the\n"
          "L-BFGS / shifted-LVMM variants are deferred to v0.3 to\n"
          "keep this surface tight.");

    m.def("system_info",
          &system_info,
          py::arg("pdb_path"),
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = true,
          py::arg("add_hydrogens")   = true,
          "Atom/residue/chain enumeration after the standard load\n"
          "preprocessing. Returns a dict with: n_atoms, n_residues,\n"
          "n_chains, chain_ids (list of strings).\n"
          "\n"
          "Cheap I/O parity check — divergence between BALL and\n"
          "another tool's counts on the same PDB is a load-side bug,\n"
          "not a force-field one.");

    m.def("rmsd",
          &rmsd,
          py::arg("pdb_a"),
          py::arg("pdb_b"),
          py::arg("atoms") = "ca",
          py::arg("superpose") = true,
          py::arg("normalize_names") = true,
          py::arg("build_bonds")     = false,
          py::arg("add_hydrogens")   = false,
          "Compute RMSD between two PDB structures via BALL's\n"
          "RMSDMinimizer (Coutsalis et al. eigenvalue method).\n"
          "\n"
          "Returns a dict with: rmsd (Angstroms), n_matched (atom pairs\n"
          "actually compared), atoms (pairing mode), superpose,\n"
          "n_atoms_a, n_atoms_b.\n"
          "\n"
          "atoms: 'ca' (default; C-alpha only, residue-ordered),\n"
          "       'backbone' (CA + C + N + H + O per residue),\n"
          "       'name' (strict <chain>:<residue>:<id>:<atom> match),\n"
          "       'all' (in-order pairing; assumes identical atom\n"
          "              count/order, e.g. before/after minimization).\n"
          "\n"
          "superpose: when True (default), report the RMSD-optimal\n"
          "Kabsch-aligned RMSD — the canonical structural-similarity\n"
          "metric. When False, report RMSD without superposition\n"
          "(useful for trajectory drift analysis where the absolute\n"
          "frame matters).\n"
          "\n"
          "Defaults differ from the force-field bindings: build_bonds\n"
          "and add_hydrogens default to False because RMSD on CA / backbone\n"
          "doesn't need bond topology or hydrogens, and hydrogen placement\n"
          "is a documented divergence axis between tools that would\n"
          "silently inflate the reported RMSD.");

    // gb_energy: deferred to v0.3 — see TODO in source. Until then,
    // proteon's gb_obc claim should compare proteon vs OpenMM and use
    // ball.charmm_energy(use_eef1=True) as the third comparator for the
    // implicit-solvation story (BALL's EEF1 is the same idea as OBC
    // GB: a per-atom solvation correction at NoCutoff).
}
