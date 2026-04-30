// Python bindings for BALL — analysis surface that mirrors the things
// a downstream consumer (e.g. proteon's EVIDENT oracle suite) wants
// to compare BALL against on a per-PDB basis.
//
// Surface today (v0.1.0a1):
//   amber_energy            single-point AMBER force-field energy
//   charmm_energy           single-point CHARMM force-field energy (EEF1 toggle)
//   hbonds                  hydrogen bond list (HBondProcessor)
//   secondary_structure     per-residue SS assignment (SecondaryStructureProcessor)
//   add_hydrogens_to_pdb    write a hydrogenated PDB (AddHydrogenProcessor)
//
// Deferred to v0.2 (each needs more setup than the surface budget):
//   sasa                    NumericalSAS reads atom.getRadius() per atom;
//                           PDB-loaded atoms have radius=0, and AmberFF
//                           stores radii in its parameter table not on
//                           the atoms themselves. Needs a proper
//                           radiusRuleProcessor pass before SASA can
//                           land. Until then, proteon's SASA claim
//                           uses Biopython + FreeSASA as oracles.
//   gb_energy               GeneralizedBornModel needs scaling-factor INI
//   minimize_energy         per-step minimizer — option-rich
//   build_bonds             bond inference oracle
//   peptide_from_sequence   generative fixture builder
//   atom_typer              FF atom-type assignment
//   system_info             atom/residue/chain enumeration
//
// Each binding is a "process and report" function: takes a path,
// returns a structured Python dict/list, raises RuntimeError with the
// underlying BALL message on failure. State-bearing class bindings
// (PySystem, PyAtom, etc.) are deferred — most oracle comparisons
// don't need them.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
#include <BALL/MOLMEC/AMBER/amber.h>
#include <BALL/MOLMEC/CHARMM/charmm.h>
#include <BALL/STRUCTURE/fragmentDB.h>
#include <BALL/STRUCTURE/addHydrogenProcessor.h>
#include <BALL/STRUCTURE/HBondProcessor.h>
#include <BALL/STRUCTURE/secondaryStructureProcessor.h>

namespace py = pybind11;
using namespace BALL;

namespace {

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
// TODO(v0.2): GeneralizedBornModel needs scaling factors via an INI
// file (`setScalingFactorFile`) and explicit solute / solvent
// dielectric constants before `setup(*sys)` will succeed. The
// straight-through pattern that works for AmberFF / CharmmFF /
// NumericalSAS is not enough here. Defer until either:
//   - a default GB INI ships in BALL's data tree and we can locate
//     it via Path::find(), or
//   - this binding grows a `gb_options=` kwarg taking a dict of
//     scaling factors keyed by atom type.
//
// Workaround for proteon's gb_obc claim: use the existing
// charmm_energy with use_eef1=true. EEF1 is BALL's preferred
// implicit-solvation model and is fully wired through CharmmFF.

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

}  // namespace

PYBIND11_MODULE(ball, m) {
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

    // gb_energy: deferred to v0.2 — see TODO in source. Until then,
    // proteon's gb_obc claim should compare proteon vs OpenMM and use
    // ball.charmm_energy(use_eef1=True) as the third comparator for the
    // implicit-solvation story (BALL's EEF1 is the same idea as OBC
    // GB: a per-atom solvation correction at NoCutoff).
}
