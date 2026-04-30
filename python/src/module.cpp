// Minimal Python bindings for BALL force-field evaluation.
//
// Surface intentionally narrow — just enough to live-regenerate
// per-component AMBER and CHARMM-EEF1 energy values from a PDB on the
// proteon EVIDENT manifest's forcefield_amber_ball claim, and to
// enable a future forcefield_charmm19_ball claim that closes the gap
// documented in proteon/evident/claims/forcefield_charmm19_internal.yaml.
//
// Out of scope here: minimization, dynamics, NMR, QSAR, docking,
// trajectory I/O. Those can land later if a proteon claim needs them.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>

#include <BALL/COMMON/exception.h>
#include <BALL/FORMAT/PDBFile.h>
#include <BALL/KERNEL/system.h>
#include <BALL/MOLMEC/AMBER/amber.h>
#include <BALL/MOLMEC/CHARMM/charmm.h>
#include <BALL/STRUCTURE/fragmentDB.h>
#include <BALL/STRUCTURE/addHydrogenProcessor.h>

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
}
