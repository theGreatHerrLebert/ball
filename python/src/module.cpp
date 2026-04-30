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

#include <BALL/FORMAT/PDBFile.h>
#include <BALL/KERNEL/system.h>
#include <BALL/MOLMEC/AMBER/amber.h>
#include <BALL/MOLMEC/CHARMM/charmm.h>

namespace py = pybind11;
using namespace BALL;

namespace {

// Read a PDB file into a freshly-allocated System. Throws on failure
// rather than silently returning an empty system — empty inputs would
// surface as confusing energy=0 results in the bindings.
//
// PDBFile::open returns void (overriding File::open's bool return), so
// success has to be probed via is_open()/good() rather than a return
// value. read() does return bool.
std::unique_ptr<System> load_pdb(const std::string& path) {
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
                      double nonbonded_cutoff) {
    auto sys = load_pdb(pdb_path);
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
}

py::dict charmm_energy(const std::string& pdb_path,
                       bool use_eef1,
                       double nonbonded_cutoff) {
    auto sys = load_pdb(pdb_path);
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
    d["torsion"]           = ff.getTorsionEnergy();
    d["improper_torsion"]  = ff.getImproperTorsionEnergy();
    d["vdw"]               = ff.getVdWEnergy();
    d["nonbonded"]         = ff.getNonbondedEnergy();
    d["solvation"]         = ff.getSolvationEnergy();
    d["total"]             = ff.getEnergy();
    d["n_atoms"]           = sys->countAtoms();
    d["use_eef1"]          = use_eef1;
    return d;
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
          "Compute AMBER force-field energy components on the given PDB.\n"
          "\n"
          "Returns a dict with keys: bond_stretch, angle_bend, torsion,\n"
          "nonbonded, total, n_atoms. Energies are in BALL's reporting\n"
          "units (kJ/mol on the production build).\n"
          "\n"
          "nonbonded_cutoff defaults to 1e6 A which is effectively NoCutoff,\n"
          "matching the proteon-side `nonbonded_cutoff=1e6` convention used\n"
          "in tests/oracle/test_ball_energy.py.");

    m.def("charmm_energy",
          &charmm_energy,
          py::arg("pdb_path"),
          py::arg("use_eef1") = true,
          py::arg("nonbonded_cutoff") = 1e6,
          "Compute CHARMM force-field energy components on the given PDB.\n"
          "\n"
          "Returns a dict with keys: bond_stretch, angle_bend, torsion,\n"
          "improper_torsion, vdw, nonbonded, solvation, total, n_atoms,\n"
          "use_eef1.\n"
          "\n"
          "use_eef1 toggles the EEF1 implicit-solvation model. Set to false\n"
          "to compare against external oracles that lack EEF1 (e.g. OpenMM\n"
          "with toppar_c19 — see the v1.1 oracle plan in\n"
          "proteon/evident/claims/forcefield_charmm19_internal.md).");
}
