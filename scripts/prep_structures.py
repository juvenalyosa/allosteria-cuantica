"""Protonated, restrained-minimised protein structures for the MOPAC tests.

Usage: python -I prep_structures.py PDB_DIR OUT_DIR

PDBFixer adds the missing internal residues and side-chain atoms (terminal gaps are left out) and hydrogens
at pH 7; Amber14 + GBn2 minimisation with every crystal heavy atom restrained (10 000 kJ/mol/nm2), the
modelled atoms and the hydrogens free. Ligands are not in the minimisation: the glycine of 1PB7 gets
hydrogens from ideal geometry (NH3+, CH2) afterwards. Output per structure: <id>_prep.pdb (protein, then the
ligand as residue LIG), <id>_atoms.csv (index, residue, atom, element, Amber charge, crystal or modelled).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
from pdbfixer import PDBFixer


def crystal_heavy(path, chain):
    out = {}
    for ln in open(path):
        if ln.startswith("ATOM") and ln[21] == chain and ln[16] in " A":
            el = ln[76:78].strip()
            if el != "H":
                out[(int(ln[22:26]), ln[12:16].strip())] = np.array([float(ln[30:38]), float(ln[38:46]),
                                                                    float(ln[46:54])]) / 10.0
    return out


def ligand_atoms(path, resname, chain):
    out = []
    for ln in open(path):
        if ln.startswith("HETATM") and ln[17:20].strip() == resname and ln[21] == chain and ln[16] in " A":
            out.append((ln[12:16].strip(), ln[76:78].strip(),
                        np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])))
    return out


def tetra_h(center, bonded, n_h, length, ref=None):
    """Hydrogens on an sp3 centre with the given bonded neighbours (Angstrom)."""
    c = center
    v = [(b - c) / np.linalg.norm(b - c) for b in bonded]
    if n_h == 2 and len(v) == 2:
        bis = -(v[0] + v[1])
        bis /= np.linalg.norm(bis)
        perp = np.cross(v[0], v[1])
        perp /= np.linalg.norm(perp)
        ang = np.radians(109.5 / 2)
        return [c + length * (np.cos(ang) * bis + s * np.sin(ang) * perp) for s in (1, -1)]
    if n_h == 3 and len(v) == 1:
        a = v[0]
        r = ref - c
        r -= (r @ a) * a
        r /= np.linalg.norm(r)
        w = np.cross(a, r)
        t = np.radians(109.5)
        hs = []
        for phi in (180.0, 60.0, -60.0):          # staggered against the reference neighbour
            p = np.radians(phi)
            d = np.cos(t) * a + np.sin(t) * (np.cos(p) * r + np.sin(p) * w)
            hs.append(c + length * d)
        return hs
    raise ValueError


def prepare(pdb_dir, out_dir, pid, chain, ligand=None):
    src = pdb_dir / f"{pid}.pdb"
    flt = out_dir / f"{pid}_protein_only.pdb"          # HETATM records out (the ligand GLY is a standard name)
    flt.write_text("".join(ln for ln in open(src) if not ln.startswith(("HETATM", "CONECT", "ANISOU"))))
    fx = PDBFixer(filename=str(flt))
    fx.removeChains([i for i, c in enumerate(fx.topology.chains()) if c.id != chain])
    fx.findMissingResidues()
    chains = list(fx.topology.chains())
    for key in list(fx.missingResidues):
        ci, ri = key
        if ri == 0 or ri == len(list(chains[ci].residues())):
            del fx.missingResidues[key]
    gaps = {k: v for k, v in fx.missingResidues.items()}
    fx.findNonstandardResidues()
    fx.replaceNonstandardResidues()
    fx.removeHeterogens(keepWater=False)
    fx.findMissingAtoms()
    fx.addMissingAtoms()
    fx.addMissingHydrogens(7.0)
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    cryst = crystal_heavy(src, chain)
    rest = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", 10000.0)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    modelled = []
    for a in fx.topology.atoms():
        key = (int(a.residue.id), a.name)
        if a.element.symbol != "H" and key in cryst:
            rest.addParticle(a.index, list(cryst[key]))
        elif a.element.symbol != "H":
            modelled.append(a.index)
    system.addForce(rest)
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(fx.positions)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy()
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 3000)
    st = ctx.getState(getEnergy=True, getPositions=True)
    pos = st.getPositions(asNumpy=True).value_in_unit(u.angstrom)
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    q = [nb.getParticleParameters(i)[0].value_in_unit(u.elementary_charge) for i in range(system.getNumParticles())]
    dev = max(np.linalg.norm(pos[a.index] - cryst[(int(a.residue.id), a.name)] * 10)
              for a in fx.topology.atoms() if a.element.symbol != "H" and (int(a.residue.id), a.name) in cryst)
    rows = [(a.index, a.residue.name, a.residue.id, a.name, a.element.symbol, q[a.index],
             "modelled" if a.index in modelled else ("H" if a.element.symbol == "H" else "crystal"),
             *pos[a.index]) for a in fx.topology.atoms()]
    lig_rows = []
    if ligand:
        la = ligand_atoms(src, ligand, chain)
        X = {n: x for n, _, x in la}
        lig_rows = [(n, el, x) for n, el, x in la]
        if ligand == "GLY":
            hn = tetra_h(X["N"], [X["CA"]], 3, 1.03, ref=X["C"])
            hca = tetra_h(X["CA"], [X["N"], X["C"]], 2, 1.09)
            lig_rows += [(f"H{i + 1}", "H", h) for i, h in enumerate(hn)] + [("HA2", "H", hca[0]), ("HA3", "H", hca[1])]
    P = pos
    L = np.array([x for _, _, x in lig_rows]) if lig_rows else np.zeros((0, 3))
    clash = float(np.linalg.norm(P[:, None] - L[None], axis=2).min()) if len(L) else None
    lines = []
    n = 0
    for (i, rn, rid, an, el, qq, kind, x, y, z) in rows:
        n += 1
        lines.append("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" %
                     (n, an if len(an) == 4 else " " + an, rn, chain, int(rid), x, y, z, el))
    lines.append("TER")
    for an, el, x in lig_rows:
        n += 1
        lines.append("HETATM%5d %-4s LIG L 901    %8.3f%8.3f%8.3f  1.00  0.00          %2s" %
                     (n, an if len(an) == 4 else " " + an, x[0], x[1], x[2], el))
    lines.append("END")
    (out_dir / f"{pid}_prep.pdb").write_text("\n".join(lines) + "\n")
    with open(out_dir / f"{pid}_atoms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "resname", "resid", "atom", "element", "amber_charge", "kind", "x", "y", "z"])
        for r in rows:
            w.writerow(list(r[:7]) + [f"{v:.4f}" for v in r[7:]])
        for k, (an, el, x) in enumerate(lig_rows):
            w.writerow([len(rows) + k, "LIG", 901, an, el, "", "ligand", *(f"{v:.4f}" for v in x)])
    net = sum(q)
    print(f"{pid} chain {chain}: {len(rows)} protein atoms ({len(modelled)} modelled heavy atoms, gaps built: "
          f"{[(k[1], v) for k, v in gaps.items()]}), Amber net charge {net:+.3f}, ligand {len(lig_rows)} atoms; "
          f"energy {e0.value_in_unit(u.kilojoule_per_mole):.0f} -> "
          f"{st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole):.0f} kJ/mol; largest crystal-atom shift "
          f"{dev:.2f} A; closest protein-ligand contact {clash if clash is None else round(clash, 2)} A",
          flush=True)


if __name__ == "__main__":
    pdb_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare(pdb_dir, out_dir, "1PB7", "A", ligand="GLY")
    prepare(pdb_dir, out_dir, "1DNP", "A")
