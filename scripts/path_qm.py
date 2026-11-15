"""Test 2: site energies and couplings along a candidate hole path (PM7, conventional SCF).

Usage: python -I path_qm.py MOPAC PREP_DIR WORK_DIR OUT_TXT

Site model: the side chain from CB on, with a hydrogen in place of CA (1.09 A along CB->CA); the glycine
ligand whole. Site energy: vertical ionisation energy, heat of formation of the cation (UHF doublet) minus
that of the neutral model, at the prepared geometry, (a) alone, (b) in the electrostatic potential of the
rest of the protein (Amber charges; glycine: PM7 ESP charges) divided by eps = 4 or 1 (MOPAC keyword QMMM,
file mol.in: the energy of a +1 charge at each model atom). The residue's own backbone N, H, C, O stay in the
environment; its CA, HA and side chain are left out. Coupling: the two highest occupied orbitals of each
model (isolated) projected on the PM7 Fock matrix of the pair (models side by side, no environment):
H_ab = c_a . F . c_b (ZDO basis, orthonormal).
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

KCAL_EV = 23.0605
COUL = 332.0637        # kcal/mol A / e^2
BB_KEEP = {"N", "H", "C", "O", "OXT", "H1", "H2", "H3"}


def load(prep_dir, pid):
    rows = list(csv.DictReader(open(prep_dir / f"{pid}_atoms.csv")))
    for r in rows:
        r["xyz"] = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
        r["q"] = float(r["amber_charge"]) if r["amber_charge"] else None
        r["resid"] = int(r["resid"])
    return rows


def site_model(rows, resid, resname=None):
    """Atoms of the model (element, xyz) and the indices of the atoms it replaces in the environment."""
    if resname == "LIG":
        at = [r for r in rows if r["resname"] == "LIG"]
        return [(r["element"], r["xyz"]) for r in at], {r["index"] for r in at}, 0
    res = [r for r in rows if r["resid"] == resid and r["resname"] != "LIG"]
    by = {r["atom"]: r for r in res}
    side = [r for r in res if r["atom"] not in BB_KEEP | {"CA", "HA", "HA2", "HA3"}]
    cb, ca = by["CB"]["xyz"], by["CA"]["xyz"]
    cap = cb + 1.09 * (ca - cb) / np.linalg.norm(ca - cb)
    atoms = [(r["element"], r["xyz"]) for r in side] + [("H", cap)]
    gone = {r["index"] for r in side} | {by["CA"]["index"]} | ({by["HA"]["index"]} if "HA" in by else set())
    name = res[0]["resname"]
    charge = 1 if name == "HIS" and "HD1" in by and "HE2" in by else 0
    return atoms, gone, charge


def deck(atoms, kw, title):
    lines = [kw, title, ""]
    for el, x in atoms:
        lines.append(f"{el:2s} {x[0]:12.6f} 1 {x[1]:12.6f} 1 {x[2]:12.6f} 1")
    return "\n".join(lines) + "\n"


def run_mopac(mopac, wd, name, atoms, kw, pot=None):
    d = wd / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.mop").write_text(deck(atoms, kw + (" QMMM" if pot is not None else ""), name))
    if pot is not None:
        (d / "mol.in").write_text("potential at each QM atom, kcal/mol for a +1 charge\n%d 0\n" % len(atoms)
                                  + "".join(f"x y z q {v:.6f}\n" for v in pot))
    if not (d / "job.aux").exists():
        subprocess.run([mopac, "job.mop"], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    out = (d / "job.out").read_text(errors="ignore") if (d / "job.out").exists() else ""
    m = re.search(r"FINAL HEAT OF FORMATION =\s+(-?\d+\.\d+)", out)
    if not m:
        raise RuntimeError(f"{name}: no heat of formation; see {d / 'job.out'}")
    return float(m.group(1)), d / "job.aux"


def aux_block(text, key):
    """The N numbers after KEY[N]= (comment lines starting with # skipped)."""
    m = re.search(r"(?<![A-Z_])" + re.escape(key) + r"(?::[A-Z/]+)?\[(\d+)\]=", text)
    if not m:
        return None
    n, vals = int(m.group(1)), []
    for ln in text[m.end():].splitlines():
        if ln.strip().startswith("#"):
            continue
        for tok in ln.split():
            vals.append(float(tok.replace("D", "E")))
            if len(vals) == n:
                return np.array(vals)
    raise ValueError(f"{key}: {len(vals)} of {n} values")


def frontier(aux):
    t = aux.read_text(errors="ignore")
    norbs = len(aux_block(t, "AO_ATOMINDEX"))
    ev = aux_block(t, "EIGENVALUES")
    occ = aux_block(t, "MOLECULAR_ORBITAL_OCCUPANCIES")
    vec = aux_block(t, "EIGENVECTORS").reshape(len(ev), norbs)
    homo = int(np.where(occ > 1.5)[0].max())
    return norbs, [(ev[homo - k], vec[homo - k]) for k in (0, 1)]


def fock(aux):
    t = aux.read_text(errors="ignore")
    norbs = len(aux_block(t, "AO_ATOMINDEX"))
    f = aux_block(t, "FOCK_MATRIX")
    F = np.zeros((norbs, norbs))
    F[np.tril_indices(norbs)] = f
    return F + np.tril(F, -1).T


def potential(rows, gone, atoms, eps):
    env = [r for r in rows if r["index"] not in gone and r["q"] is not None]
    Q = np.array([r["q"] for r in env])
    X = np.array([r["xyz"] for r in env])
    return [COUL / eps * float((Q / np.linalg.norm(X - x, axis=1)).sum()) for _, x in atoms]


def analyse_path(mopac, wd, rows, label, sites, out):
    lines = [f"\n## {label}"]
    lines.append("  site   model charge | vertical ionisation energy, eV: alone | eps=4 | eps=1 "
                 "(relative to the first site: alone / eps=4 / eps=1)")
    info = {}
    ref = None
    for resid, resname, tag in sites:
        atoms, gone, ch = site_model(rows, resid, resname)
        e = {}
        for env, eps in (("vac", None), ("e4", 4.0), ("e1", 1.0)):
            pot = potential(rows, gone, atoms, eps) if eps else None
            h0, _ = run_mopac(mopac, wd, f"{tag}_{env}_n", atoms, f"PM7 1SCF CHARGE={ch} GEO-OK AUX(PRECISION=4)", pot)
            h1, _ = run_mopac(mopac, wd, f"{tag}_{env}_c", atoms, f"PM7 1SCF CHARGE={ch + 1} UHF DOUBLET GEO-OK AUX(PRECISION=4)",
                              pot)
            e[env] = (h1 - h0) / KCAL_EV
        ref = ref or e
        info[tag] = (atoms, ch, e)
        lines.append(f"  {tag:6s} {ch:+d}            | {e['vac']:6.2f} | {e['e4']:6.2f} | {e['e1']:6.2f}   "
                     f"({e['vac'] - ref['vac']:+.2f} / {e['e4'] - ref['e4']:+.2f} / {e['e1'] - ref['e1']:+.2f})")
    lines.append("  pair            closest heavy atoms | coupling |H| between the highest occupied orbitals, meV: "
                 "HOMO-HOMO, largest of the 2x2 | step in eps=4 site energy, eV")
    tags = [t for _, _, t in sites]
    for a, b in zip(tags, tags[1:]):
        A, cha, ea = info[a]
        B, chb, eb = info[b]
        _, aux_a = run_mopac(mopac, wd, f"{a}_vac_n", A, f"PM7 1SCF CHARGE={cha} GEO-OK AUX(PRECISION=4)")
        _, aux_b = run_mopac(mopac, wd, f"{b}_vac_n", B, f"PM7 1SCF CHARGE={chb} GEO-OK AUX(PRECISION=4)")
        _, aux_ab = run_mopac(mopac, wd, f"pair_{a}_{b}", A + B, f"PM7 1SCF CHARGE={cha + chb} GEO-OK AUX(PRECISION=4) FOCK")
        na, fa = frontier(aux_a)
        nb, fb = frontier(aux_b)
        F = fock(aux_ab)
        if F.shape[0] != na + nb:
            raise RuntimeError(f"pair {a}-{b}: {F.shape[0]} orbitals, expected {na} + {nb}")
        H = np.array([[abs(ca @ F[:na, na:] @ cb) for _, cb in fb] for _, ca in fa]) * 1000.0
        ha = np.array([x for el, x in A if el != "H"])
        hb = np.array([x for el, x in B if el != "H"])
        dmin = np.linalg.norm(ha[:, None] - hb[None], axis=2).min()
        lines.append(f"  {a:>6s}-{b:<6s}  {dmin:5.1f} A                 | {H[0, 0]:8.1f} | {H.max():8.1f} | "
                     f"{eb['e4'] - ea['e4']:+.2f}")
    print("\n".join(lines), flush=True)
    out.write("\n".join(lines) + "\n")


def esp_glycine(mopac, wd, rows):
    lig = [r for r in rows if r["resname"] == "LIG"]
    atoms = [(r["element"], r["xyz"]) for r in lig]
    _, aux = run_mopac(mopac, wd, "glycine_esp", atoms, "PM7 1SCF CHARGE=0 ESP GEO-OK AUX(PRECISION=4)")
    q = aux_block(aux.read_text(errors="ignore"), "ELECTROSTATIC_POTENTIAL_CHARGES")
    for r, v in zip(lig, q):
        r["q"] = float(v)
    return q


def main():
    mopac, prep, wd, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    wd.mkdir(parents=True, exist_ok=True)
    out = open(outp, "w")
    g = load(prep, "1PB7")
    q = esp_glycine(mopac, wd / "1PB7", g)
    out.write(f"glycine PM7 ESP charges (sum {q.sum():+.3f}): {np.round(q, 3).tolist()}\n")
    analyse_path(mopac, wd / "1PB7", g, "GluN1 (1PB7), candidate aromatic path from the glycine",
                 [(901, "LIG", "Gly"), (92, None, "F92"), (223, None, "W223"), (221, None, "F221"),
                  (203, None, "Y203"), (201, None, "H201"), (197, None, "H197"), (173, None, "Y173"),
                  (171, None, "F171")], out)
    p = load(prep, "1DNP")
    analyse_path(mopac, wd / "1DNP", p, "photolyase (1DNP), Trp triad (control; FAD and MHF not in the environment)",
                 [(382, None, "W382"), (359, None, "W359"), (306, None, "W306")], out)
    analyse_path(mopac, wd / "1DNP", p, "photolyase (1DNP), branch seen in the geometric search",
                 [(382, None, "W382"), (316, None, "W316")], out)
    out.close()


if __name__ == "__main__":
    main()
