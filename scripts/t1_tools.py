"""Test 1: how far the glycine changes the protein's electrons, at a frozen geometry (MOZYME, PM7).

    python t1_tools.py decks ATOMS_CSV WORK_DIR CHARGE      write the decks (A, B, D; C after D)
    python t1_tools.py mmcharges WORK_DIR                   C's point charges from D's ESP charges
    python t1_tools.py analyse WORK_DIR OUT_DIR             tables and figures

A = protein + glycine, B = protein alone, C = protein + the glycine as PM7 ESP point charges (MM_CHARGES),
D = glycine alone (ESP charges). Every protein atom keeps its place and its index in A, B and C, so
A - B is the whole effect of the glycine on the protein's charges and forces, C - B the part a fixed set of
point charges gives (the classical field), A - C what the quantum glycine adds (charge transfer, exchange,
polarisation of the glycine, its true charge distribution). Repeats of the same job give the noise.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

COMMON = "PM7 1SCF GRADIENTS GEO-OK AUX(PRECISION=4) SCFCRT=1.D-7 ITRY=500"


def load(atoms_csv):
    rows = list(csv.DictReader(open(atoms_csv)))
    prot = [(r["element"], np.array([float(r[k]) for k in "xyz"])) for r in rows if r["resname"] != "LIG"]
    lig = [(r["element"], np.array([float(r[k]) for k in "xyz"])) for r in rows if r["resname"] == "LIG"]
    meta = [(r["resname"], r["resid"], r["atom"], r["kind"]) for r in rows if r["resname"] != "LIG"]
    return prot, lig, meta


def deck(atoms, kw, title):
    lines = [kw, title, ""]
    lines += [f"{el:2s} {x[0]:12.6f} 1 {x[1]:12.6f} 1 {x[2]:12.6f} 1" for el, x in atoms]
    return "\n".join(lines) + "\n"


VARIANTS = {          # name: (extra keywords, environment, MOZYME?)
    "vac": ("", {}, True), "r2": ("", {}, True), "cpu": ("", {"MOPAC_NOGPU": "1"}, True),
    "c15": ("CUTOFF=15", {}, True), "c25": ("CUTOFF=25", {}, True), "eps": ("EPS=78.4", {}, True),
    "conv": ("", {}, False),
}


def write_decks(atoms_csv, wd, charge):
    prot, lig, meta = load(atoms_csv)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "meta.json").write_text(json.dumps({"meta": meta, "lig": [x.tolist() for _, x in lig],
                                              "prot": [x.tolist() for _, x in prot]}))
    (wd / "D").mkdir(exist_ok=True)
    (wd / "D" / "job.mop").write_text(deck(lig, "PM7 1SCF CHARGE=0 ESP GEO-OK AUX(PRECISION=4)", "glycine alone"))
    for v, (kw, env, mozyme) in VARIANTS.items():
        for sysname, atoms in (("A", prot + lig), ("B", prot), ("C", prot)):
            d = wd / f"{sysname}_{v}"
            d.mkdir(exist_ok=True)
            extra = (" MOZYME" if mozyme else "") + (f" {kw}" if kw else "")
            if sysname == "C":
                extra += ' MM_CHARGES="glycine.chg"' if mozyme else " QMMM"
            (d / "job.mop").write_text(deck(atoms, f"{COMMON} CHARGE={charge}{extra}", f"{sysname} {v}"))
            (d / "env.json").write_text(json.dumps(env))


def aux_block(text, key):
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
    return None


def write_mmcharges(wd):
    meta = json.loads((wd / "meta.json").read_text())
    q = aux_block((wd / "D" / "job.aux").read_text(errors="ignore"), "ELECTROSTATIC_POTENTIAL_CHARGES")
    lig = np.array(meta["lig"])
    prot = np.array(meta["prot"])
    text = "".join(f"{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {c:.6f}\n" for x, c in zip(lig, q))
    pot = [332.0637 * float((q / np.linalg.norm(lig - x, axis=1)).sum()) for x in prot]
    molin = f"glycine ESP charges, potential at each protein atom (kcal/mol, +1 charge)\n{len(prot)} 0\n" + \
        "".join(f"x y z q {v:.6f}\n" for v in pot)
    for d in wd.glob("C_*"):
        (d / "glycine.chg").write_text(text)
        (d / "mol.in").write_text(molin)
    return q


def read_run(d, n):
    aux = d / "job.aux"
    if not aux.exists():
        return None
    t = aux.read_text(errors="ignore")
    q = aux_block(t, "ATOM_CHARGES")
    g = aux_block(t, "GRADIENTS")
    x = aux_block(t, "ATOM_X_OPT") if aux_block(t, "ATOM_X_OPT") is not None else aux_block(t, "ATOM_X")
    if q is None:
        return None
    return {"q": q[:n], "g": g.reshape(-1, 3)[:n] if g is not None and len(g) >= 3 * n else None,
            "x": x.reshape(-1, 3)[:n] if x is not None else None}


BINS = [0, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40]


def binned(r, v):
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        s = v[(r >= lo) & (r < hi)]
        out.append((lo, hi, len(s), np.median(s) if len(s) else np.nan, np.percentile(s, 95) if len(s) else np.nan,
                    s.max() if len(s) else np.nan))
    return out


def analyse(wd, od):
    od.mkdir(parents=True, exist_ok=True)
    meta = json.loads((wd / "meta.json").read_text())
    prot, lig = np.array(meta["prot"]), np.array(meta["lig"])
    n = len(prot)
    r = np.linalg.norm(prot[:, None] - lig[None], axis=2).min(1)
    runs = {}
    for d in sorted(wd.glob("[ABC]_*")):
        res = read_run(d, n)
        if res is not None:
            if res["x"] is not None:
                dev = np.abs(res["x"] - prot).max()
                if dev > 1e-3:
                    raise RuntimeError(f"{d.name}: atom order or geometry differs from the input ({dev:.4f} A)")
            runs[d.name] = res
    lines = [f"runs read: {', '.join(sorted(runs))}", f"protein atoms: {n}; farthest from the glycine: {r.max():.1f} A"]
    table = []

    def diff(a, b, what):
        if a not in runs or b not in runs:
            return None
        if what == "q":
            return np.abs(runs[a]["q"] - runs[b]["q"])
        if runs[a]["g"] is None or runs[b]["g"] is None:
            return None
        return np.linalg.norm(runs[a]["g"] - runs[b]["g"], axis=1)

    for what, unit in (("q", "e"), ("g", "kcal/mol/A")):
        lines.append(f"\n### {'atomic charges' if what == 'q' else 'forces'} ({unit}): median / 95th percentile / "
                     f"largest per distance bin")
        pairs = [("A-B (whole effect)", "A_vac", "B_vac"), ("C-B (point charges)", "C_vac", "B_vac"),
                 ("A-C (beyond point charges)", "A_vac", "C_vac"),
                 ("noise A-A", "A_vac", "A_r2"), ("noise B-B", "B_vac", "B_r2"), ("noise C-C", "C_vac", "C_r2"),
                 ("GPU-CPU A", "A_vac", "A_cpu"),
                 ("A-C, CPU", "A_cpu", "C_cpu"), ("A-B, CPU", "A_cpu", "B_cpu"),
                 ("A-C, CUTOFF=15", "A_c15", "C_c15"), ("A-C, CUTOFF=25", "A_c25", "C_c25"),
                 ("A-B, CUTOFF=25", "A_c25", "B_c25"),
                 ("A-B, water (COSMO)", "A_eps", "B_eps"), ("A-C, water (COSMO)", "A_eps", "C_eps"),
                 ("A-B, conventional SCF", "A_conv", "B_conv")]
        if what == "q":
            pairs.append(("A-C, conventional SCF (C: point potentials)", "A_conv", "C_conv"))
        hdr = "  " + " " * 44 + "".join(f"{lo:>4d}-{hi:<4d}" + " " * 14 for lo, hi in zip(BINS, BINS[1:]))
        lines.append("  bins (A from the glycine): " + ", ".join(f"{lo}-{hi}" for lo, hi in zip(BINS, BINS[1:])))
        for label, a, b in pairs:
            v = diff(a, b, what)
            if v is None:
                continue
            bs = binned(r, v)
            table += [{"quantity": what, "pair": label, "lo": lo, "hi": hi, "n": k, "median": med, "p95": p95,
                       "max": mx} for lo, hi, k, med, p95, mx in bs]
            lines.append(f"  {label:44s} " + " | ".join(f"{med:.1e}/{p95:.1e}/{mx:.1e}" for _, _, _, med, p95, mx in bs))
    with open(od / "binned.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "pair", "lo", "hi", "n", "median", "p95", "max"])
        w.writeheader()
        w.writerows(table)
    # residues: largest |A-C| charge change per residue against its distance; the aromatic path singled out
    if "A_vac" in runs and "C_vac" in runs:
        dq = np.abs(runs["A_vac"]["q"] - runs["C_vac"]["q"])
        noise = np.abs(runs["A_vac"]["q"] - runs["A_r2"]["q"]) if "A_r2" in runs else None
        res = {}
        for i, (rn, rid, an, kind) in enumerate(meta["meta"]):
            k = f"{rn}{rid}"
            res.setdefault(k, []).append(i)
        path = {"PHE92", "TRP223", "PHE221", "TYR203", "HIS201", "HIS197", "TYR173", "PHE171"}
        lines.append("\n### residues: largest |A-C| charge change (e) and distance of the residue from the glycine (A)")
        rows = sorted(((dq[ix].max(), r[ix].min(), k) for k, ix in res.items()), key=lambda t: t[1])
        for lo, hi in zip(BINS, BINS[1:]):
            sel = [t for t in rows if lo <= t[1] < hi]
            if not sel:
                continue
            top = max(sel)
            pth = [f"{k}:{m:.1e}" for m, d_, k in sel if k in path]
            lines.append(f"  {lo:>2d}-{hi:<2d} A: {len(sel):3d} residues, median {np.median([t[0] for t in sel]):.1e}, "
                         f"largest {top[0]:.1e} ({top[2]})" + (f"; path residues {', '.join(pth)}" if pth else ""))
        if noise is not None:
            lines.append(f"  noise (A twice): median {np.median(noise):.1e}, 99th percentile {np.percentile(noise, 99):.1e},"
                         f" largest {noise.max():.1e}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax, what, unit in ((axs[0], "q", "e"), (axs[1], "g", "kcal/mol/Å")):
                for label, a, b, c in (("A−B", "A_vac", "B_vac", "#1f5fa8"), ("C−B", "C_vac", "B_vac", "#9a9a9a"),
                                       ("A−C", "A_vac", "C_vac", "#c2452d"), ("noise A−A", "A_vac", "A_r2", "#2b8a3e")):
                    v = diff(a, b, what)
                    if v is not None:
                        ax.scatter(r, np.maximum(v, 1e-9), s=2, alpha=0.35, color=c, label=label)
                ax.set_yscale("log")
                ax.set_xlabel("distance from the glycine (Å)")
                ax.set_ylabel(f"|change| ({unit})")
                ax.legend(markerscale=5, fontsize=8)
            fig.tight_layout()
            fig.savefig(od / "decay.png", dpi=150)
        except Exception as e:      # the figure is optional
            lines.append(f"(no figure: {e})")
    (od / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "decks":
        write_decks(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "mmcharges":
        print(write_mmcharges(Path(sys.argv[2])))
    elif cmd == "analyse":
        analyse(Path(sys.argv[2]), Path(sys.argv[3]))
