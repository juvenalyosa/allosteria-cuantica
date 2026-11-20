"""Test 1: how far the glycine changes the protein's electrons, at a frozen geometry (MOZYME, PM7).

    python t1_tools.py decks ATOMS_CSV WORK_DIR CHARGE      write the decks (A, B, D; C after D)
    python t1_tools.py mmcharges WORK_DIR                   C's point charges from D's ESP charges
    python t1_tools.py c2 WORK_DIR VARIANT                  C2_VARIANT: the glycine's charges in A_VARIANT
    python t1_tools.py analyse WORK_DIR OUT_DIR             tables and figures

A = protein + glycine, B = protein alone, C = protein + the glycine as PM7 ESP point charges (MM_CHARGES),
D = glycine alone (ESP charges). Every protein atom keeps its place and its index in A, B and C, so
A - B is the whole effect of the glycine on the protein's charges and forces, C - B the part a fixed set of
point charges gives (the classical field), A - C what the quantum glycine adds (charge transfer, exchange,
polarisation of the glycine, its true charge distribution). C2 = the glycine as the point charges it has inside
the protein (from A), so A - C2 leaves out its polarisation by the protein. The noise: the same jobs repeated (GPU)
and with the atoms in reverse order (another path through the SCF to the same answer).
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# MOZYME's density stops improving at ~1.5e-5 with its default LMO threshold (1e-13); RELTHR=0.01 (1e-15) takes it
# to ~2e-6, where the energy wanders by a few 1e-6 kcal/mol between iterations, so SCFCRT=1e-7 never converges.
# XS, XP, XW: no overlap, density or M.O. matrices in the AUX file (1.7 GB per run for this protein; not used here).
COMMON = "PM7 1SCF GRADIENTS GEO-OK AUX(PRECISION=4,XS,XP,XW) SCFCRT=2.D-6 RELTHR=0.01 ITRY=300"


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
    "rev": ("", {}, True),      # atoms in reverse order: same physics, another path through the SCF
    "c15": ("CUTOFF=15", {}, True), "c25": ("CUTOFF=25", {}, True), "eps": ("EPS=78.4", {}, True),
    "conv": ("", {}, False),
    # CUTOFF=15 and 25 failed the SCF at SCFCRT=2.D-6 (L4, 300 iterations): the same at 1.D-5, with its own base
    # and repeat, the GPU SCF trace of one failing job, and the CPU path of the same job
    "vacl": ("SCFCRT=1.D-5", {}, True), "vacl2": ("SCFCRT=1.D-5", {}, True),
    "c15l": ("CUTOFF=15 SCFCRT=1.D-5", {}, True), "c25l": ("CUTOFF=25 SCFCRT=1.D-5", {}, True),
    "c15t": ("CUTOFF=15", {"MOPAC_MOZYME_SCF_TRACE": "1"}, True),
    "c15cpu": ("CUTOFF=15 PL", {"MOPAC_NOGPU": "1"}, True),
}


def keywords(kw):
    """COMMON with the variant's keywords; a keyword=value already in COMMON is replaced, not repeated."""
    base, extra = COMMON, []
    for tok in kw.split():
        key = tok.split("=")[0] + "="
        if "=" in tok and re.search(rf"(?<!\S){re.escape(key)}", base):
            base = re.sub(rf"(?<!\S){re.escape(key)}\S+", tok, base)
        else:
            extra.append(tok)
    return base, " ".join(extra)


def write_decks(atoms_csv, wd, charge):
    prot, lig, meta = load(atoms_csv)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "meta.json").write_text(json.dumps({"meta": meta, "lig": [x.tolist() for _, x in lig],
                                              "prot": [x.tolist() for _, x in prot]}))
    (wd / "D").mkdir(exist_ok=True)
    (wd / "D" / "job.mop").write_text(deck(lig, "PM7 1SCF CHARGE=0 ESP GEO-OK AUX(PRECISION=4)", "glycine alone"))
    for v, (kw, env, mozyme) in VARIANTS.items():
        for sysname, atoms in (("A", prot + lig), ("B", prot), ("C", prot)):
            if v == "rev":
                atoms = atoms[::-1]
            d = wd / f"{sysname}_{v}"
            d.mkdir(exist_ok=True)
            base, kws = keywords(kw)
            extra = (" MOZYME" if mozyme else "") + (f" {kws}" if kws else "")
            if sysname == "C":
                extra += ' MM_CHARGES="glycine.chg"' if mozyme else " QMMM"
            (d / "job.mop").write_text(deck(atoms, f"{base} CHARGE={charge}{extra}", f"{sysname} {v}"))
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


def write_c2(wd, v):
    """C2_v: deck of C_v, the glycine's charges taken from A_v (its last atoms)."""
    lig = np.array(json.loads((wd / "meta.json").read_text())["lig"])
    q = aux_block((wd / f"A_{v}" / "job.aux").read_text(errors="ignore"), "ATOM_CHARGES")
    if q is None:
        raise SystemExit(f"A_{v}/job.aux has no ATOM_CHARGES (see A_{v}/job.out); C2_{v} not written")
    q = q[-len(lig):]
    d = wd / f"C2_{v}"
    d.mkdir(exist_ok=True)
    lines = (wd / f"C_{v}" / "job.mop").read_text().splitlines()
    lines[1] = f"C2 {v}: the glycine as the point charges it has in A"
    (d / "job.mop").write_text("\n".join(lines) + "\n")
    (d / "env.json").write_text((wd / f"C_{v}" / "env.json").read_text())
    (d / "glycine.chg").write_text("".join(f"{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {c:.6f}\n" for x, c in zip(lig, q)))
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
    g = g.reshape(-1, 3) if g is not None and len(g) == 3 * len(q) else None
    x = x.reshape(-1, 3) if x is not None else None
    if d.name.endswith("_rev"):     # back to the input order
        q, g, x = q[::-1], g[::-1] if g is not None else None, x[::-1] if x is not None else None
    return {"q": q[:n], "g": g[:n] if g is not None else None, "x": x[:n] if x is not None else None}


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
    for d in sorted(wd.glob("[ABC]*_*")):
        res = read_run(d, n)
        if res is not None:
            if res["x"] is not None:
                dev = np.abs(res["x"] - prot).max()
                if dev > 1e-3:
                    raise RuntimeError(f"{d.name}: atom order or geometry differs from the input ({dev:.4f} A)")
            runs[d.name] = res
    lines = [f"runs read: {', '.join(sorted(runs))}", f"protein atoms: {n}; farthest from the glycine: {r.max():.1f} A"]
    table = []

    def change(a, b, what):         # b may be None (a alone) or a pair (a minus the difference of two runs)
        if a not in runs or (b is not None and b not in runs):
            return None
        v = runs[a][what]
        return None if v is None else v if b is None else None if runs[b][what] is None else v - runs[b][what]

    def diff(a, b, what):           # a, b: run names, or pairs of run names (the difference of two differences)
        va = change(*a, what) if isinstance(a, tuple) else change(a, None, what)
        vb = change(*b, what) if isinstance(b, tuple) else change(b, None, what)
        if va is None or vb is None:
            return None
        return np.abs(va - vb) if what == "q" else np.linalg.norm(va - vb, axis=1)

    v0 = "vac" if "A_vac" in runs else "cpu"       # the base runs: GPU, or CPU when there are no GPU runs
    A, B, C = f"A_{v0}", f"B_{v0}", f"C_{v0}"
    lines.append(f"base runs: {v0}")
    for what, unit in (("q", "e"), ("g", "kcal/mol/A")):
        lines.append(f"\n### {'atomic charges' if what == 'q' else 'forces'} ({unit}): median / 95th percentile / "
                     f"largest per distance bin")
        pairs = [("A-B (whole effect)", A, B), ("C-B (point charges)", C, B), ("A-C (beyond point charges)", A, C),
                 ("C2-B (in-situ point charges)", f"C2_{v0}", B), ("A-C2 (beyond in-situ charges)", A, f"C2_{v0}"),
                 ("noise A-A", "A_vac", "A_r2"), ("noise B-B", "B_vac", "B_r2"), ("noise C-C", "C_vac", "C_r2"),
                 ("noise A-A, reversed order", A, "A_rev"), ("noise B-B, reversed order", B, "B_rev"),
                 ("noise C-C, reversed order", C, "C_rev"),
                 ("noise of A-B (reversed order)", (A, B), ("A_rev", "B_rev")),
                 ("noise of C-B (reversed order)", (C, B), ("C_rev", "B_rev")),
                 ("noise of A-C (reversed order)", (A, C), ("A_rev", "C_rev")),
                 ("GPU-CPU A", "A_vac", "A_cpu"),
                 ("A-C, CPU", "A_cpu", "C_cpu"), ("A-B, CPU", "A_cpu", "B_cpu"),
                 ("A-C, CUTOFF=15", "A_c15", "C_c15"), ("A-C, CUTOFF=25", "A_c25", "C_c25"),
                 ("A-B, CUTOFF=25", "A_c25", "B_c25"), ("C2-B, CUTOFF=25", "C2_c25", "B_c25"),
                 ("A-C2, CUTOFF=25", "A_c25", "C2_c25"),
                 ("A-B, water (COSMO)", "A_eps", "B_eps"),
                 # MM_CHARGES enter only the core Hamiltonian; COSMO does not screen them, so in water C and C2
                 # feel the glycine's charges unscreened while A's glycine is screened
                 ("A-C, water (C unscreened)", "A_eps", "C_eps"), ("C2-B, water (C2 unscreened)", "C2_eps", "B_eps"),
                 ("A-C2, water (C2 unscreened)", "A_eps", "C2_eps"),
                 ("A-B, SCFCRT=1e-5", "A_vacl", "B_vacl"), ("C2-B, SCFCRT=1e-5", "C2_vacl", "B_vacl"),
                 ("A-C2, SCFCRT=1e-5", "A_vacl", "C2_vacl"),
                 ("noise A-A, SCFCRT=1e-5", "A_vacl", "A_vacl2"), ("noise B-B, SCFCRT=1e-5", "B_vacl", "B_vacl2"),
                 ("A: SCFCRT 1e-5 - 2e-6", "A_vacl", "A_vac"),
                 ("A-B, CUTOFF=15, 1e-5", "A_c15l", "B_c15l"), ("C2-B, CUTOFF=15, 1e-5", "C2_c15l", "B_c15l"),
                 ("A-C2, CUTOFF=15, 1e-5", "A_c15l", "C2_c15l"),
                 ("A-B, CUTOFF=25, 1e-5", "A_c25l", "B_c25l"), ("C2-B, CUTOFF=25, 1e-5", "C2_c25l", "B_c25l"),
                 ("A-C2, CUTOFF=25, 1e-5", "A_c25l", "C2_c25l"),
                 ("B: CUTOFF=15 - default, 1e-5", "B_c15l", "B_vacl"),
                 ("B: CUTOFF=15, CPU - GPU", "B_c15cpu", "B_c15t"),
                 ("A-B, conventional SCF", "A_conv", "B_conv")]
        if what == "q":
            pairs.append(("A-C, conventional SCF (C: point potentials)", "A_conv", "C_conv"))
        hdr = "  " + " " * 44 + "".join(f"{lo:>4d}-{hi:<4d}" + " " * 14 for lo, hi in zip(BINS, BINS[1:]))
        lines.append("  bins (A from the glycine): " + ", ".join(f"{lo}-{hi}" for lo, hi in zip(BINS, BINS[1:])))
        for label, a, b in pairs:
            v = diff(a, b, what)
            if v is None or (v0 == "cpu" and label.endswith(", CPU")):
                continue
            bs = binned(r, v)
            table += [{"quantity": what, "pair": label, "lo": lo, "hi": hi, "n": k, "median": med, "p95": p95,
                       "max": mx} for lo, hi, k, med, p95, mx in bs]
            lines.append(f"  {label:44s} " + " | ".join(f"{med:.1e}/{p95:.1e}/{mx:.1e}" for _, _, _, med, p95, mx in bs))
        # the question of the CUTOFF runs: is A-C2 still ~0.6 times C2-B far from the glycine?
        rows = []
        for label, a, c2, b in (("default cutoff, 2e-6", "A_vac", "C2_vac", "B_vac"),
                                ("default cutoff, CPU", "A_cpu", "C2_cpu", "B_cpu"),
                                ("default cutoff, 1e-5", "A_vacl", "C2_vacl", "B_vacl"),
                                ("CUTOFF=15, 1e-5", "A_c15l", "C2_c15l", "B_c15l"),
                                ("CUTOFF=25, 1e-5", "A_c25l", "C2_c25l", "B_c25l")):
            v1, v2 = diff(a, c2, what), diff(c2, b, what)
            if v1 is None or v2 is None:
                continue
            rows.append(f"  {label:44s} " + " | ".join(f"{m1 / m2:6.2f}" for (_, _, _, m1, _, _), (_, _, _, m2, _, _)
                                                        in zip(binned(r, v1), binned(r, v2))))
        if rows:
            lines.append(f"  median(A-C2) / median(C2-B) per bin:")
            lines += rows
    with open(od / "binned.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "pair", "lo", "hi", "n", "median", "p95", "max"])
        w.writeheader()
        w.writerows(table)
    # residues: largest |A-C| charge change per residue against its distance; the aromatic path singled out
    if A in runs and C in runs:
        dq = np.abs(runs[A]["q"] - runs[C]["q"])
        noise = diff((A, C), ("A_rev", "C_rev"), "q")
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
            lines.append(f"  noise of A-C (atoms in reverse order): median {np.median(noise):.1e}, 99th percentile "
                         f"{np.percentile(noise, 99):.1e}, largest {noise.max():.1e}")
        # is the aromatic path more affected than other residues as far from the glycine (within 2 A)?
        lines.append("\n### path residues: percent of residues at the same distance (+-2 A) with a smaller largest change")
        for label, a, b in (("A-B", A, B), ("A-C", A, C), ("A-C2", A, f"C2_{v0}")):
            v = diff(a, b, "q")
            if v is None:
                continue
            per = {k: (v[ix].max(), r[ix].min()) for k, ix in res.items()}
            ranks = []
            for k in sorted(path, key=lambda k: per[k][1]):
                m, d_ = per[k]
                peers = np.array([w for w, dd in per.values() if abs(dd - d_) < 2.0])
                ranks.append(f"{k} ({d_:.1f} A) {100 * np.mean(peers < m):.0f}")
            lines.append(f"  {label:5s} " + ", ".join(ranks))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax, what, unit in ((axs[0], "q", "e"), (axs[1], "g", "kcal/mol/Å")):
                for label, a, b, c in (("A−B", A, B, "#1f5fa8"), ("C−B", C, B, "#9a9a9a"), ("A−C", A, C, "#c2452d"),
                                       ("noise of A−C", (A, C), ("A_rev", "C_rev"), "#2b8a3e")):
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
    elif cmd == "c2":
        print(write_c2(Path(sys.argv[2]), sys.argv[3]))
    elif cmd == "analyse":
        analyse(Path(sys.argv[2]), Path(sys.argv[3]))
