"""Hopping paths when the side chains may turn: the crystal conformation and a rotamer scan of every node.

Usage: python -I rotamer_paths.py PDB_DIR OUT_TXT

Each node side chain (Trp, Tyr, Phe, His, Met, free Cys) is rotated about chi1 (staggered values +- 15 deg),
chi2 (every 15 or 30 deg; Met staggered) and chi3 (Met, staggered), keeping the crystal bond geometry. A
conformation is kept if no heavy atom comes closer than CLASH to the fixed atoms: every backbone atom, the side
chains that are not nodes, the ligand and other hetero groups. Two conformations of different nodes that clash
with each other cannot be used together on a hop. The best path is then searched over (node, conformation)
states: the path whose longest hop is shortest, each node in one conformation. Pairs are scored one hop at a
time, so this is the most favourable case for side-chain rotation with the backbone fixed.
"""
import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import survey_paths as s  # noqa: E402

CLASH = 3.0
FAR = 20.0
NODE_ATOMS = {"TRP": s.SIDE["TRP"], "TYR": s.SIDE["TYR"], "FY3": s.SIDE["TYR"], "MET": ("SD", "CE"),
              "CYS": ("SG",), "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
              "HIS": ("CG", "ND1", "CD2", "CE1", "NE2")}
ONE = {"TRP": "W", "TYR": "Y", "FY3": "Y", "MET": "M", "CYS": "C", "PHE": "F", "HIS": "H"}
CHI2_ATOM = {"TRP": "CD1", "TYR": "CD1", "FY3": "CD1", "PHE": "CD1", "HIS": "ND1", "MET": "SD"}
CHI2_GRID = {"TRP": range(0, 360, 30), "HIS": range(0, 360, 30), "TYR": range(0, 180, 15),
             "FY3": range(0, 180, 15), "PHE": range(0, 180, 15), "MET": (-60, 60, 180)}
BB = {"N", "CA", "C", "O", "OXT"}


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - b0 @ b1n * b1n, b2 - b2 @ b1n * b1n
    return np.degrees(np.arctan2(np.cross(b1n, v) @ w, v @ w))


def rotate(X, a, b, ang):
    k = (b - a) / np.linalg.norm(b - a)
    t = np.radians(ang)
    Y = X - a
    return a + Y * np.cos(t) + np.cross(k, Y) * np.sin(t) + np.outer(Y @ k, k) * (1 - np.cos(t))


def conformations(r):
    """List of dicts name -> xyz for the side chain beyond CB (crystal first)."""
    rn, at = r["name"], r["atoms"]
    names = [n for n in at if n not in BB and n != "CB"]
    base = {n: at[n].copy() for n in names}
    out = [base]
    if rn == "CYS" or "CB" not in at or "N" not in at:
        if rn == "CYS" and "CB" in at and "N" in at:
            g = "SG"
            c1 = dihedral(at["N"], at["CA"], at["CB"], at[g])
            for t1 in (-60, 60, 180):
                for o in (-15, 0, 15):
                    X = rotate(np.array([base[g]]), at["CA"], at["CB"], t1 + o - c1)
                    out.append({g: X[0]})
        return out
    g = "CG"
    c1 = dihedral(at["N"], at["CA"], at["CB"], at[g])
    for t1 in (-60, 60, 180):
        for o in (-15, 0, 15):
            X1 = rotate(np.array([base[n] for n in names]), at["CA"], at["CB"], t1 + o - c1)
            d1 = dict(zip(names, X1))
            d2a = CHI2_ATOM[rn]
            c2 = dihedral(at["CA"], at["CB"], d1["CG"], d1[d2a])
            beyond = [n for n in names if n not in ("CG",)]
            for t2 in CHI2_GRID[rn]:
                X2 = rotate(np.array([d1[n] for n in beyond]), at["CB"], d1["CG"], t2 - c2)
                d2 = dict(d1, **dict(zip(beyond, X2)))
                if rn == "MET":
                    c3 = dihedral(at["CB"], d2["CG"], d2["SD"], d2["CE"])
                    for t3 in (-60, 60, 180):
                        X3 = rotate(np.array([d2["CE"]]), d2["CG"], d2["SD"], t3 - c3)
                        out.append(dict(d2, CE=X3[0]))
                else:
                    out.append(d2)
    return out


def chis(r, conf):
    at = r["atoms"]
    full = dict(at, **conf)
    if r["name"] == "CYS":
        return f"{dihedral(at['N'], at['CA'], at['CB'], full['SG']):.0f}"
    c = [dihedral(at["N"], at["CA"], at["CB"], full["CG"]),
         dihedral(at["CA"], at["CB"], full["CG"], full[CHI2_ATOM[r["name"]]])]
    return "/".join(f"{x:.0f}" for x in c)


POLAR_CLASH = 2.5


def polar(name):
    return name[0] in "NO"


def clash_any(X, PX, Y, PY):
    """Heavy atoms closer than CLASH (POLAR_CLASH when both are N/O)."""
    d = np.sqrt(((X[:, None] - Y[None]) ** 2).sum(-1))
    lim = np.where(PX[:, None] & PY[None], POLAR_CLASH, CLASH)
    return bool((d < lim).any())


def build(res, kinds, src_xyz, src_key=None, disulfide=()):
    nodes = []
    for k, r in sorted(res.items()):
        if r["het"] or r["name"] not in kinds or k == src_key:
            continue
        if not all(a in r["atoms"] for a in NODE_ATOMS[r["name"]]):
            continue
        nodes.append(k)
    mobile = set(nodes) - set(disulfide)
    fx, fp, fo = [], [], []
    for k, r in res.items():
        for n, x in r["atoms"].items():
            if r["het"] or n in BB or n == "CB" or k not in mobile:
                fx.append(x)
                fp.append(polar(n))
                fo.append(k)
    fx, fp = np.array(fx + list(src_xyz)), np.array(fp + [True] * len(src_xyz))
    fo = fo + [None] * len(src_xyz)
    fo_arr = np.array([hash(o) for o in fo])
    states = []        # (node index, conformation dict, node atoms, all side-chain atoms, crystal?, polar mask)
    for i, k in enumerate(nodes):
        r = res[k]
        confs = conformations(r) if k in mobile else [{n: x for n, x in r["atoms"].items() if n not in BB}]
        near = (np.linalg.norm(fx - r["atoms"]["CA"], axis=1) < 16.0) & (fo_arr != hash(k))
        ex, ep = fx[near], fp[near]
        for c_i, c in enumerate(confs):
            names = list(c)
            X = np.array([c[n] for n in names])
            P = np.array([polar(n) for n in names])
            if c_i > 0 and len(ex) and clash_any(X, P, ex, ep):
                continue
            nodeX = np.array([c[n] for n in NODE_ATOMS[r["name"]]])
            states.append((i, c, nodeX, X, c_i == 0, P))
    return nodes, states


def search(res, nodes, states, src_xyz):
    n = len(states)
    owner = np.array([st[0] for st in states])
    by_node = {}
    for u, st in enumerate(states):
        by_node.setdefault(st[0], []).append(u)
    ca = {i: res[nodes[i]]["atoms"]["CA"] for i in by_node}
    blocks = {}                                   # (i, j) -> (hop distance matrix, usable mask)
    keys = sorted(by_node)
    for a_i, i in enumerate(keys):
        Ui = by_node[i]
        Ni = np.stack([states[u][2] for u in Ui])
        Ai = np.stack([states[u][3] for u in Ui]) if len({len(states[u][3]) for u in Ui}) == 1 else None
        for j in keys[a_i + 1:]:
            if np.linalg.norm(ca[i] - ca[j]) > 30.0:
                continue
            Uj = by_node[j]
            Nj = np.stack([states[u][2] for u in Uj])
            D = np.sqrt(((Ni[:, :, None, None] - Nj[None, None]) ** 2).sum(-1)).min(axis=(1, 3))
            ok = np.ones_like(D, bool)
            Aj = np.stack([states[u][3] for u in Uj]) if len({len(states[u][3]) for u in Uj}) == 1 else None
            if Ai is not None and Aj is not None:
                Pi, Pj = states[Ui[0]][5], states[Uj[0]][5]
                dd = np.sqrt(((Ai[:, :, None, None] - Aj[None, None]) ** 2).sum(-1))
                lim = np.where(Pi[:, None] & Pj[None], POLAR_CLASH, CLASH)[None, :, None, :]
                ok = ~(dd < lim).any(axis=(1, 3))
            cry_i = np.array([states[u][4] for u in Ui])
            cry_j = np.array([states[u][4] for u in Uj])
            ok |= cry_i[:, None] & cry_j[None]
            blocks[(i, j)] = (np.where(ok, D, np.inf), np.array(Ui), np.array(Uj))
    nbr = {}
    for (i, j), (D, Ui, Uj) in blocks.items():
        nbr.setdefault(i, []).append((Ui, Uj, D))
        nbr.setdefault(j, []).append((Uj, Ui, D.T))
    pos = {}
    for i, U in by_node.items():
        for r_, u in enumerate(U):
            pos[u] = r_
    b = np.array([np.sqrt(((st[2][:, None] - src_xyz[None]) ** 2).sum(-1)).min() for st in states])
    prev = np.full(n, -1)
    done = np.zeros(n, bool)
    heap = [(b[u], u) for u in range(n)]
    heapq.heapify(heap)
    while heap:
        bu, u = heapq.heappop(heap)
        if done[u] or bu > b[u]:
            continue
        done[u] = True
        on_path, w = set(), u
        while w >= 0:                      # a residue may sit on the path in one conformation only
            on_path.add(owner[w])
            w = prev[w]
        for Ui, Uj, D in nbr.get(owner[u], []):
            if owner[Uj[0]] in on_path:
                continue
            row = np.maximum(bu, D[pos[u]])
            better = row < b[Uj] - 1e-9
            for v, val in zip(Uj[better], row[better]):
                if not done[v]:
                    b[v], prev[v] = val, u
                    heapq.heappush(heap, (val, v))
    return b, prev


def describe(res, nodes, states, prev, v, src_xyz, src_label):
    path = [v]
    while prev[path[-1]] >= 0:
        path.append(prev[path[-1]])
    path = path[::-1]
    parts, last = [src_label], None
    for p in path:
        k = nodes[states[p][0]]
        r = res[k]
        moved = "" if states[p][4] else f" [chi {chis(r, states[p][1])}, turned]"
        ref = states[last][2] if last is not None else src_xyz
        hop = np.sqrt(((states[p][2][:, None] - ref[None]) ** 2).sum(-1)).min()
        parts.append(f"({hop:.1f}) {ONE[r['name']]}{k[1]}" + (f"/{k[0]}" if k[0] != "A" else "") + moved)
        last = p
    return " > ".join(parts)


SRC = [None]


def run(title, res, src_xyz, src_label, kinds, targets=(), src_key=None, out=None):
    disulfide = [k for k, r in res.items() if r["name"] == "CYS" and "SG" in r["atoms"] and any(
        k2 != k and r2["name"] == "CYS" and "SG" in r2["atoms"]
        and np.linalg.norm(r2["atoms"]["SG"] - r["atoms"]["SG"]) < 2.5 for k2, r2 in res.items())]
    nodes, states = build(res, kinds, src_xyz, src_key, disulfide)
    far = {st[0] for st in states if st[4]
           and np.sqrt(((st[2][:, None] - src_xyz[None]) ** 2).sum(-1)).min() >= FAR}
    lines = [f"\n## {title}  [{'/'.join(dict.fromkeys(ONE[k] for k in kinds))}: {len(nodes)} side chains, "
             f"{len(states)} conformations without clashes]"]
    for label, sts in (("crystal", [st for st in states if st[4]]), ("turned ", states)):
        b, prev = search(res, nodes, sts, src_xyz)
        farst = [i for i in range(len(sts)) if sts[i][0] in far]
        if farst:
            v = min(farst, key=lambda i: b[i])
            lines.append(f"  {label}: to a side chain >= {FAR:.0f} A away, longest hop {b[v]:.1f} A: "
                         + describe(res, nodes, sts, prev, v, src_xyz, src_label))
        for t in targets:
            ts = [i for i in range(len(sts)) if nodes[sts[i][0]] == t]
            if ts:
                v = min(ts, key=lambda i: b[i])
                lines.append(f"  {label}: to {ONE[res[t]['name']]}{t[1]}" + (f"/{t[0]}" if t[0] != "A" else "")
                             + f", longest hop {b[v]:.1f} A: " + describe(res, nodes, sts, prev, v, src_xyz,
                                                                         src_label))
    print("\n".join(lines), flush=True)
    if out is not None:
        out.write("\n".join(lines) + "\n")
        out.flush()


def main():
    d = Path(sys.argv[1])
    out = open(sys.argv[2], "w")
    WY = ("TRP", "TYR", "FY3")
    AR = ("TRP", "TYR", "FY3", "PHE", "HIS")
    ALL = AR + ("MET", "CYS")
    for kinds in (WY, AR, ALL):
        holo = s.parse(d / "1PB7.pdb")
        L = s.lig_xyz(holo, "GLY")
        prot = {k: r for k, r in holo.items() if not (r["het"] and r["name"] == "GLY")}
        run("GluN1 LBD 1PB7 + glycine", prot, L, "glycine", kinds, out=out)
        holo_p = {k: r for k, r in holo.items()}
        for oid, oname in (("1PBQ", "DCKA antagonist; glycine site placed"),):
            oth = {k: r for k, r in s.parse(d / f"{oid}.pdb").items() if k[0] == "A"}
            Lo, _, _ = s.place_ligand(holo_p, {k: r for k, r in oth.items() if not r["het"]}, L)
            run(f"GluN1 LBD {oid} ({oname})", oth, Lo, "glycine", kinds, out=out)
        phl = {k: r for k, r in s.parse(d / "1DNP.pdb").items() if k[0] == "A"}
        run("photolyase 1DNP, from FAD", phl, s.lig_xyz(phl, "FAD", "A"), "FAD", kinds,
            targets=(("A", 306, ""),), out=out)
        rnr = {k: r for k, r in s.parse(d / "6W4X.pdb").items() if k[0] in "BC"}
        sk = ("C", 122, "")
        src = np.array([rnr[sk]["atoms"][x] for x in s.SIDE["FY3"]])
        run("ribonucleotide reductase 6W4X (alpha B + beta C), from Y122/C", rnr, src, "Y122/C",
            kinds + (("CYS",) if "CYS" not in kinds else ()), targets=(("B", 439, ""), ("B", 731, "")), src_key=sk,
            out=out)
    out.close()


if __name__ == "__main__":
    main()
