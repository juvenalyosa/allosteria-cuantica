"""Trp/Tyr (and Met/Cys) hopping networks from a ligand or residue, in several proteins and states.

Usage: python -I survey_paths.py PDB_DIR OUT_JSON

Nodes are side chains (heavy atoms: Trp and Tyr rings, Tyr OH, Met SD/CE, Cys SG); the ligand or source
residue is one more node. An edge is the shortest heavy-atom distance between two nodes. For every node the
bottleneck b is the longest single hop of the path from the source that makes that longest hop as short as
possible (minimax path). Null model: the same number of randomly chosen side chains (any residue but Gly and
Ala, atoms beyond CB) instead of the Trp/Tyr set.
"""
import heapq
import json
import sys
from pathlib import Path

import numpy as np

SIDE = {"TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
        "FY3": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),      # 2,3,5-trifluorotyrosine (6W4X)
        "MET": ("SD", "CE"), "CYS": ("SG",)}
ONE = {"TRP": "W", "TYR": "Y", "FY3": "Y", "MET": "M", "CYS": "C"}
BACKBONE = {"N", "CA", "C", "O", "CB", "OXT"}
FAR = 20.0
THRESH = (5.0, 6.0, 7.0, 8.0)


def parse(path, chains=None, all_models=False, rename=None):
    res, model = {}, 1
    for ln in open(path):
        if ln.startswith("MODEL"):
            model = int(ln.split()[1])
            if model > 1 and not all_models:
                break
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name, rn, ch = ln[12:16].strip(), ln[17:20].strip(), ln[21]
        el = ln[76:78].strip() or name[0]
        if el in ("H", "D") or rn == "HOH" or ln[16] not in " A":
            continue
        ch = (rename or {}).get((model, ch), ch)
        if chains and ch not in chains:
            continue
        key = (ch, int(ln[22:26]), ln[26].strip())
        r = res.setdefault(key, {"name": rn, "het": ln.startswith("HETATM"), "atoms": {}})
        r["atoms"].setdefault(name, np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    return res


def label(key, rn):
    return f"{ONE.get(rn, rn)}{key[1]}{key[2]}" + (f"/{key[0]}" if key[0] not in "A" else "")


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def place_ligand(holo, other, lig_xyz):
    """Ligand of holo moved onto `other` by the backbone of the residues within 6 A of it."""
    P, Q = [], []
    for k, r in holo.items():
        if r["het"] or k not in other or other[k]["name"] != r["name"]:
            continue
        X = np.array(list(r["atoms"].values()))
        if np.linalg.norm(X[:, None] - lig_xyz[None], axis=2).min() > 6.0:
            continue
        for a in ("N", "CA", "C"):
            if a in r["atoms"] and a in other[k]["atoms"]:
                P.append(r["atoms"][a])
                Q.append(other[k]["atoms"][a])
    P, Q = np.array(P), np.array(Q)
    R, t = kabsch(P, Q)
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    return lig_xyz @ R.T + t, len(P) // 3, rmsd


def nodes_of(res, kinds, extra=()):
    out = []
    for k, r in sorted(res.items()):
        if r["name"] in kinds or k in extra:
            atoms = SIDE.get(r["name"], ())
            X = [r["atoms"][a] for a in atoms if a in r["atoms"]]
            if X:
                out.append((label(k, r["name"]), k, np.array(X)))
    return out


def minimax(D, src):
    n = len(D)
    b, s, prev = np.full(n, np.inf), np.full(n, np.inf), [-1] * n
    b[src], s[src] = 0.0, 0.0
    heap, done = [(0.0, 0.0, src)], np.zeros(n, bool)
    while heap:
        bu, su, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v in range(n):
            if v == u or done[v]:
                continue
            nb, ns = max(bu, D[u, v]), su + D[u, v]
            if nb < b[v] - 1e-9 or (abs(nb - b[v]) < 1e-9 and ns < s[v]):
                b[v], s[v], prev[v] = nb, ns, u
                heapq.heappush(heap, (nb, ns, v))
    return b, prev


def path_to(prev, v):
    p = [v]
    while prev[p[-1]] >= 0:
        p.append(prev[p[-1]])
    return p[::-1]


def analyse(nodes, src_xyz, src_label):
    """nodes: list of (label, key, xyz); the source becomes node 0."""
    allx = [src_xyz] + [x for _, _, x in nodes]
    labels = [src_label] + [l for l, _, _ in nodes]
    n = len(allx)
    X = np.concatenate(allx)
    starts = np.cumsum([0] + [len(x) for x in allx[:-1]])
    M = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    D = np.minimum.reduceat(np.minimum.reduceat(M, starts, axis=0), starts, axis=1)
    np.fill_diagonal(D, 0.0)
    b, prev = minimax(D, 0)
    dist = D[0]
    far = [i for i in range(1, n) if dist[i] >= FAR]
    best = min(far, key=lambda i: (b[i], dist[i])) if far else None
    reach = {}
    for T in THRESH:
        ok = [i for i in range(1, n) if b[i] <= T]
        reach[T] = (len(ok), float(max((dist[i] for i in ok), default=0.0)))
    return {"labels": labels, "D": D, "b": b, "prev": prev, "dist": dist, "far": far, "best": best,
            "reach": reach}


def fmt_path(a, v):
    p = path_to(a["prev"], v)
    return " > ".join(a["labels"][p[0]] if i == 0 else f"({a['D'][p[i - 1], p[i]]:.1f}) {a['labels'][p[i]]}"
                      for i in range(len(p)))


def null_far_bottleneck(res, n_nodes, src_xyz, draws=200, seed=7):
    pool = []
    for k, r in res.items():
        if r["het"] or r["name"] in ("GLY", "ALA"):
            continue
        X = [x for a, x in r["atoms"].items() if a not in BACKBONE]
        if X:
            pool.append((label(k, r["name"]), k, np.array(X)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        pick = [pool[i] for i in rng.choice(len(pool), size=min(n_nodes, len(pool)), replace=False)]
        a = analyse(pick, src_xyz, "src")
        vals.append(a["b"][a["best"]] if a["best"] is not None else np.inf)
    return np.array(vals)


def report(title, res, src_xyz, src_label, kinds=("TRP", "TYR", "FY3"), targets=(), null=True, extra=()):
    nodes = nodes_of(res, kinds, extra)
    a = analyse(nodes, src_xyz, src_label)
    prot = np.array([x for r in res.values() if not r["het"] for x in r["atoms"].values()])
    span = np.linalg.norm(prot[:, None] - src_xyz[None], axis=2).min(1).max()
    out = {"title": title, "kinds": list(kinds), "nodes": len(nodes), "span": float(span),
           "reach": {str(k): v for k, v in a["reach"].items()}}
    print(f"\n## {title}  [{'/'.join(ONE[k] for k in kinds if k != 'FY3')}: {len(nodes)} side chains; "
          f"farthest protein atom {span:.1f} A from {src_label}]")
    print("  reach from the source with every hop <= T: " + "; ".join(
        f"T={T:.0f}: {c} side chains, out to {d:.1f} A" for T, (c, d) in a["reach"].items()))
    if a["best"] is not None:
        bf = a["b"][a["best"]]
        out["far_bottleneck"] = float(bf)
        out["far_path"] = fmt_path(a, a["best"])
        print(f"  shortest longest-hop to any side chain >= {FAR:.0f} A away: {bf:.1f} A, to "
              f"{a['labels'][a['best']]} ({a['dist'][a['best']]:.1f} A): {fmt_path(a, a['best'])}")
        if null:
            nv = null_far_bottleneck(res, len(nodes), src_xyz)
            out["null_median"], out["null_frac_le"] = float(np.median(nv)), float((nv <= bf + 1e-9).mean())
            print(f"  random side chains (same number, 200 draws): median {np.median(nv):.1f} A "
                  f"(5-95 %: {np.percentile(nv, 5):.1f}-{np.percentile(nv, 95):.1f}); "
                  f"draws as good or better: {100 * (nv <= bf + 1e-9).mean():.0f} %")
    for tl in targets:
        if tl in a["labels"]:
            i = a["labels"].index(tl)
            out.setdefault("targets", {})[tl] = {"b": float(a["b"][i]), "dist": float(a["dist"][i]),
                                                  "path": fmt_path(a, i)}
            print(f"  to {tl} ({a['dist'][i]:.1f} A): longest hop {a['b'][i]:.1f} A: {fmt_path(a, i)}")
    return out, a


def compare(name_a, a, name_b, b):
    """Side chains whose bottleneck from the source changes, and pair distances that change most."""
    common = [l for l in a["labels"][1:] if l in b["labels"]]
    ch = []
    for l in common:
        i, j = a["labels"].index(l), b["labels"].index(l)
        if abs(a["b"][i] - b["b"][j]) >= 1.0:
            ch.append((b["b"][j] - a["b"][i], l, a["b"][i], b["b"][j], a["dist"][i], b["dist"][j]))
    print(f"  {name_a} -> {name_b}: {len(common)} side chains in both; bottleneck changes >= 1 A: {len(ch)}")
    for d, l, x, y, dx, dy in sorted(ch, key=lambda t: t[0])[:8]:
        print(f"    {l}: {x:.1f} -> {y:.1f} A (distance from source {dx:.1f} -> {dy:.1f})")
    pairs = []
    for p in range(len(common)):
        for q in range(p + 1, len(common)):
            ia, ja = a["labels"].index(common[p]), a["labels"].index(common[q])
            ib, jb = b["labels"].index(common[p]), b["labels"].index(common[q])
            da, db = a["D"][ia, ja], b["D"][ib, jb]
            if min(da, db) <= 8.0 and abs(da - db) >= 1.5:
                pairs.append((db - da, common[p], common[q], da, db))
    if pairs:
        print("    pair distances that change by >= 1.5 A (one of them <= 8 A): " + "; ".join(
            f"{p}-{q} {x:.1f}->{y:.1f}" for _, p, q, x, y in sorted(pairs, key=lambda t: abs(t[0]), reverse=True)[:8]))
    return ch


def lig_xyz(res, resname, chain=None):
    X = [x for k, r in res.items() if r["het"] and r["name"] == resname and (chain is None or k[0] == chain)
         for x in r["atoms"].values()]
    return np.array(X)


def main():
    d, out = Path(sys.argv[1]), {}
    W = ("TRP", "TYR")
    WM = ("TRP", "TYR", "MET", "CYS")

    print("# Allosteric pairs: source = the agonist / substrate (moved onto the other state by the backbone of the "
          "binding site)")
    pairs = [("GluN1 LBD", "1PB7", "glycine", "GLY", None, [("1PB8", "D-serine"), ("1PBQ", "DCKA, antagonist")]),
             ("GluA2 LBD", "1FTJ", "glutamate", "GLU", "A", [("1FTO", "apo")]),
             ("maltose-binding protein", "1ANF", "maltose", "GLC", None, [("1OMP", "apo")]),
             ("adenylate kinase", "1AKE", "Ap5A", "AP5", "A", [("4AKE", "apo")])]
    for prot, hid, hname, lres, lch, others in pairs:
        holo = parse(d / f"{hid}.pdb", chains=None)
        L = lig_xyz(holo, lres, lch)
        holo_p = {k: r for k, r in holo.items() if k[0] == "A" or (r["het"] and r["name"] == lres)}
        for kinds in (W, WM):
            o, a = report(f"{prot}, {hid} ({hname})", {k: r for k, r in holo_p.items() if not r["het"]}, L, hname,
                          kinds=kinds, null=(kinds == W))
            out[f"{hid}:{'+'.join(kinds)}"] = o
            for oid, oname in others:
                oth = {k: r for k, r in parse(d / f"{oid}.pdb").items() if k[0] == "A" and not r["het"]}
                Lo, nres, rmsd = place_ligand(holo_p, oth, L)
                o2, a2 = report(f"{prot}, {oid} ({oname}); {hname} placed by {nres} site residues, "
                                f"rmsd {rmsd:.2f} A", oth, Lo, hname, kinds=kinds, null=(kinds == W))
                out[f"{oid}:{'+'.join(kinds)}"] = o2
                compare(hid, a, oid, a2)

    print("\n# Hemoglobin: source = haem of chain A (alpha 1); targets = the other haems")
    states = {"2DN1 (oxy, R)": parse(d / "2DN1.pdb1", all_models=True, rename={(2, "A"): "C", (2, "B"): "D"}),
              "2DN2 (deoxy, T)": parse(d / "2DN2.pdb1", all_models=True)}
    fe = {}
    for name, res in states.items():
        fe[name] = {c: next(r["atoms"]["FE"] for k, r in res.items() if r["name"] == "HEM" and k[0] == c)
                    for c in "ABCD"}
        print(f"  {name}: Fe-Fe from A: " + ", ".join(f"{c} {np.linalg.norm(fe[name][c] - fe[name]['A']):.1f}"
                                                     for c in "BCD"))
    hb = {}
    for name, res in states.items():
        prot = {k: r for k, r in res.items() if not r["het"]}
        for c in "BCD":
            hk = next(k for k, r in res.items() if r["name"] == "HEM" and k[0] == c)
            prot[hk] = dict(res[hk], het=False)
        SIDE["HEM"] = tuple(a for a in res[next(k for k, r in res.items() if r["name"] == "HEM" and k[0] == "B")]
                            ["atoms"])
        ONE["HEM"] = "haem"
        tg = [label(k, "HEM") for k, r in prot.items() if r["name"] == "HEM"]
        for kinds in (W, WM):
            o, a = report(f"hemoglobin, {name}", prot, lig_xyz(res, "HEM", "A"), "haem A",
                          kinds=kinds + ("HEM",), targets=tg, null=False)
            out[f"Hb {name}:{'+'.join(kinds)}"] = o
            hb[(name, kinds)] = a
    for kinds in (W, WM):
        compare("R", hb[("2DN1 (oxy, R)", kinds)], "T", hb[("2DN2 (deoxy, T)", kinds)])

    print("\n# Known hopping chains (positive controls)")
    phl = {k: r for k, r in parse(d / "1DNP.pdb").items() if k[0] == "A"}
    for kinds in (W, WM):
        o, a = report("E. coli DNA photolyase 1DNP: from the FAD", {k: r for k, r in phl.items() if not r["het"]},
                      lig_xyz(phl, "FAD", "A"), "FAD", kinds=kinds, targets=("W382", "W359", "W306"),
                      null=(kinds == W))
        out[f"1DNP:{'+'.join(kinds)}"] = o
    rnr = parse(d / "6W4X.pdb")
    src_key = ("C", 122, "")
    src = np.array([rnr[src_key]["atoms"][x] for x in SIDE["FY3"]])
    prot = {k: r for k, r in rnr.items() if (not r["het"] or r["name"] == "FY3") and k != src_key}
    for kinds in (W, WM):
        o, a = report("E. coli ribonucleotide reductase 6W4X: from Y122 (trifluoro-Tyr) of beta chain C",
                      {k: dict(r, het=False) for k, r in prot.items()}, src, "Y122/C",
                      kinds=kinds + ("FY3",), targets=("W48/C", "Y356/C", "Y731", "Y730", "C439", "Y731/B",
                                                       "Y730/B", "C439/B"),
                      null=(kinds == W), extra=(("A", 439, ""), ("B", 439, "")))
        out[f"6W4X:{'+'.join(kinds)}"] = o
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
