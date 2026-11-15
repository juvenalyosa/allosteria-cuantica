"""Glycine contacts and Trp/Tyr/Met/Cys side-chain networks in a PDB file (heavy atoms only)."""
import itertools
import sys

import numpy as np

SC = {"TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
      "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
      "MET": {"SD", "CE"}, "CYS": {"SG"}}

ONE = {"TRP": "W", "TYR": "Y", "MET": "M", "CYS": "C"}
prot, lig = [], []
for ln in open(sys.argv[1]):
    if ln.startswith(("ATOM", "HETATM")) and ln[16] in " A":
        rec = (ln[17:20], int(ln[22:26]), ln[12:16].strip(),
               np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
        if ln.startswith("HETATM") and rec[0] == "GLY":
            lig.append(rec)
        elif ln.startswith("ATOM"):
            prot.append(rec)
L = np.array([r[3] for r in lig])
P = np.array([r[3] for r in prot])
dmin = np.linalg.norm(P[:, None, :] - L[None, :, :], axis=2).min(axis=1)
print(f"protein heavy atoms {len(prot)}, glycine atoms {len(lig)}; farthest protein atom from glycine "
      f"{dmin.max():.1f} A")
print("contacts within 4.0 A of glycine:")
for (res, num), grp in itertools.groupby(sorted(zip([(r[0], r[1]) for r in prot], [r[2] for r in prot], dmin),
                                                key=lambda t: t[0][1]), key=lambda t: t[0]):
    g = [(a, d) for _, a, d in grp if d <= 4.0]
    if g:
        print(f"  {res}{num}: " + ", ".join(f"{a} {d:.2f}" for a, d in sorted(g, key=lambda t: t[1])))

groups = {}
for res, num, name, x in prot:
    if res in SC and name in SC[res]:
        groups.setdefault((res, num), []).append(x)
keys = sorted(groups, key=lambda k: k[1])
cys_ss = {28, 62, 44, 63, 236, 290}
print(f"\nside chains: " + ", ".join(f"{r} {sum(k[0] == r for k in keys)}" for r in SC)
      + f" (Cys in disulfides: {sorted(cys_ss)})")
X = {k: np.array(v) for k, v in groups.items()}
dg = {k: np.linalg.norm(X[k][:, None, :] - L[None, :, :], axis=2).min() for k in keys}
print("nearest to glycine: " + ", ".join(f"{k[0]}{k[1]} {dg[k]:.1f}" for k in sorted(keys, key=dg.get)[:8]))
E = {}
for a, b in itertools.combinations(keys, 2):
    E[(a, b)] = np.linalg.norm(X[a][:, None, :] - X[b][None, :, :], axis=2).min()
for cut in (4.5, 6.0, 8.0):
    adj = {k: set() for k in keys}
    for (a, b), d in E.items():
        if d <= cut:
            adj[a].add(b)
            adj[b].add(a)
    seen, comps = set(), []
    for k in keys:
        if k in seen:
            continue
        stack, comp = [k], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u] - seen)
        if len(comp) > 1:
            comps.append(sorted(comp, key=lambda t: t[1]))
    comps.sort(key=len, reverse=True)
    print(f"\nedge-to-edge <= {cut} A: {len(comps)} chains of 2 or more")
    for c in comps[:6]:
        span = max(dg[k] for k in c)
        print(f"  {len(c):2d}: " + " ".join(f"{ONE[k[0]]}{k[1]}" for k in c)
              + f"  (distance to glycine {min(dg[k] for k in c):.1f}-{span:.1f} A)")
for k in keys:
    if dg[k] < 9.0:
        nb = sorted(((E.get((k, j), E.get((j, k))), j) for j in keys if j != k))[:3]
        print(f"{ONE[k[0]]}{k[1]} (glycine {dg[k]:.1f} A) nearest: " + ", ".join(f"{ONE[j[0]]}{j[1]} {d:.1f}" for d, j in nb))
