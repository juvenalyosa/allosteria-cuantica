"""Proton wires: chains of hydrogen bonds between groups that can take and give a proton.

Usage: python -I hbond_paths.py PDB_DIR OUT_TXT

Nodes: the polar atoms of Asp, Glu, Lys, Arg, His, Tyr, Ser, Thr side chains, the chain termini, crystal waters
and the ligand (or chromophore). An edge joins two nodes whose N/O atoms are closer than the cut-off (3.2 A;
3.5 A as a looser bound). The wire is counted in hops (one proton handed on per hop). A second network adds
the groups that only pass hydrogen bonds on (backbone N-H / C=O, Asn, Gln, Trp NE1), which cannot hand a proton
on but can hold the chain in place.
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np

WIRE = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"),
        "HIS": ("ND1", "NE2"), "TYR": ("OH",), "SER": ("OG",), "THR": ("OG1",), "HOH": ("O",)}
RELAY = {"ASN": ("OD1", "ND2"), "GLN": ("OE1", "NE2"), "TRP": ("NE1",)}
ACIDBASE = {"ASP", "GLU", "LYS", "ARG", "HIS", "TYR"}
ONE = {"ASP": "D", "GLU": "E", "LYS": "K", "ARG": "R", "HIS": "H", "TYR": "Y", "SER": "S", "THR": "T",
       "HOH": "w", "ASN": "N", "GLN": "Q", "TRP": "W"}


def parse(path, chains="A"):
    res = {}
    for ln in open(path):
        if ln.startswith("ENDMDL"):
            break
        if not ln.startswith(("ATOM", "HETATM")) or ln[16] not in " A":
            continue
        name, rn, ch = ln[12:16].strip(), ln[17:20].strip(), ln[21]
        el = ln[76:78].strip() or name[0]
        if el in ("H", "D") or (ch not in chains and rn != "HOH"):
            continue
        key = (ch, int(ln[22:26]), ln[26].strip(), rn)
        r = res.setdefault(key, {"name": rn, "het": ln.startswith("HETATM"), "atoms": {}})
        r["atoms"].setdefault(name, np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    return res


def nodes_of(res, ligand, src_atoms, relay, water="shell"):
    nodes = []
    if water != "all":
        solute = np.array([x for r in res.values() if r["name"] != "HOH" for x in r["atoms"].values()])
        keep = set()
        for k, r in res.items():
            if r["name"] == "HOH" and water == "shell":
                if np.linalg.norm(solute - r["atoms"]["O"], axis=1).min() <= 3.5:
                    keep.add(k)
        res = {k: r for k, r in res.items() if r["name"] != "HOH" or k in keep}
    prot = sorted(k for k, r in res.items() if not r["het"])
    first, last = {}, {}
    for k in prot:
        first.setdefault(k[0], k)
        last[k[0]] = k
    for k, r in sorted(res.items()):
        rn = r["name"]
        if rn == ligand:
            names = [n for n in r["atoms"] if n[0] in "NO" and (not src_atoms or n in src_atoms)]
            kind = "src"
        elif rn in WIRE:
            names = [n for n in WIRE[rn] if n in r["atoms"]]
            kind = "wire"
        else:
            names, kind = [], "wire"
        if not r["het"]:
            if k == first.get(k[0]):
                names = names + ["N"]
            if k == last.get(k[0]) and "OXT" in r["atoms"]:
                names = names + ["O", "OXT"]
        if names:
            nodes.append((k, kind, np.array([r["atoms"][n] for n in names])))
        if relay and not r["het"]:
            rl = [n for n in RELAY.get(rn, ()) if n in r["atoms"]]
            if rl:
                nodes.append((k + ("sc",), "relay", np.array([r["atoms"][n] for n in rl])))
            bb = [n for n in ("N", "O") if n in r["atoms"]]
            if bb:
                nodes.append((k + ("bb",), "relay", np.array([r["atoms"][n] for n in bb])))
    return nodes


def lab(k):
    s = "" if len(k) == 4 else ("(bb)" if k[4] == "bb" else "(sc)")
    return f"{ONE.get(k[3], k[3])}{k[1]}{s}" if k[3] != "HOH" else "w"


def analyse(nodes, cut):
    X = np.concatenate([x for _, _, x in nodes])
    owner = np.concatenate([[i] * len(x) for i, (_, _, x) in enumerate(nodes)])
    n = len(nodes)
    adj = [set() for _ in range(n)]
    for a in range(len(X)):
        d = np.linalg.norm(X - X[a], axis=1)
        for bidx in np.where((d <= cut) & (owner != owner[a]))[0]:
            adj[owner[a]].add(owner[bidx])
    srcs = [i for i, (_, kind, _) in enumerate(nodes) if kind == "src"]
    S = np.concatenate([nodes[i][2] for i in srcs])
    dist = np.array([np.linalg.norm(x[:, None] - S[None], axis=2).min() for _, _, x in nodes])
    hops, prev = np.full(n, -1), np.full(n, -1)
    q = deque(srcs)
    for i in srcs:
        hops[i] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if hops[v] < 0:
                hops[v], prev[v] = hops[u] + 1, u
                q.append(v)
    return adj, dist, hops, prev


def path(prev, v):
    p = [v]
    while prev[p[-1]] >= 0:
        p.append(prev[p[-1]])
    return p[::-1]


def run(title, res, ligand, cut_list=(3.2, 3.5), src_atoms=(), target=None, out=None, far=20.0):
    lines = [f"\n## {title}"]
    for relay, water in ((False, "none"), (False, "shell"), (True, "shell")):
        nodes = nodes_of(res, ligand, src_atoms, relay, water)
        for cut in cut_list:
            adj, dist, hops, prev = analyse(nodes, cut)
            reach = [i for i in range(len(nodes)) if hops[i] > 0]
            ab = sorted({nodes[i][0][:4] for i in reach if nodes[i][0][3] in ACIDBASE and len(nodes[i][0]) == 4})
            tot_ab = sorted({k[:4] for k, kind, _ in nodes if k[3] in ACIDBASE and len(k) == 4})
            maxd = max((dist[i] for i in reach), default=0.0)
            tag = ("proton wire, no water" if water == "none" else "proton wire, waters touching the protein") \
                if not relay else "+ backbone/amide H-bonds"
            lines.append(f"  {tag}, H-bond <= {cut} A: {len(reach)} groups reached "
                         f"({sum(nodes[i][0][3] == 'HOH' for i in reach)} waters), acid/base residues "
                         f"{len(ab)} of {len(tot_ab)}, out to {maxd:.1f} A"
                         + (f": {' '.join(ONE[k[3]] + str(k[1]) for k in ab)}" if not relay else ""))
            farn = [i for i in reach if dist[i] >= far]
            if farn:
                v = min(farn, key=lambda i: (hops[i], -dist[i]))
                p = path(prev, v)
                lines.append(f"    fewest hops to >= {far:.0f} A: {hops[v]} hops, to {lab(nodes[v][0])} "
                             f"({dist[v]:.1f} A): " + " > ".join(lab(nodes[i][0]) if nodes[i][1] != "src" else ligand
                                                                for i in p))
            if target is not None:
                t = [i for i in range(len(nodes)) if nodes[i][0][:4] == target and len(nodes[i][0]) == 4]
                if t and hops[t[0]] > 0:
                    p = path(prev, t[0])
                    lines.append(f"    to {lab(nodes[t[0]][0])}: {hops[t[0]]} hops: "
                                 + " > ".join(lab(nodes[i][0]) if nodes[i][1] != "src" else ligand for i in p))
                elif t:
                    lines.append(f"    to {lab(nodes[t[0]][0])}: not reached")
    print("\n".join(lines), flush=True)
    if out is not None:
        out.write("\n".join(lines) + "\n")


def main():
    d = Path(sys.argv[1])
    out = open(sys.argv[2], "w")
    run("GFP 2WUR (positive control): from the chromophore phenol O (OH), target Glu222",
        parse(d / "2WUR.pdb"), "GYS", src_atoms=("OH",), target=("A", 222, "", "GLU"), out=out)
    for pid, lig, name in (("1PB7", "GLY", "glycine"), ("1PB8", "DSN", "D-serine"), ("1PBQ", "DK1", "DCKA")):
        res = parse(d / f"{pid}.pdb")
        res = {k: r for k, r in res.items() if not (r["name"] == lig and not r["het"])}
        if lig == "GLY":
            res = {(k[0], k[1], k[2], "LIG" if r["het"] and r["name"] == "GLY" else k[3]):
                   dict(r, name="LIG" if r["het"] and r["name"] == "GLY" else r["name"]) for k, r in res.items()}
            lig = "LIG"
        run(f"GluN1 LBD {pid}: from {name}", res, lig, out=out)
    for pid, lig, name in (("1FTJ", "GLU", "glutamate"),):
        res = parse(d / f"{pid}.pdb")
        res = {(k[0], k[1], k[2], "LIG" if r["het"] and r["name"] == lig else k[3]):
               dict(r, name="LIG" if r["het"] and r["name"] == lig else r["name"]) for k, r in res.items()}
        run(f"GluA2 LBD {pid}: from {name}", res, "LIG", out=out)
    out.close()


if __name__ == "__main__":
    main()
