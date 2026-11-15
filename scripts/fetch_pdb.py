"""Download the PDB entries the scripts read: python fetch_pdb.py OUT_DIR

Asymmetric units, plus the biological assemblies of the two hemoglobin entries (2DN1.pdb1, 2DN2.pdb1).
"""
import sys
import urllib.request
from pathlib import Path

ENTRIES = ["1PB7", "1PB8", "1PBQ", "1FTO", "1FTJ", "1OMP", "1ANF", "4AKE", "1AKE", "2DN1", "2DN2", "1DNP", "6W4X",
           "2WUR"]
ASSEMBLIES = ["2DN1", "2DN2"]

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name in [f"{e}.pdb" for e in ENTRIES] + [f"{e}.pdb1" for e in ASSEMBLIES]:
    dest = out / name
    if not dest.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{name}", dest)
    print(name, dest.stat().st_size)
