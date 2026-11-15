"""Writes the Colab notebook of test 1 (range of the glycine's electronic effect on GluN1, frozen geometry).
    python make_allo_t1_nb.py OUT.ipynb COMMIT SCRIPTS_DIR ATOMS_CSV"""
import base64
import gzip
import json
import sys
from pathlib import Path

out, commit, sdir, atoms_csv = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


def embed(path, dest):
    return f"Path('{dest}').write_text({path.read_text()!r})\n"


blob = base64.b64encode(gzip.compress(atoms_csv.read_bytes())).decode()

md("""
# Prueba 1: ¿hasta dónde llega el efecto electrónico de la glicina en GluN1?

Geometría congelada (1PB7 con hidrógenos, preparada en local). PM7, MOZYME, `1SCF GRADIENTS`.

| Sistema | Contenido |
|---|---|
| A | proteína + glicina |
| B | proteína sola |
| C | proteína + la glicina como cargas puntuales (sus cargas ESP de PM7, `MM_CHARGES`) |
| D | glicina sola (cargas ESP) |

A − B es todo el efecto de la glicina sobre las cargas y fuerzas de la proteína; C − B, lo que da un campo clásico; A − C,
lo que la glicina cuántica añade. Variantes: dos corridas iguales (ruido), CPU (`MOPAC_NOGPU=1`), `CUTOFF=15` y `25`
(MOZYME trata las interacciones más allá de ~10 Å como cargas y dipolos), agua implícita (`EPS=78.4`) y SCF convencional
sin orbitales localizados.

| Bloque | Tiempo (A100, estimado) |
|---|---|
| Compilar MOPAC | ~10 min |
| GPU: base, repeticiones, cutoffs, agua | ~10–20 min |
| CPU: A, B, C en paralelo | ~15–30 min (corre a la vez que la GPU) |
| SCF convencional, 4 614 átomos | hasta 45 min por corrida (límite) |

**Antes de empezar:** *Entorno de ejecución → Cambiar tipo de entorno → GPU A100*, y luego *Ejecutar todo*. Los resultados
se guardan en Google Drive (`MyDrive/mopac_paper_bench/RESULTS_NAME`); si la sesión se corta, vuelva a ejecutar todo y lo
terminado se salta. La última celda descarga un zip.
""")

md("## 1. Configuración")
code(f"""
COMMIT = '{commit}'
RESULTS_NAME = 'allo_t1_a100'
USE_DRIVE = True
REQUIRE_GPU = 'A100'           # None: aceptar cualquier GPU
CONV_TIMEOUT = 2700            # s por corrida del SCF convencional
""")
md("## 2. GPU y carpeta de resultados")
code("""
import os, re, subprocess, sys, time, shutil, json, gzip, base64, threading
from pathlib import Path

def run(cmd, check=True, env=None):
    print('$', cmd, flush=True)
    full = dict(os.environ); full.update(env or {})
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=full)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f'exit {p.returncode}: {cmd}')
    return p.returncode

run('nvidia-smi')
gpu = subprocess.run('nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader', shell=True,
                     capture_output=True, text=True).stdout.strip()
NCPU = os.cpu_count()
CC = gpu.split(',')[-1].strip().replace('.', '')
if REQUIRE_GPU and REQUIRE_GPU not in gpu:
    raise RuntimeError(f'Esta sesión tiene {gpu}, no una {REQUIRE_GPU}.')
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/mopac_paper_bench')
else:
    BASE = Path('/content/mopac_paper_bench')
OUT = BASE / RESULTS_NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'env.json').write_text(json.dumps({'gpu': gpu, 'ncpu': NCPU, 'commit': COMMIT}, indent=1))
print('resultados en', OUT)
""")
md("## 3. Compilar MOPAC\n\n~10 minutos.")
code("""
SRC, BUILD = Path('/content/mopac_gpu'), Path('/content/build')
MOPAC = BUILD / 'mopac'
if not MOPAC.exists():
    run('apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gfortran ninja-build '
        'libopenblas-dev liblapack-dev > /dev/null')
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']
    shutil.rmtree(SRC, ignore_errors=True); shutil.rmtree(BUILD, ignore_errors=True)
    run(f'git clone -q https://github.com/juvenalyosa/mopac_gpu.git {SRC}')
    run(f'git -C {SRC} checkout -q {COMMIT}')
    head = subprocess.run(f'git -C {SRC} rev-parse HEAD', shell=True, capture_output=True, text=True).stdout.strip()
    if not head.startswith(COMMIT):
        raise SystemExit(f'HEAD {head} no es COMMIT {COMMIT}')
    stamp = OUT / 'commit.txt'
    if stamp.exists() and not head.startswith(stamp.read_text().strip()[:8]):
        raise SystemExit(f'{OUT} tiene resultados de {stamp.read_text().strip()}; use otro RESULTS_NAME')
    stamp.write_text(head + '\\n')
    t0 = time.time()
    run(f'cmake -S {SRC} -B {BUILD} -G Ninja -DGPU=ON -DTESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS={CC} '
        f'-DENABLE_MKL=OFF -DBLA_VENDOR=OpenBLAS > /content/cmake.log 2>&1 || (tail -40 /content/cmake.log; exit 1)')
    run(f'cmake --build {BUILD} --target mopac --parallel {NCPU} > /content/build.log 2>&1 '
        f'|| (grep -i -B2 -A6 error /content/build.log | head -80; exit 1)')
    print(f'compilado en {time.time() - t0:.0f} s')
run(f'ls -la {MOPAC}')
""")
md("## 4. Entradas (estructura preparada en local y scripts)")
code(embed(sdir / "t1_tools.py", "/content/t1_tools.py") + embed(sdir / "run_t1.py", "/content/run_t1.py") + f"""
Path('/content/1PB7_atoms.csv').write_bytes(gzip.decompress(base64.b64decode({blob!r})))
WD = OUT / 'runs'
if not (WD / 'meta.json').exists():
    run(f'{{sys.executable}} /content/t1_tools.py decks /content/1PB7_atoms.csv {{WD}} 2')
run(f'{{sys.executable}} /content/run_t1.py {{MOPAC}} {{WD}} D')
run(f'{{sys.executable}} /content/t1_tools.py mmcharges {{WD}}')
""")
md("""
## 5. Corridas

Las tres corridas en CPU arrancan en segundo plano (una por núcleo) y las de GPU van una tras otra.
""")
code("""
cpu_log = OUT / 'cpu_jobs.log'
cpu = subprocess.Popen(f'{sys.executable} /content/run_t1.py {MOPAC} {WD} A_cpu,B_cpu,C_cpu 3 > {cpu_log} 2>&1',
                       shell=True, executable='/bin/bash')
gpu_jobs = ['A_vac', 'B_vac', 'C_vac', 'A_r2', 'B_r2', 'C_r2', 'A_c15', 'B_c15', 'C_c15', 'A_c25', 'B_c25', 'C_c25',
            'A_eps', 'B_eps', 'C_eps']
run(f'{sys.executable} /content/run_t1.py {MOPAC} {WD} {",".join(gpu_jobs)} 1')
run(f'{sys.executable} /content/t1_tools.py analyse {WD} {OUT / "analysis"}')
""")
md("""
## 6. SCF convencional (sin orbitales localizados)

Control de MOZYME: el mismo A − B con la matriz completa. Hasta `CONV_TIMEOUT` segundos por corrida; si no termina, el
resto de los resultados vale igual.
""")
code("""
run(f'{sys.executable} /content/run_t1.py {MOPAC} {WD} A_conv,B_conv,C_conv 1 {CONV_TIMEOUT}', check=False)
print('esperando las corridas en CPU...', flush=True)
cpu.wait()
run(f'cat {cpu_log}', check=False)
run(f'{sys.executable} /content/t1_tools.py analyse {WD} {OUT / "analysis"}')
""")
md("## 7. Figura y descarga")
code("""
from IPython.display import Image, display
if (OUT / 'analysis' / 'decay.png').exists():
    display(Image(str(OUT / 'analysis' / 'decay.png')))
zipbase = f'/content/{RESULTS_NAME}'
shutil.make_archive(zipbase, 'zip', OUT.parent, OUT.name)
print(f'{zipbase}.zip', os.path.getsize(f'{zipbase}.zip') // 1024, 'kB')
try:
    from google.colab import files
    files.download(f'{zipbase}.zip')
except Exception as e:
    print('descárguelo a mano:', e)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
Path(out).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(json.dumps(nb)) // 1024, "kB")
