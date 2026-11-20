"""Writes the Colab notebook of test 1c: why CUTOFF=15/25 failed the SCF in test 1, and the CUTOFF question at a
looser SCF criterion.
    python make_allo_t1c_nb.py OUT.ipynb COMMIT SCRIPTS_DIR ATOMS_CSV"""
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


blob = base64.b64encode(gzip.compress(atoms_csv.read_bytes(), mtime=0)).decode()

md("""
# Prueba 1c: ¿por qué falla el SCF con `CUTOFF=15` y `25`, y qué pasa con un criterio más suelto?

En la prueba 1 (L4, `allo_t1b_l4`) las seis corridas con `CUTOFF=15` o `CUTOFF=25` terminaron con *THE SCF CALCULATION
FAILED* (300 iteraciones, `SCFCRT=2.D-6`), así que no se pudo ver si A − C2 ≈ 0,6 · (C2 − B) lejos de la glicina depende de
cómo MOZYME trata el largo alcance (con el valor por defecto, más allá de ~10 Å usa cargas y dipolos). El SCF convencional
no terminó en 45 min en la L4.

| Paso | Qué responde | Tiempo en L4 |
|---|---|---|
| `B_c15t`: B con `CUTOFF=15`, la traza del SCF en GPU (`MOPAC_MOZYME_SCF_TRACE=1`) | ¿la energía se estanca en un piso o oscila? | ~5 min |
| `B_c15cpu`: la misma corrida en el camino de CPU (`MOPAC_NOGPU=1`, con `PL`), en segundo plano | ¿falla también en CPU o es cosa de la GPU? | horas, hasta `CPU_TIMEOUT` |
| A, B y C2 con `SCFCRT=1.D-5`: corte por defecto (`vacl`, y `vacl2` repetida), `CUTOFF=15` (`c15l`), `CUTOFF=25` (`c25l`) | ¿converge con un criterio más suelto, y cambia A − C2 frente a C2 − B? | ~2 min (corte por defecto) a ~10 min (`CUTOFF=25`) cada una |

Sirve cualquier GPU; los resultados van a `MyDrive/mopac_paper_bench/allo_t1c_<gpu>` y, si la sesión se corta, lo
terminado se salta. **Antes de empezar:** *Entorno de ejecución → Cambiar tipo de entorno → GPU* (la L4 sirvió), y
luego *Ejecutar todo*. La última celda espera a la corrida de CPU y descarga un zip; si no quiere esperar, la penúltima
ya imprime el análisis de lo que hizo la GPU.
""")

md("## 1. Configuración")
code(f"""
COMMIT = '{commit}'
RESULTS_NAME = 'allo_t1c'      # la carpeta lleva además el nombre de la GPU: allo_t1c_l4, allo_t1c_a100
USE_DRIVE = True
REQUIRE_GPU = None             # 'A100', 'L4' o 'T4': parar si la sesión tiene otra GPU; None: cualquiera
CPU_TIMEOUT = 4 * 3600         # s para la corrida de CPU
""")
md("## 2. GPU y carpeta de resultados")
code("""
import os, re, subprocess, sys, time, shutil, json, gzip, base64
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
GPU_TAG = [w for w in gpu.split(',')[0].split() if w.lower() not in ('nvidia', 'tesla')][0].split('-')[0].lower()
if not RESULTS_NAME.endswith(f'_{GPU_TAG}'):
    RESULTS_NAME = f'{RESULTS_NAME}_{GPU_TAG}'
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
## 5. La corrida de CPU, en segundo plano

B con `CUTOFF=15` en el camino de CPU (un núcleo; MOZYME en CPU no usa más). Sigue mientras la GPU trabaja; su salida va a
`cpu_job.log`.
""")
code("""
cpu = subprocess.Popen([sys.executable, '/content/run_t1.py', str(MOPAC), str(WD), 'B_c15cpu', '1', str(CPU_TIMEOUT)],
                       stdout=open(OUT / 'cpu_job.log', 'a'), stderr=subprocess.STDOUT)
print('corrida de CPU en marcha, pid', cpu.pid)
""")
md("""
## 6. La traza del SCF en GPU

Una línea por iteración: `escf` (energía SCF), `de` (cambio respecto de la anterior), `ovmax` y la decisión del bucle.
""")
code("""
run(f'{sys.executable} /content/run_t1.py {MOPAC} {WD} B_c15t 1', check=False)
tr = [l for l in (WD / 'B_c15t' / 'job.log').read_text(errors='ignore').splitlines() if 'SCF trace' in l]
print(len(tr), 'iteraciones')
print('\\n'.join(tr[:5] + ['...'] + tr[-15:]))
de = [abs(float(m.group(1))) for l in tr for m in [re.search(r' de=(\\S+)', l)] if m]
if de:
    print(f'|de| más pequeño: {min(de):.2e}; mediana de las últimas 50: {sorted(de[-50:])[len(de[-50:]) // 2]:.2e}')
""")
md("""
## 7. Criterio `SCFCRT=1.D-5`

C2 sale de las cargas de A de la misma variante, así que va después y solo si A terminó con calor de formación.
""")
code("""
blocks = [('vacl', ['A_vacl', 'B_vacl', 'C2_vacl']), ('c15l', ['A_c15l', 'B_c15l', 'C2_c15l']),
          ('c25l', ['A_c25l', 'B_c25l', 'C2_c25l']), ('vacl2', ['A_vacl2', 'B_vacl2'])]
for v, jobs in blocks:
    first = [j for j in jobs if not j.startswith('C2_')]
    run(f'{sys.executable} /content/run_t1.py {MOPAC} {WD} {",".join(first)} 1', check=False)
    st = WD / f'A_{v}' / 'status.json'
    if len(first) < len(jobs) and st.exists() and json.loads(st.read_text()).get('heat'):
        if run(f'{sys.executable} /content/t1_tools.py c2 {WD} {v}', check=False) == 0:
            run(f'{sys.executable} /content/run_t1.py {MOPAC} {WD} C2_{v} 1', check=False)
run(f'{sys.executable} /content/t1_tools.py analyse {WD} {OUT / "analysis"}', check=False)
""")
md("""
## 8. Esperar a la CPU, análisis final y descarga

Si la corrida de CPU no terminó en `CPU_TIMEOUT`, las últimas líneas `ITER.` muestran cómo iba su SCF.
""")
code("""
if cpu.poll() is None:
    print('esperando a la corrida de CPU...', flush=True)
    cpu.wait()
print((OUT / 'cpu_job.log').read_text())
outf = WD / 'B_c15cpu' / 'job.out'
if outf.exists():
    it = [l for l in outf.read_text(errors='ignore').splitlines() if l.strip().startswith('ITER.')]
    print(len(it), 'iteraciones en CPU'); print('\\n'.join(it[-15:]))
run(f'{sys.executable} /content/t1_tools.py analyse {WD} {OUT / "analysis"}', check=False)
zipbase = f'/content/{RESULTS_NAME}'
shutil.make_archive(zipbase, 'zip', OUT.parent, OUT.name)
print(f'{zipbase}.zip', os.path.getsize(f'{zipbase}.zip') // 1024, 'kB')
try:
    from google.colab import files
    files.download(f'{zipbase}.zip')
except Exception as e:
    print('descárguelo a mano:', e)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
Path(out).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(json.dumps(nb)) // 1024, "kB")
