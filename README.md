# Señal cuántica y alosteria en GluN1

**Pregunta.** Cuando la glicina se une al dominio de unión de GluN1 (receptor NMDA), ¿la información viaja por la
proteína de forma electrónica (un "cable" de electrones, huecos o protones), o el efecto electrónico se queda en el sitio
y lo que viaja es mecánico?

Los cálculos cuánticos usan el MOPAC de [juvenalyosa/mopac_gpu](https://github.com/juvenalyosa/mopac_gpu) (PM7, MOZYME,
GPU). Este repositorio tiene los scripts, las estructuras preparadas, el cuaderno de Colab y los resultados.

## Estado (2026-11-20)

### 1. Rastreo geométrico en cristales (`results/survey.txt`, `rotamers.txt`, `hbonds.txt`)

Distancias de borde a borde entre cadenas laterales; camino que hace más corto su salto más largo. Es geometría, no
transporte.

- **Controles:** el método encuentra la tríada de Trp de la fotoliasa (FAD → W382 → W359 → W306, salto más largo 5,2 Å) y
  el camino de la ribonucleótido reductasa (Y122 → W48 → Y356 → Y731 → Y730 → C439, 8,9 Å). Cable de protones de la GFP:
  cromóforo → agua → S205 → E222, en 3 pasos.
- **GluN1 + glicina, solo Trp/Tyr:** W223 a 3,8 Å de la glicina y después un hueco de 13,2 Å (12,3 Å girando rotámeros
  con la cadena principal fija).
- **Con Phe e His:** glicina → F92 → W223 → F221 → Y203 → H201 → H197 → Y173 → F171, saltos de 6,4 Å como mucho (5,4 Å con
  giros); el 8 % de los conjuntos de cadenas laterales al azar está igual de conectado.
- **Cables de protones desde la glicina:** solo R131, D224 y, con aguas, E130; hasta 11–13 Å.

### 2. Energías de sitio y acoplamientos del camino aromático (`results/test2.txt`)

Modelo de cada cadena lateral (H en lugar del Cα), PM7: energía vertical para quitar un electrón, sola y en el potencial
de la proteína (cargas Amber, ε = 4 y 1); acoplamiento entre los orbitales ocupados más altos de vecinos (proyección
sobre la matriz de Fock del par).

- W223 es el sitio más fácil de oxidar del camino en los tres modelos de entorno; pasar de W223 a F221 cuesta +1,2 a
  +1,3 eV (unas 47 kT).
- F221–Y203 y Y203–H201 se acoplan con 0,1–0,3 meV (sin los átomos de en medio; puede estar subestimado).
- Control: el modelo **no** reproduce la dirección conocida de la tríada de la fotoliasa (energías casi planas; faltan el
  FAD y la solvatación). Diferencias de unas décimas de eV no son fiables; el paso de 1,2 eV sí lo es.

### 3. Alcance del efecto electrónico de la glicina, geometría congelada (`results/test1_cpu/`)

A = proteína + glicina, B = proteína sola, C = proteína + la glicina como cargas puntuales (sus cargas ESP aislada),
C2 = lo mismo con las cargas que la glicina tiene dentro de la proteína (de A). PM7, MOZYME, 4 604 átomos de proteína,
vacío, CPU. Ruido: las mismas corridas con los átomos en orden inverso.

Cambio de las cargas atómicas de la proteína (e), mediana por distancia a la glicina:

| Distancia | A − B | C − B | C2 − B | A − C2 | ruido de A − C |
|---|---|---|---|---|---|
| 0–4 Å | 1,5·10⁻² | 1,4·10⁻² | 1,6·10⁻² | 3,3·10⁻³ | 2·10⁻⁷ |
| 8–10 Å | 1,2·10⁻³ | 6,4·10⁻⁴ | 7,7·10⁻⁴ | 5,5·10⁻⁴ | 2·10⁻⁷ |
| 15–20 Å | 2,3·10⁻⁴ | 1,3·10⁻⁴ | 1,6·10⁻⁴ | 9,2·10⁻⁵ | 2·10⁻⁷ |
| 25–30 Å | 6,5·10⁻⁵ | 3,9·10⁻⁵ | 4,5·10⁻⁵ | 2,7·10⁻⁵ | 1·10⁻⁷ |

- El efecto llega a toda la proteína (34 Å) y está muy por encima del ruido. Decae como una potencia de la distancia
  (≈ r^−2,6 entre 9 y 28 Å), como un campo electrostático, no como un efecto electrónico de corto alcance.
- Dentro de la proteína la glicina está más polarizada (dipolo 14,9 D contra 12,4 D aislada) y le pasa 0,008 e a la
  proteína.
- Con las cargas que la glicina tiene en la proteína (C2), la respuesta lejana sigue siendo ~1,5 veces menor que con la
  glicina cuántica: A − C2 es ~0,6 veces C2 − B en cada intervalo más allá de 10 Å (misma forma, otra amplitud). El dipolo
  de la glicina más la capa cercana (≤ 12 Å) difiere solo un 6–19 % entre A y C2, así que esto no está explicado. Puede
  depender de cómo MOZYME trata el largo alcance; el cuaderno lo mira con `CUTOFF=25` y agua implícita.
- Los residuos del camino aromático no se afectan más que otros a la misma distancia (percentiles 3–87 entre los residuos
  a ±2 Å, en A − B, A − C y A − C2).

#### En GPU y en agua implícita (`results/test1_l4/`)

Colab con una L4, el mismo código de MOPAC (commit `323b6076`), el camino de GPU:

- La GPU reproduce la CPU: los calores de A, B, C y C2 difieren en 0,0003–0,0012 kcal/mol (entre dos corridas de CPU con
  los átomos en otro orden, hasta 0,0025), y las tablas de cargas y fuerzas coinciden con las de CPU en dos cifras. Cada corrida tardó
  unos 2 min (en CPU, 35–95 min).
- Ruido: repetir la misma corrida en la GPU cambia las cargas en una mediana de 2–3·10⁻⁸ e, y el orden inverso de los
  átomos en 3–8·10⁻⁸ e. El A − B más pequeño (30–40 Å, 5·10⁻⁵ e) es unas 2 000 veces mayor.
- Agua (COSMO, `EPS=78.4`): A − B se reduce poco cerca de la glicina y más lejos. Cargas: 73–91 % del valor en vacío hasta
  12 Å, 37–54 % más allá de 15 Å; fuerzas: 74–87 % hasta 10 Å, 38–43 % más allá de 15 Å. Las diferencias con C y C2 en agua
  no se pueden comparar con A: según el código de este MOPAC, las cargas puntuales (`MM_CHARGES`) solo entran en el
  hamiltoniano y COSMO no las apantalla, mientras que la glicina de A sí queda apantallada.
- En vacío, A − C2 dividido por C2 − B (medianas por intervalo): 0,56–0,61 en las cargas entre 10 y 30 Å (0,48 a 30–40 Å)
  y 0,37–0,53 en las fuerzas más allá de 8 Å.
- Sin resultado: las seis corridas con `CUTOFF=15` y `CUTOFF=25` no convergieron (*THE SCF CALCULATION FAILED*, 300
  iteraciones con `SCFCRT=2.D-6`), y el SCF convencional no terminó en 45 min por corrida. La pregunta de si el cociente
  anterior depende del trato de largo alcance de MOZYME sigue abierta.

Siguiente: `notebooks/allo_t1c_colab.ipynb` (cualquier GPU;
[abrir en Colab](https://colab.research.google.com/github/juvenalyosa/allosteria-cuantica/blob/main/notebooks/allo_t1c_colab.ipynb)):
la traza del SCF en GPU de una corrida con `CUTOFF=15` y la misma corrida en CPU (¿falla también?), y A, B y C2 con
`SCFCRT=1.D-5` con el corte por defecto, `CUTOFF=15` y `CUTOFF=25`. Pendiente también: el SCF convencional en una GPU
con doble precisión rápida (A100), y si un relevo dinámico (con movimiento de átomos) lleva la señal; esta prueba es con
geometría congelada.

## Reproducir

```bash
python scripts/fetch_pdb.py pdb                                  # entradas del PDB
python scripts/survey_paths.py pdb results/survey.json            # rastreo geométrico (NumPy)
python scripts/rotamer_paths.py pdb results/rotamers.txt
python scripts/hbond_paths.py pdb results/hbonds.txt
python scripts/prep_structures.py pdb structures                  # OpenMM + PDBFixer
python scripts/path_qm.py /ruta/a/mopac structures work/t2 results/test2.txt
python scripts/t1_tools.py decks structures/1PB7_atoms.csv work/t1 2
python scripts/run_t1.py /ruta/a/mopac work/t1 D && python scripts/t1_tools.py mmcharges work/t1
python scripts/run_t1.py /ruta/a/mopac work/t1 A_cpu,B_cpu,C_cpu,A_rev,B_rev,C_rev 6
python scripts/t1_tools.py c2 work/t1 cpu && python scripts/run_t1.py /ruta/a/mopac work/t1 C2_cpu
python scripts/t1_tools.py analyse work/t1 results/test1_cpu
python scripts/make_allo_t1_nb.py notebooks/allo_t1_colab.ipynb 323b6076 scripts structures/1PB7_atoms.csv
python scripts/make_allo_t1_nb.py notebooks/allo_t1_colab_a100.ipynb 323b6076 scripts structures/1PB7_atoms.csv A100
python scripts/t1_tools.py analyse work/t1_l4/runs results/test1_l4        # runs of the L4 session
python scripts/make_allo_t1c_nb.py notebooks/allo_t1c_colab.ipynb 323b6076 scripts structures/1PB7_atoms.csv
```

## Referencias

Todas comprobadas por DOI en Crossref: `docs/referencias.md`.
