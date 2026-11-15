# Señal cuántica y alosteria en GluN1

**Pregunta.** Cuando la glicina se une al dominio de unión de GluN1 (receptor NMDA), ¿la información viaja por la
proteína de forma electrónica (un "cable" de electrones, huecos o protones), o el efecto electrónico se queda en el sitio
y lo que viaja es mecánico?

Los cálculos cuánticos usan el MOPAC de [juvenalyosa/mopac_gpu](https://github.com/juvenalyosa/mopac_gpu) (PM7, MOZYME,
GPU). Este repositorio tiene los scripts, las estructuras preparadas, el cuaderno de Colab y los resultados.

## Estado (2026-11-15)

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

Siguiente: `notebooks/allo_t1_colab.ipynb` (Colab A100; *Archivo → Subir cuaderno*) repite la base en GPU y añade
`CUTOFF=25`, agua implícita, `CUTOFF=15` y SCF convencional. Pendiente también: si un relevo dinámico (con movimiento
de átomos) lleva la señal; esta prueba es con geometría congelada.

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
```

## Referencias

Todas comprobadas por DOI en Crossref: `docs/referencias.md`.
