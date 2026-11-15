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

### 3. Alcance del efecto electrónico de la glicina, geometría congelada (en curso)

A = proteína + glicina, B = proteína sola, C = proteína + la glicina como cargas puntuales, D = glicina sola. A − B,
C − B y A − C contra la distancia, con el ruido de dos corridas iguales, CPU, `CUTOFF=15` y `25`, agua implícita y SCF
convencional. Cuaderno: `notebooks/allo_t1_colab.ipynb` (Colab A100; *Archivo → Subir cuaderno*).

## Reproducir

```bash
python scripts/fetch_pdb.py pdb                                  # entradas del PDB
python scripts/survey_paths.py pdb results/survey.json            # rastreo geométrico (NumPy)
python scripts/rotamer_paths.py pdb results/rotamers.txt
python scripts/hbond_paths.py pdb results/hbonds.txt
python scripts/prep_structures.py pdb structures                  # OpenMM + PDBFixer
python scripts/path_qm.py /ruta/a/mopac structures work/t2 results/test2.txt
python scripts/t1_tools.py decks structures/1PB7_atoms.csv work/t1 2
python scripts/make_allo_t1_nb.py notebooks/allo_t1_colab.ipynb 323b6076 scripts structures/1PB7_atoms.csv
```

## Referencias

Todas comprobadas por DOI en Crossref: `docs/referencias.md`.
