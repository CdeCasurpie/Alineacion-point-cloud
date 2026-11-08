# Alineación de bloques SfM de Barranco (experimento de tesis)

Este repositorio alinea reconstrucciones COLMAP independientes de cuadras
(bloques) de Barranco en un solo marco métrico común, usando prior GPS,
selección GIS/tracks, matching SIFT cruzado y validación robusta de Sim(3).
No vuelve a ejecutar SfM: las reconstrucciones per-cuadra del pipeline de
fotogrametría se toman como entrada inmutable.

## Contenido principal

- **`src/blockalign/` — API modular BlockAlign 0.2** (usar esta). Registro de
  bloques, planes inmutables, alineación por pares auditada, grafo robusto,
  campañas multi-bloque, corrección local explícita y exportación.
  **Uso y contratos:** [`docs/modular-api.md`](docs/modular-api.md).
- **`src/alignexp/` — prototipo 005–009** publicado en
  `results/prototype-61384/` (ver `results/LATEST.md`). Conservado como
  regresión; sus kernels (`binary.py`, `sim3.py`, `colmap_io.py`) son
  compartidos con el motor de `blockalign`. No requiere ejecutarse para usar
  la API nueva.
- **`docs/`** — arquitectura planificada
  ([`architecture.md`](docs/architecture.md)), validación de la API
  ([`validation-modular.md`](docs/validation-modular.md)), resultados de la
  campaña de 12 cuadras
  ([`campaign-barranco.md`](docs/campaign-barranco.md)).
- **`var/barranco/`** — workspace operativo (fuera de Git): registros,
  auditorías, mediciones, mapas y exports.
- **`tools/` y `slurm/`** — demo, inventario Barranco y runners de Slurm.

## Estado y resultados

1. **Prototipo 005–009** (Slurm 2026-11-07, `results/prototype-61384/`):
   30/32 cámaras puente, discrepancia de centros reservados 0.13–0.70 m,
   rotación <0.28°; fusión de 1131 imágenes y 880155 puntos; reproyección
   global 1.0710 → 1.0087 px; BA acotado usable. Productos:
   `sparse_metric.ply`, `E1_visual_colored.ply` (005 azul / 009 naranja) y
   `model_colmap_ref009/` con `coordinates.json` explicando los marcos.

2. **Mapa incremental 001–005–009** (`docs/validation-modular.md`): 25 pruebas
   aprobadas; arista 001–005 aceptada (fit mediano 0.0896 m) y 001–009
   rechazada por evidencia insuficiente; export `combined.ply` de 1113792
   puntos.

3. **Campaña Barranco de 12 cuadras** (`docs/campaign-barranco.md`): 001, 002,
   003, 004, 005, 007, 008, 009, 012, 013, 017 y 018 forman **un solo
   componente conexo resuelto** (mapa `d00786da…`, 4.17 M puntos sparse,
   4,715 imágenes; 13 aristas aceptadas de 36 intentos; 0 fallos técnicos).
   Slurm 61429, 44 min en CPU. Exports en `var/barranco/exports/`:
   - `9310a631…/` — `combined.ply` con **color plano por cuadra** y campo
     `block_id` (uint16), más `palette.json`, `legend.png` y `overview_xy.png`.
   - `fafe04c2…/` — `combined.ply` con el RGB fotográfico original.

   Las otras 13 cuadras del mapa provisional aún no tienen reconstrucción
   utilizable; 003 entra con su componente principal y estado `partial`.

## Flujo típico con la API

```python
from blockalign import Workspace
from blockalign.contracts import from_colmap_run

w = Workspace.open('var/barranco')
ref = w.register_block(from_colmap_run('005', 'ruta/al/run',
    region_uri='manzanas.geojson', region_ids=['MZ-BAR-FD7E0473D4']))

plan = w.plan_add([ref], reference=ref_base)      # o w.plan_campaign(refs, ...)
run  = w.submit(plan, image='containers/alignment313.sif',
                cpus=8, memory_gb=24, time='02:00:00')
w.result(run['id'])

plan = w.plan_export(map_id, product='combined_ply',
                     block_assets={ref: 'sparse'}, color_mode='by_block')
run  = w.submit(plan, image='containers/alignment313.sif')
```

Para Barranco completo hay tres atajos documentados en
[`docs/modular-api.md`](docs/modular-api.md):

```bash
sbatch slurm/inventory-barranco.slurm /ruta/a/photogrammetry-pipeline
python3 tools/submit_barranco.py --inventory results/barranco-inventory-<JOB>/inventory.json \
  --workspace var/barranco --image containers/alignment313.sif
sbatch slurm/blockalign-test.slurm   # regresión completa (28 pruebas)
```

## Operación en Khipu

Todo cómputo y pruebas corren por **Slurm**; el login solo sirve para editar,
planificar y enviar. Sin GPU: 8 CPUs / 24 GB bastan para una campaña. El
scratch se crea en `/local/$USER/blockalign/<JOB>` y se limpia al terminar;
el workspace recibe solo artefactos pequeños publicados con verificación de
hashes. Los planes aceptan snapshots de código y runtime; el mismo plan con
mismo código/imagen es idempotente (no duplica jobs). Fuente con SQLite WAL
activa se rechaza: hay que exportar primero.

## Contratos clave (detallados en docs/modular-api.md)

- Intrínsecos leídos de `cameras.bin` optimizado, nunca de priors de la BD.
- Manifiestos y políticas 1.0 estrictos; revisión declarada ≠ fingerprint
  observado del contenido; cambios bajo una revisión auditada producen error,
  no actualización silenciosa.
- Extraños rechazados en el grafo; bloques sin conexión verificada quedan en
  `disconnected`, nunca presentados como alineados por solo tener GPS.
- IDs de vista/cámara operación-local, sin inferencia por rango numérico.
- Los `combined.ply` no deduplican superficies; el denso OpenMVS se consume
  por activo con lineage verificado (no se transforma un `.mvs` como PLY).

## Límites actuales

- Solo sparse en campaña; sin densos OpenMVS disponibles en los runs.
- Sin cierre de ciclo real (validado con ciclos sintéticos) ni BA urbano
  global; la corrección es una unión por pares derivada.
- COLMAP/PyCOLMAP 3.13.0, SIFT 128, SIMPLE_RADIAL, rigs triviales,
  EPSG:32718, referencias GPS requeridas, nombres de imagen sin espacios.

## Historial de diagnóstico

Los primeros registros cruzados fallaron la validación: un remapeador copiaba
intrínsecos de la BD (priors) en lugar de los refinados del modelo (corregido
y validado por reproyección), y un diagnóstico intermedio convirtió mal las
unidades (descartado). La auditoría de fusión exigía erróneamente una feature
única por imagen y punto; la versión correcta valida reciprocidad, no
duplicación idéntica y no aumento de multiplicidad contra los originales.
Los gates no se relajan para publicar resultados fallidos.
