# Campaña Barranco: 12 cuadras alineadas (nov 2026)

Primera campaña real de la API modular más allá del caso 001–005–009. Extiende
la validación con un componente conexo de **12 cuadras** de Barranco.

## Resultados clave

| concepto | valor |
|---|---|
| componente conexo resuelto | 001, 002, 003, 004, 005, 007, 008, 009, 012, 013, 017, 018 |
| mapa principal | `maps/d00786dac35760cee43c13e7486247bf79f964c32578f771a368fa8133f38aba` |
| puntos sparse colocados | 4,171,639 (4,715 imágenes) |
| aristas | 13 aceptadas de 36 intentos; 23 rechazos por evidencia insuficiente |
| fallos técnicos / auditorías rechazadas | 0 / 0 |
| job de campaña | Slurm 61429 (44 min, 8 CPUs, 24 GB, pico ~25 GB) |
| informe de campaña | `campaigns/04aacc244c8f98…/campaign.json` |
| export by_block | `exports/9310a631bb80…` (`combined.ply` 68 MB, color + `block_id` por cuadra, `legend.png`, `overview_xy.png`) |
| export original | `exports/fafe04c2531…` (`combined.ply` 68 MB, RGB fotográfico) |

La geometría del componente en `overview_xy.png` reproduce la disposición del
mapa provisional: 001–004 al norte, 005 y 009 al sur de 001, 013/018 al sur,
012/017 al este y 007/008 al centro; solape contiguo entre vecinas.

## Método

1. **Inventario** (`slurm/inventory-barranco.slurm` → job 61424, ~1 min):
   valida metadatos de los 25 runs bajo `photogrammetry-pipeline/outputs/NNN`
   (native models 3.13, DB sin WAL, identidades modelo/DB, descriptores 128,
   rasters y telemetría presentes) y materializa un manifiesto por bloque
   elegible en `results/barranco-inventory-61424/manifests/`. Eligió 12 de 25
   (las demás cuadras no tienen aún reconstrucción compatible); 003 entró con su
   componente principal marcado `partial`.
2. **Registro + plan** (`tools/submit_barranco.py`, login): registra los 12
   bloques con `from_colmap_run` y fija puentes visuales de
   `examples/barranco-map-neighbors.json`. Para 005/009 reutilizó las regiones
   GIS ya verificadas; el resto quedó sin etiqueta de manzana (la evidencia de
   arista no depende del GIS para aceptarse, solo el prior).
3. **Campaña** (job 61429): auditoría de los 12 (001 reutilizó caché), pares
   vecinos + puentes inversos + 12 puentes expandidos de reserva, con checkpoint
   tras cada par. Los 23 rechazos corresponden a pares no vecinos cuyo apoyo
   GPS/tracks fue insuficiente; no afectan la conectividad lograda.
4. **Exports** (jobs 61441/61442, ~14 s c/u): cada bloque materializado como
   sparse-preview desde su native model, transformado por streaming y
   concatenado con paleta determinista + `block_id` ushort.

## Lecciones durante la campaña

- Los exports requieren `block_assets={ref: 'sparse'}` cuando el manifiesto no
  trae activos PLY registrados; si falta, la concatenación falla con
  `No cloud inputs`. Dos jobs se gastaron en detectarlo (61439/61440).
- Los checkpoints por par (`campaign-progress.json`) y el caché persistente de
  auditorías/mediciones hacen que un reintento sea barato: 001 reutilizó su
  auditoría del experimento anterior.
- El reinversión de pares solo se dispara para pares que separan componentes,
  evitando recomputar rechazos internos de un componente ya verificado.

## Limitaciones

- Solo sparse: no hay densos OpenMVS disponibles en los runs analizados, y el
  `combined.ply` no deduplica superficies.
- 003 mantiene su estado `partial` (componente principal recolectado); otras 13
  cuadras del mapa siguen sin reconstrucción utilizable.
- Vecindades sugeridas visualmente en `barranco-map-neighbors.json`; la
  aceptación de cada arista sigue validada por tracks + holdout Sim(3).
