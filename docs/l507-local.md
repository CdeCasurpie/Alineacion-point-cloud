# Ejecución local supervisada en L507

La campaña L507 ejecuta el motor `blockalign` en la maestra, sin scheduler Slurm.
El entorno numérico es independiente del pipeline de reconstrucción: este usa
COLMAP/PyCOLMAP 3.11.1; el alineador conserva su contrato estricto 3.13.0.

## Cambios dentro de este repositorio

- `artifacts.require_job()` acepta Slurm o un contexto local explícito ligado
  al PID real. En modo local comprueba los límites efectivos de cgroup v2:
  RAM finita de hasta 40 GiB, swap adicional cero y cuota de CPU de hasta ocho.
  Una variable de entorno por sí sola no admite trabajo numérico sin límites.
- La campaña obtiene sus hilos de `BLOCKALIGN_CPUS`, con el fallback original
  `SLURM_CPUS_PER_TASK` para el cluster.
- La publicación local en el mismo filesystem mueve el producto derivado por
  rename a `.pending-*` y después a su destino. Evita una segunda copia temporal
  de exportaciones densas muy grandes. En Slurm/otros filesystems se conserva
  el comportamiento de copia.

No se modifican los kernels de matching/PnP/Sim(3), sus tolerancias, el grafo ni
los controles de procedencia de activos densos.

## Integración con el controlador de laboratorio

Los scripts de integración se versionan en el repositorio independiente
`/home/exitoso/lab-deploy`, bajo `tools/`:

- `local_alignment.py`: runner CPU con scopes systemd reales, supervisor local,
  identidad del runtime, auditoría, campaña y exportaciones.
- `import_alignment_run.py`: importación inmutable 3.11→3.13, rigs triviales,
  adaptación de BD manteniendo bytes/índices SIFT y verificación de poses,
  intrínsecos, observaciones y XYZ. No ejecuta SfM ni optimiza esa importación.
- `campaign_post.py`: catálogo frame/video/PTS, snapshot SQLite consistente,
  verificación del marco sparse/undistorted y PLY escalar compatible.
- `check_alignment_identity.py`: fixture de dos namespaces de la misma
  reconstrucción; valida el motor numérico sin confundirla con dos cuadras reales.
- `run_reconstruction_campaign.py` y `collect_reconstruction.py`: coordinación,
  publicación de snapshots y recolección con SHA256.

El runtime local está en `/data/barranco/runtime/alignment313`. El modo local
es opt-in; la API `Workspace.submit()` sigue siendo el envío a Slurm.

Ejemplo de auditoría con el runner de integración:

```bash
systemd-run --user --scope --unit=barranco-alignment-audit \
  -p MemoryMax=40G -p MemorySwapMax=0 -p CPUQuota=800% \
  /data/barranco/runtime/alignment313/bin/python -B \
  /home/exitoso/lab-deploy/tools/local_alignment.py audit \
  --workspace /data/barranco/alignment/barranco-full-v1 \
  --manifests /ruta/al/block.json
```

El runner crea `BLOCKALIGN_LOCAL_CONTEXT` para su proceso y valida un scope
`barranco-alignment-*`; no fabrica identificadores Slurm. Los archivos de
registro y workspace son estado operativo fuera de Git.

## Densos y memoria

OpenMVS produce PLY con listas de visibilidad. El lector del alineador admite
PLY escalares: la integración deriva XYZ/RGB/normales y conserva el original
en la hija. Para escenas grandes, la reconstrucción usa lotes de cámaras
solapados, en el marco sparse global inmutable, con pruebas de equivalencia
de poses/intrínsecos/XYZ. La concatenación retiene redundancia de los solapes.

La importación comprueba la procedencia del modelo y el fingerprint sparse
canónico. La exportación sigue exigiendo `generated_from_sparse` correcto y
compone `world_from_sparse @ sparse_from_asset`. No aplica un transform a un
`.mvs` como si fuera PLY, ni presenta esta concatenación como una malla o una
fusión de superficies deduplicada.

## Validación realizada

- Regresión del repositorio y pruebas del backend local: admisión/denegación
  por límites efectivos, PID, compatibilidad Slurm y publicación sin duplicar.
- Fixture de identidad real: ocho cámaras puente registradas, Sim(3) próxima
  a identidad y discrepancias reservadas de aproximadamente 2–4 mm.
- Campaña de 15 manzanas: 17 aristas aceptadas/utilizadas, un componente de 15
  bloques, cero auditorías rechazadas y cero fallos técnicos de alineación.
- Sparse combinada: 4.065.767 puntos. Densa: 5.675.791.302 puntos, 153,29 GiB;
  se exportó por streaming y se verificó SHA256 sin cargarla entera en RAM.
- 023 conserva aviso de cobertura: 468/480 vistas, 97,5%, con los cuatro clips
  representados. No se redujeron tolerancias para forzar las doce vistas restantes.

Entrega local: `/data/barranco/exports/barranco-full-v1/`, con `RESUMEN.md`,
`report.json`, atlas, nube global, preview y vista sparse por manzana.
