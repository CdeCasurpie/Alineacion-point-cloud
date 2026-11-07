# BlockAlign 0.2: API modular y operación en Khipu

La nueva API está en `blockalign`. `alignexp` conserva el prototipo como
regresión y aporta kernels compartidos; no hay que ejecutar sus fases para
utilizar la API nueva. Los experimentos nuevos se almacenan en un workspace
separado (`var/barranco/` en esta instalación).

## Capacidades implementadas

- Manifiestos y políticas 1.0 con campos estrictos.
- Registro de revisiones por hash de la declaración; auditoría posterior de
  contenido y calibración/identidades. La revisión declarada y el fingerprint
  observado del contenido son valores distintos y se conservan ambos.
- Lectura de la API de control sin importar NumPy, PyCOLMAP o ejecutar cómputo.
- Solicitudes con entradas fijas, snapshot de código y runner empaquetado.
- Envío asíncrono e idempotente a Slurm; reconciliación explícita de estados.
- Resúmenes GIS/GPS, vecinos espaciales y selección por tracks.
- Matching SIFT CPU y PnP con intrínsecos refinados del modelo.
- Validación de Sim(3) con cámaras reservadas y grafo robusto de bloques.
- Caché de auditorías, mediciones y rechazos por insuficiencia de evidencia.
- Incorporación de bloques conservando aristas compatibles de una versión base.
- Atlas, PLY por activo y concatenación PLY por streaming, conservando atributos
  escalares y transformando normales cuando están presentes.
- Campaña multi-bloque (`plan_campaign`): auditoría de todos los bloques,
  medición de vecinos declarados + vecinos espaciales, puentes inversos con
  presupuesto acotado y solución independiente por componente conexo.
- Exports categóricos: campo `block_id` (uint16) y color plano por cuadra con
  paleta determinista, leyenda PNG y vista XY de las nubes sparse colocadas.
- Corrección explícita de una unión por pares: fusión auditada, BA acotado,
  variante sparse derivada y solicitudes de regeneración.

## Frontera de soporte de esta versión

El adaptador numérico acepta COLMAP/PyCOLMAP 3.13.0, SIFT de 128 columnas,
cámaras SIMPLE_RADIAL y rigs triviales de referencia de una cámara. El GIS
operativo es EPSG:32718. Se requieren referencias geográficas de cámaras; no
se implementa aún recuperación global visual sin prior geográfico. Los nombres
de imagen no pueden contener espacios, por el transporte de pares de COLMAP.

Se admiten entradas densas PLY ASCII o little-endian con propiedades escalares;
se rechazan mallas/listas, big-endian, coordenadas enteras y esquemas incompatibles
para concatenación. No se transforma un `.mvs` como si fuera PLY. La versión
actual no deduplica superficies densas ni construye una malla urbana conjunta.

El grafo se valida con ciclos sintéticos. Incorporar tres bloques conectados no
demuestra por sí solo un cierre de ciclo real. El soporte de corrección es una
unión por pares derivada, no un BA urbano global ni un atlas corregido automático.

## Entorno y paquete

```bash
# Instalación de software en login; requiere un entorno Python compatible.
python -m pip install -e .

# Construcción alternativa del runtime, con base COLMAP 3.13 proporcionada.
bash tools/build_environment.sh /ruta/a/sparse313.sif
```

La base debe contener `/opt/colmap/bin/python` y COLMAP 3.13. Las versiones
resueltas se conservan dentro de la imagen. El workspace registra identidad del
contenedor y cada worker verifica las versiones y calcula su SHA-256.
El contenedor de esta instalación sigue siendo `containers/alignment313.sif`.

## Registrar una reconstrucción

```python
from blockalign import Workspace
from blockalign.contracts import from_colmap_run

w = Workspace.open('/ruta/al/workspace')

# Adaptador de los resultados del pipeline de reconstrucción existente.
manifest = from_colmap_run(
    block_id='cuadra-norte',
    run='/ruta/al/run-completo',
    region_uri='/ruta/a/manzanas.geojson',
    region_ids=['ID-GIS-VERIFICADO'],
)
ref = w.register_block(manifest)

# Alternativamente, un manifiesto JSON independiente.
ref = w.register_block('/ruta/a/block.json')
```

El registro hace validaciones de metadatos y captura stat de los archivos
principales disponibles. No calcula features ni hashes de imágenes en login.
La auditoría pesada se ejecuta en Slurm, registra hashes y conserva resúmenes.
Antes de reutilizar auditorías se verifican sus stamps, incluidos los destinos
de enlaces. Fuentes con WAL SQLite activo se rechazan: primero se requiere un
export consistente e inmutable.

Los bloques pueden referenciar datos existentes, no se duplica cada BD en el
workspace. Esto requiere conservar sus rutas y revisiones inmutables. Cambiar
archivos bajo una revisión auditada produce un error, no una actualización
silenciosa. Para datos portables debe prepararse un snapshot autosuficiente.

## Crear un mapa y ejecutar

```python
plan = w.plan_add([ref_a, ref_b], reference=ref_b)
run = w.submit(
    plan,
    image='/ruta/al/alignment313.sif',
    cpus=8,
    memory_gb=24,
    time='02:00:00',
)
print(run['id'], run['job_id'])

w.status(run['id'])
# Si hay cancelación/timeout o una respuesta de envío incierta:
w.reconcile(run['id'])

result = w.result(run['id'])  # exige publicación verificada
map_id = result['map_revision']
```

La misma solicitud con código y runtime idénticos devuelve la ejecución activa
o completada existente. No crea otro job. Un recibo de envío incierto queda
bloqueado hasta reconciliación; no se reenvía a ciegas. El éxito del scheduler
sin un resultado publicado no se interpreta como éxito científico.

Los planes no contienen punteros a «última fase». Los workers reciben la copia
de manifiestos, política y referencia elegidos. El job crea su scratch en
`/local/$USER/blockalign/$SLURM_JOB_ID`; las BD de matching se generan allí y no
se conserva una copia de varios GB por cada arista.

## Campaña multi-cuadra

Cuando hay varios bloques por registrar y medir, `plan_campaign` encapsula el
flujo completo de la campaña: auditoría de todos los bloques, medición de cada
vecindad declarada más la vecindad espacial sugerida por GIS, y puentes inversos
/expandidos con presupuesto acotado para reconectar componentes. Cada componente
conexo verificado se resuelve como su propio mapa; los bloques no conectados se
listan explícitamente y nunca se presentan como alineados.

```python
plan = w.plan_campaign(
    refs,                      # todas las referencias registradas
    reference=by_label['009'], # fija el origen del marco (opcional)
    map_hints=hints,           # pares visuales del mapa provisional (etiquetas)
    max_expanded_attempts=12,  # puentes extra por distancia
)
run = w.submit(plan, image='/ruta/al/alignment313.sif', cpus=8, memory_gb=24,
               time='04:00:00')
```

- `map_hints` son pares sugeridos (no correspondencias verificadas); el motor
  sigue exigiendo evidencia suficiente por arista en ambos sentidos.
- Los pares que fallan por primera vez se reintentan en sentido inverso solo si
  aún separan componentes; los rechazos por insuficiencia de evidencia se
  conservan como artefactos persistentes y no se recomputan.
- El progreso se escribe en el checkpoint del run (`campaign-progress.json`)
  después de cada par; un reinicio reutiliza auditorías y mediciones cacheadas.
- El informe final (incluido en `result(run)`) lista componentes, intentos,
  aceptados, fallos técnicos y rechazos de auditoría.

Para Barranco, `tools/submit_barranco.py` encadena inventario previo + registro
+ campaña con las vecindades de `examples/barranco-map-neighbors.json`:

```bash
sbatch slurm/inventory-barranco.slurm /ruta/a/photogrammetry-pipeline
python3 tools/submit_barranco.py \
  --inventory results/barranco-inventory-<JOB>/inventory.json \
  --workspace var/barranco --image containers/alignment313.sif
```

`tools/inventory_barranco.py` valida los native models de las 25 etiquetas del
mapa provisional (metadatos solamente; la auditoría pesada sigue en la campaña)
y materializa `manifests/<block>.json` por bloque elegible.

## Agregar otra cuadra

```python
new_ref = w.register_block('/ruta/a/otra-cuadra/block.json')
plan = w.plan_add([new_ref], base_map=map_id)
run = w.submit(plan, image='/ruta/al/alignment313.sif')
```

Se conserva el origen y la referencia de la versión base. Las conexiones
anteriores se reutilizan si política y motor son compatibles. Los parámetros
numéricos, revisiones y fingerprints forman la clave de cada conexión; cambios
de exportador o CLI no invalidan el motor de medición.

En esta primera versión la caché de conexiones está ligada también a las
revisiones declaradas: añadir activos editando el manifiesto crea otra revisión
y puede recalcular conexiones. La separación completa de AssetRevision de la
revisión geométrica queda como optimización posterior. Cambiar solamente la
configuración de una exportación no invalida matching.

El resultado puede ser `complete` o `partial`. Los bloques desconectados se
enumeran y no se presentan como alineados solo por contar con GPS. Los rechazos
visuales no convierten un cálculo válido en un error técnico; sus causas quedan
en el mapa. Excepciones de formato/runtime sí marcan la ejecución como fallida.

## Productos dense y marcos

Un activo del manifiesto debe incluir:

```json
{
  "asset_id": "dense-main",
  "kind": "point_cloud_ply",
  "uri": "dense/cloud.ply",
  "frame_id": "dense-local",
  "generated_from_sparse": "FINGERPRINT-DEL-MODELO-GENERADOR",
  "sparse_from_asset": [
    [1, 0, 0, 0], [0, 1, 0, 0],
    [0, 0, 1, 0], [0, 0, 0, 1]
  ]
}
```

El fingerprint esperado está en `map.json → sparse_fingerprints`. Debe constar
la procedencia real del activo, no completarse por suposición. Para transformar
un denso se compone `world_from_sparse @ sparse_from_asset`. La última fila debe
ser `[0,0,0,1]`; una diagonal 4x4 completa de 0.05 no representa una escala
física 0.05. No se normaliza automáticamente una matriz ambigua.

```python
plan = w.plan_export(map_id, product='atlas')
# O product='transformed_ply' / 'combined_ply'
run = w.submit(plan, image='/ruta/al/alignment313.sif')
```

Para una previsualización explícitamente sparse, sin fingir un resultado denso:

```python
plan = w.plan_export(
    map_id,
    product='combined_ply',
    block_assets={ref_a: 'sparse', ref_b: 'sparse'},
)
```

### Color por cuadra (exports categóricos)

Con `color_mode='by_block'` (por defecto `'original'` conserva el RGB
fotográfico) el export materializa además:

- `block_id`: campo escalar `ushort` adicional por vértice, con la etiqueta
  numérica de la cuadra; si la fuente ya lo trae se rechaza el export.
- Un color plano por cuadra de una paleta determinista (basada en `digest` de la
  etiqueta + proporción áurea de tono); no depende del orden de registro y se
  conserva al añadir más cuadras numeradas (estabilidad probada).
- `palette.json` (etiqueta → block_id/RGB/hex) y `legend.png` (leyenda).
- `overview_xy.png`: muestreo de cada bloque sparse en el marco común para
  inspección rápida de costuras.

```python
plan = w.plan_export(map_id, product='combined_ply',
                     block_assets={ref: 'sparse' for ref in refs},
                     color_mode='by_block')
run = w.submit(plan, image='/ruta/al/alignment313.sif', cpus=4, memory_gb=16,
               time='00:40:00')
```

Nota: si los manifiestos no tienen activos PLY registrados (`assets: []`, el
caso habitual con solo modelos COLMAP), hay que pasar `block_assets={ref:
'sparse'}` para cada bloque; sin ello la concatenación no recibe entradas y
falla con `ValueError: No cloud inputs`.

La concatenación no elimina duplicados ni corrige geometría interna. Sus
vértices pueden superar el conteo del modelo sparse fusionado del prototipo.

## Corrección y regeneración

```python
plan = w.plan_correction(map_id, measurement_id, max_iterations=40)
run = w.submit(plan, image='/ruta/al/alignment313.sif')
```

Publica `corrections/<id>/` con el modelo de componente en el marco original del
destino, PLY métrico, informe del solver, membresía de imágenes y auditorías.
Las imágenes se identifican como `source/<nombre>` y `target/<nombre>` y
`images.json` declara las raíces originales para el consumidor.

Los dos bloques de la unión se marcan conservadoramente para regeneración de
sus densos. No se modifica la versión de mapa de colocación ni los originales.
Las solicitudes especifican modelo e image_ids; el futuro consumidor debe
soportar este contrato de componente. Aún no se regeneran OpenMVS, profundidad,
mallas ni texturas desde esta biblioteca.

## CLI

```bash
blockalign --workspace /ruta/workspace register block.json
blockalign --workspace /ruta/workspace plan-add REF_A REF_B --reference REF_B
blockalign --workspace /ruta/workspace submit PLAN_ID --image /ruta/runtime.sif
blockalign --workspace /ruta/workspace status RUN_ID
blockalign --workspace /ruta/workspace result RUN_ID
blockalign --workspace /ruta/workspace plan-export MAP_ID --product atlas
```

Cada comando imprime JSON. También puede ejecutarse sin instalación editable
con `PYTHONPATH=src python3 -B -m blockalign.cli ...` desde este repositorio.

## Pruebas y regresión

```bash
sbatch slurm/blockalign-test.slurm
```

Incluye contratos, API ligera, solicitudes inmutables, deduplicación de envíos,
matrices, grafo sintético de 25 bloques con outlier/isla, PLY con normales y
atributos, rechazo de mallas/payload incompleto y de lineage dense incompatible.
Los tests históricos de COLMAP siguen incluidos. Todo el cálculo se hace en
el nodo asignado, no en login.

`tools/modular_demo.py` es un ejemplo explícito para el snapshot local 005–009;
sus nombres específicos no forman parte del motor. Acepta `baseline`,
`incremental --base-map ... --block-id ... --run ...` y `export --base-map ...`.

## Evolución pendiente

- Más modelos de cámara/rig, CRS y referencias sin GPS.
- Separación total entre revisiones geométricas y activos densos.
- Holdout por grupos temporales y validación independiente después de BA.
- Cierres de ciclo reales y calibración de pesos/umbrales con más bloques.
- Materialización sparse multibloque con reconciliación transitiva de tracks.
- Adaptador de regeneración OpenMVS y mallas/texturas/teselas.
- Scheduler por etapas para campañas largas; actualmente un job por plan con
  caché persistente de etapas costosas entre reintentos.
