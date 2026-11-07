# Arquitectura prevista: alineación incremental de bloques

Estado: arquitectura objetivo. La primera API modular está implementada como
BlockAlign 0.2, con capacidades y límites en `modular-api.md` y ensayos reales
en `validation-modular.md`. Este documento incluye también evolución posterior
a la versión actual; no todos sus formatos y escenarios están implementados.

## Objetivo y frontera entre proyectos

El proyecto de reconstrucción produce cada bloque: segmentación, sparse COLMAP,
undistorsión y densificación OpenMVS. Este repositorio recibe paquetes de
resultados versionados y mantiene conexiones y posiciones comunes entre bloques.
La integración futura se hará mediante manifiestos, sin importar internals del
otro repositorio. El mapa provisional contiene 25 cuadras numeradas; su
disponibilidad y calidad de reconstrucción deben registrarse individualmente.

## Dos modos

1. **Placement:** conservar geometría interna y estimar una similitud por bloque.
   Reutilizar las nubes densas existentes mediante transformaciones conocidas.
2. **Correction:** ajustar cámaras y puntos de uniones seleccionadas. Generar una
   variante sparse y declarar los derivados densos que deben regenerarse.

Una nube densa previa no incorpora la corrección interna de un BA mediante una
sola Sim(3). La primera versión general tendrá placement como modo por defecto.

## Contrato del bloque

Cada `BlockRevision` declarará:

- Identidad de bloque, captura y revisión inmutable.
- Modelo COLMAP completo, incluidas cámaras optimizadas, rigs y frames.
- BD de features, receta, identidad y orden de keypoints.
- Raíz de imágenes con identidades y hashes verificables.
- Máscaras y procedencia de segmentación cuando correspondan.
- Región principal, regiones observadas, referencias GPS y sus incertidumbres.
- Activos densos y mallas opcionales, fingerprint del sparse que los generó,
  marco de coordenadas y transformación del activo al sparse.
- Versiones de software y procedencia.

No asumir compatibilidad por nombres de carpetas. Un PLY aislado no ofrece el
mismo contrato de registro visual que un modelo con cámaras y tracks.

## Geografía

Separar núcleo de manzana, región ampliada de selección, cobertura reconstruida
y zona de unión visualmente verificada. Mantener CRS, unidades, origen y datum
vertical explícitos. La etiqueta de captura no es un código catastral.

Conservar observaciones sparse periféricas aunque la densificación se recorte
a una ROI. La cámara puede observar una manzana sin situarse dentro de ella.

## Grafo como representación principal

- Nodo: una revisión de bloque y sus resúmenes.
- Arista: una alineación relativa con evidencia, validación y configuración.
- Solución del mapa: revisiones y aristas aceptadas, ancla, componentes,
  transformaciones globales y diagnósticos de ciclos.

Medir conexiones contra bloques originales. Evitar que una concatenación de
modelos fusionados sea el único estado del sistema: introduce dependencia del
orden y dificulta caché y recuperación.

Para una medición `Z_j_from_i`, exigir aproximadamente:

```text
Z_j_from_i = inverse(S_world_from_j) * S_world_from_i
```

Optimizar robustamente el grafo Sim(3), fijar el gauge e incorporar GPS con
incertidumbre. No forzar conexiones entre componentes sin evidencia visual.
Evaluar diversidad de soporte, orientación, escala y ciclos; el número de
matches no es una medida suficiente de confianza.

## Coste incremental

Proponer vecinos con un índice espacial, filtrar por cobertura y tracks y hacer
matching solo entre vistas seleccionadas. Presupuesto inicial orientativo de
4–6 conexiones candidatas por bloque, ajustable para asegurar conectividad y
ciclos útiles.

Con N bloques, k vecinos y presupuesto visual B por conexión, el trabajo visual
esperado se aproxima a O(N k B), y el de una incorporación a O(k B). El riesgo
de comparar todos los pares es cuadrático, no exponencial. La optimización
global y la exportación siguen dependiendo del tamaño del mapa y los datos.

Al agregar un bloque, reutilizar conexiones anteriores compatibles. Al cambiar
una revisión, invalidar sus dependencias afectadas, no todo el workspace.

## Motor por pares

```text
auditar entradas → prior geográfico → selección GIS/tracks → pares de imágenes
→ matching existente → PnP fijo destino → Sim(3) → validación → arista
```

Generalizar el prototipo eliminando escenas, rutas, rangos y offsets fijos.
Mantener identidad física de imagen, identidad de features y calibración como
conceptos distintos. Asignar IDs enteros COLMAP solo en operaciones concretas,
guardando mapas explícitos. No fusionar índices 2D de recetas incompatibles.

## Densos, mallas y marcos

Para un activo D perteneciente al bloque B:

```text
X_world = S_world_from_B * S_B_from_D * X_D
```

Transformar posiciones y normales correctamente y conservar procedencia. El
soporte de escenas `.mvs` requiere un adaptador específico y probado; no se
tratarán como PLY. Una similitud global no exige regenerar texturas UV, pero
una corrección interna puede invalidar profundidad, malla y texturas.

El producto predeterminado será un atlas de activos más transformaciones.
Concatenar en un PLY, consolidar puntos redundantes y reconstruir una superficie
conjunta serán operaciones distintas, opcionales y regionales. No reescribir
todos los densos al incorporar un bloque.

## API y módulos

API Python y CLI compartirán una fachada; HTTP podrá añadirse después.

```python
workspace.register_block(manifest)
plan = workspace.plan_incremental(add=[block], mode="placement", policy=policy)
run = workspace.submit(plan, executor=slurm)
workspace.status(run.id)
workspace.result(run.id)
workspace.export(map_version=version, product="dense_atlas", executor=slurm)
```

Las llamadas de control usarán metadatos y resúmenes disponibles. Las auditorías
costosas se incorporarán al plan como jobs. `submit` será asíncrono e idempotente.

Separar módulos de dominio, ingestión, geografía, registro por pares, grafo,
corrección, activos, adaptadores, ejecución, almacenamiento y evaluación.
El núcleo matemático no dependerá de Slurm ni de rutas globales.

## Estado y ejecución

Registro SQLite para metadatos transaccionales, manifiestos versionados y
almacenamiento de artefactos grandes separado. Solicitudes inmutables y salidas
publicadas de forma atómica. No depender de punteros `.state` cambiantes durante
una ejecución; resolver y fijar las entradas al crear el plan.

Claves de caché por etapa: revisiones, fingerprints de modelos/features,
selección, parámetros y motor. Cambiar la visualización no invalida matching;
cambiar las máscaras usadas para generar features puede invalidarlo.

Todas las fases numéricas y pruebas se ejecutarán mediante Slurm, con temporales
en `/local`, cuotas de concurrencia, checkpoints y recuperación de envíos
inciertos. CPU como base; GPU solo para tareas que realmente la requieran.

## Corrección e invalidación

El BA genera una variante derivada de una versión del mapa. Conservar originales
y aristas medidas. Declarar bloques modificados, activos incompatibles y
solicitudes de regeneración para el proyecto de reconstrucción. No mezclar sin
declararlo densos antiguos con poses o geometría corregidas.

## Pruebas y adopción

- Regresión 005–009 con tolerancias geométricas y preservación de calibraciones.
- Grafo sintético de 25 bloques: ruido, ciclos, outliers, componentes y revisiones.
- Identidades, segmentación, orden de features y consistencia sparse–denso.
- Transformaciones, normales, unidades y marcos adicionales de exportación.
- Incrementos reales: 005–009, agregar 001, cerrar un ciclo, 8–10 bloques y luego
  las 25 cuadras disponibles.
- Medir caché reutilizada, coste por incorporación, RAM, cambios globales y
  calidad de unión. No extrapolar automáticamente el tiempo de un solo par.

Orden de implementación: contratos → motor por pares → grafo incremental → API,
caché y Slurm → activos densos → corrección/invalidation → validación de escala.
