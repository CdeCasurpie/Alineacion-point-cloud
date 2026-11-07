# Validación de BlockAlign 0.2 — 2026-11-07

Todas las pruebas y fases numéricas se ejecutaron por Slurm, CPU, con scratch
en `/local`. El control/API se ejecutó desde login sin cargar NumPy/PyCOLMAP.
Se conservaron los originales y el resultado histórico del prototipo.

## Pruebas

Job **61420**: **25 pruebas aprobadas**, 26.53 s de pytest, 29 s de job.
Incluyen los 11 contratos históricos y las pruebas de API, registro, planes,
identidad de caché, scheduler idempotente, matrices, grafo sintético de 25
bloques con enlace erróneo/isla, transformaciones PLY y lineage incompatible.

## Regresión real 005–009

Job **61416**, COMPLETED, **2 min 26 s**:

- Dos bloques auditados y una conexión calculada por el motor genérico.
- 30 cámaras puente registradas.
- Mapa completo, referencia 009 y mundo UTM18S local.
- Versión: `b3b034b40992bb5d926a846076fea190e38b43fe907c1c1925efecd9d3fb934a`.

No se exigió igualdad bit a bit con la selección del experimento histórico;
se aplicaron los controles de consistencia y cámaras reservadas.

## Incorporación real de 001

Job **61419**, COMPLETED, **1 min 39 s**:

- Conservó las auditorías de 005 y 009: dos cache hits.
- Reutilizó la conexión 005–009: un pair cache hit.
- Calculó 001–005, aceptada con 10 cámaras puente.
- Rechazó la propuesta 001–009 por soporte GIS/tracks insuficiente; no forzó
  un vínculo solo por la proximidad geográfica.
- Solución completa para **001, 005 y 009**, sin bloques desconectados.
- Optimización del grafo convergió (`ftol` y `xtol`).
- Versión: `57dbedf5f85e7ac57abfef8bf2956d4a8e41a55cc122ee1b6ad07daac018c3a7`.

Esta incorporación demuestra reutilización e independencia de nombres fijos,
pero no constituye un cierre de ciclo visual real: las dos aristas forman un
árbol. Los ciclos se validaron por ahora en datos sintéticos.

## Exportación de los tres bloques

Job **61421**, COMPLETED, **8 s**:

```text
var/barranco/exports/
  7c6a255d2ebd053470ee399a724e35a7d890e7afc3cd96379bad0e52cb4793a9/
    atlas.json
    combined.ply
```

Contiene **1113792 puntos sparse** colocados en el marco común. Es una
concatenación explícita para inspección, no una densificación ni un modelo de
tracks reconciliado entre los tres bloques. Las procedencias están en el atlas.
La transformación de densos con normales/atributos se probó con fixtures PLY;
todavía no se ensayó una campaña real de densos OpenMVS.

## Corrección genérica por pares

Job **61411**, COMPLETED, **4 min 46 s**, sobre una versión modular anterior
del mismo par 005–009 y las mismas entradas inmutables:

```text
var/barranco/corrections/
  4f10a1ad7d3b5558eefaa6f8de5e39097d888f46b503aef63da8f451a3df916f/
    model/
    corrected_metric.ply
    correction.json
    images.json
    solver.txt
```

- Unión: 1131 imágenes, 880002 puntos.
- 64 cámaras y 2151 puntos variables.
- Reproyección media global: **1.0791 → 1.0120 px**.
- Reproyección media de tracks compartidos: **2.8559 → 2.2134 px**.
- Solución usable con límite de 40 iteraciones: `NO_CONVERGENCE`.
- Auditoría de identidad, tracks y pérdidas de observaciones aprobada.
- Publicó solicitudes de regeneración de los derivados de 005 y 009 sin
  marcar los densos anteriores como corregidos ni modificar el mapa original.

La corrección tiene una versión padre explícita y no se mezcló con el atlas
de colocación de tres bloques. Sus métricas son de reproyección interna, no
precisión topográfica contra referencias independientes.

## Alcance siguiente

Validar un ciclo real con más vecinos; ensayar densos OpenMVS con lineage
conocido; calibrar presupuestos con 8–10 bloques antes de extender a 25.
La documentación de API y las limitaciones están en `modular-api.md`.
