# Experimento independiente: alineación 005 → 009

## API modular implementada: BlockAlign 0.2

La API nueva permite registrar bloques, crear planes inmutables, alinear pares
sin IDs de escena incrustados, resolver un grafo, agregar bloques con caché y
exportar un atlas o PLY. La corrección local es una operación derivada explícita
que declara solicitudes de regeneración de densos.

**Uso y contratos:** [`docs/modular-api.md`](docs/modular-api.md).
**Validación real:** [`docs/validation-modular.md`](docs/validation-modular.md):
25 pruebas aprobadas, mapa incremental 001–005–009 y exportación de 1113792
puntos sparse alineados. Incluye tiempos, versiones y límites del ensayo.
El código del prototipo sigue disponible bajo `alignexp` como referencia.
Los datos operativos nuevos se guardan en un workspace separado (`var/`, fuera
de Git). Los resultados históricos descritos debajo corresponden al prototipo.

La propuesta para convertir el prototipo en un pipeline modular incremental,
compatible con reconstrucciones sparse y densos OpenMVS, está en
[`docs/architecture.md`](docs/architecture.md). Es una arquitectura planificada;
el documento describe también etapas posteriores a la primera API modular.

## Resultado vigente

Prototipo ejecutado por Slurm el 2026-11-07 y publicado en
**`results/prototype-61384/`**. Consultar `results/LATEST.md`.

- 30/32 cámaras puente registradas; 23 de ajuste y 7 reservadas para E1.
- Discrepancia de centros reservados: 0.13–0.70 m, rotación <0.28°.
- Modelo fusionado/refinado: 1131 imágenes y 880155 puntos.
- Reproyección media global: 1.0710 → 1.0087 px.
- Reproyección de tracks compartidos: 2.6724 → 2.0984 px.
- Distancia mediana a vecinos de 009 dentro de ROI: GPS 1.304 m,
  visual 0.372 m; diagnóstico de consistencia, no exactitud topográfica.
- BA acotado: 64 cámaras y 2351 puntos variables; 40 iteraciones, solución
  usable con límite de iteraciones alcanzado, sin convergencia completa.
- 11 pruebas de contratos pasaron en Slurm. Se verificaron de nuevo los
  hashes de 1188 archivos de entrada, sin cambios.

Abrir `sparse_metric.ply` para la nube final o `E1_visual_colored.ply` para
comparar las dos procedencias (005 azul, 009 naranja). El modelo nativo está
en `model_colmap_ref009/`; su raíz de imágenes es `images/` del resultado.
El modelo nativo y el PLY métrico tienen marcos distintos, documentados en
`coordinates.json`. Mantener el proyecto completo para resolver los enlaces
internos de imágenes; no dependen del pipeline anterior.

Prototipo de unión de dos reconstrucciones COLMAP existentes mediante prior
GPS, selección GIS por puntos y tracks, matching SIFT cruzado, PnP, Sim(3),
fusión de tracks y corrección fotogramétrica local. No vuelve a ejecutar SfM.

## Ejecución y almacenamiento

Todas las fases numéricas y pruebas requieren `SLURM_JOB_ID`. El login se usa
para edición, instalación y envío/monitorización. `slurm/phase.slurm` crea un
directorio temporal propio en `/local/$USER/alignment005009/$SLURM_JOB_ID`.
Las BD modificables se copian allí; los resultados se conservan en `work/`.
Las entradas persistentes se leen desde `inputs/`, con imágenes reales y sin
dependencias de las cachés del proyecto original. Los modelos originales del
pipeline no se modifican. No hace falta GPU para esta configuración.

```bash
# Desde la raíz de este proyecto, con recursos explícitos por fase:
sbatch --partition=standard --ntasks=1 --cpus-per-task=8 --mem=16G \
  --time=02:00:00 --output=logs/%x-%j.out --error=logs/%x-%j.err \
  slurm/phase.slurm matching
```

Orden: doctor → snapshot → geography → identities → selection → matching →
localization → fusion → refinement → report. Cada fase publica su puntero en
`.state/` solo después de pasar sus controles. Los intentos fallidos conservan
sus diagnósticos, sin publicarse como resultados aceptados. Antes de repetir
una fase, comprobar dependencias y no ejecutar dos cadenas simultáneamente.
El runner guarda copia de código y configuración por job en
`results/provenance/`. El snapshot se crea una sola vez y sus archivos se dejan
en solo lectura. Su inventario contiene hashes de modelos, BD e imágenes.

## Delimitación del piloto

Las capturas 005 y 009 no son códigos catastrales. Se eligieron núcleos de
parcelas compatibles con el mapa provisional y el soporte de las nubes:

- 005: `MZ-BAR-FD7E0473D4`, 23 lotes.
- 009: `MZ-BAR-C440DB7CFC`, 23 lotes.

Son agrupaciones geométricas, no manzanas municipales certificadas. Se conserva
la lista de lotes y se usa un buffer de 15 m para selección. La nube no se
recorta: las observaciones fuera del núcleo sirven para el solapamiento.
GIS operativo EPSG:32718, con origen local registrado en `geography.json`;
los GeoJSON de visualización se exportan a longitud/latitud WGS84.

## Contratos de identidad y calibración

Referencia: 009. Fuente: 005. Las identidades de imágenes 009 reciben offset
1000000 y cámaras/rigs offset 1000; las features no se reordenan. Los frames
triviales se normalizan a IDs de imagen. Se conservan mapas de puntos.
**Los intrínsecos se leen de `cameras.bin` optimizado, no de los priors de la
BD.** El remapeo debe conservar la reproyección y la calibración nativa.
Las imágenes puente se añaden al destino con poses y observaciones verificadas.

## Entorno

`containers/alignment313.sif`, basado en `sparse313.sif`, con COLMAP y PyCOLMAP
3.13.0. La construcción e instalación se hacen en login; las pruebas por Slurm.
Las versiones resueltas se guardan dentro de la imagen en
`/opt/alignment/requirements-resolved.txt`. Las fuentes y hashes de entradas
están en `inputs/manifest.json`.

## Resultados y límites

- E0: PLY coloreado con alineamiento GPS.
- E1: PLY coloreado con Sim(3) visual y cámaras reservadas para validar el ajuste.
- E2: modelo nativo fusionado, tracks y pérdidas auditados.
- E3: BA acotado de la unión, con intrínsecos y contexto fijos.
- `report`: evaluación JSON, comparación gráfica y resumen de artefactos.

La validación reservada de cámaras evalúa E1; el merger usa luego todos los
puentes aceptados. Las distancias entre nubes y el error de reproyección son
diagnósticos de consistencia, no precisión topográfica certificada.

## Historial de diagnóstico

Los primeros registros cruzados fallaron la validación y no produjeron una
fusión aceptada. Se detectó que el primer remapeador copiaba intrínsecos de la
BD (priors) en lugar de los refinados del modelo; se corrigió el origen y se
añadió una validación de reproyección real. Un diagnóstico intermedio convirtió
mal las unidades (división en lugar de multiplicación): su conclusión de
aproximadamente 1 m era incorrecta y queda descartada. No se relajan los gates
de alineación para publicar un resultado fallido.

La primera auditoría de fusión también exigía erróneamente una sola feature por
imagen y punto. La comprobación de los originales mostró 147524 observaciones
adicionales de distintas features de la misma imagen, sin elementos idénticos
duplicados. La auditoría corregida valida reciprocidad, ausencia de elementos
idénticos duplicados, no aumento de multiplicidad respecto a los originales y
pérdidas de observaciones cuantificadas. No elimina features originales para
hacer pasar artificialmente esa comprobación.
