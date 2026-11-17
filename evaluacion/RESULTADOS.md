# Resultados de la evaluación de la búsqueda

Corpus: 4 documentos de seguros (1.054 fragmentos). Modelo de embeddings: `intfloat/multilingual-e5-base`.
Preguntas: 28 con respuesta y 6 sin respuesta (`preguntas.json`). Cada pregunta con respuesta trae una frase de
evidencia comprobada contra el texto del documento; un fragmento recuperado es un acierto si es de esa fuente y
contiene la frase.

Reproducir: `python -m src.evaluacion` (añade `--detalle` para ver los fallos).

- **acierto@k**: % de preguntas con algún acierto entre los k primeros (k = 5).
- **MRR**: media de 1/posición del primer acierto. Premia poner el fragmento correcto arriba.
- **AUC sin respuesta**: si la puntuación del primer resultado distingue las preguntas con respuesta de las que
  no la tienen (0,5 = no distingue; 1 = las separa).

## Línea base y mejoras (troceado 900 / solape 150)

| Configuración | Modo | acierto@1 | acierto@5 (IC 95 %) | MRR | AUC sin resp. |
|---|---|---|---|---|---|
| Línea base | denso | 36 % | 71 % (53–85) | 0,530 | 0,79 |
| Línea base | BM25 | 29 % | 79 % (60–90) | 0,465 | 0,77 |
| Línea base | híbrido | 54 % | 79 % (60–90) | 0,641 | 0,80 |
| + prefijo del documento | denso | 50 % | 75 % (57–87) | 0,590 | 0,76 |
| + prefijo del documento | BM25 | 39 % | 82 % (64–92) | 0,545 | 0,81 |
| **+ prefijo del documento** | **híbrido** | **57 %** | **86 % (69–94)** | **0,670** | **0,93** |
| + prefijo + cortar por artículos (ley) | híbrido | 54 % | 89 % (73–96) | 0,661 | 0,92 |

## Otros tamaños de fragmento (sin prefijo, modo híbrido)

| Tamaño / solape | Fragmentos | acierto@1 | acierto@5 | MRR |
|---|---|---|---|---|
| 450 / 80 | 2.218 | 46 % | 79 % | 0,583 |
| 650 / 100 | 1.472 | 50 % | 75 % | 0,599 |
| 900 / 150 | 1.054 | 54 % | 79 % | 0,641 |
| 1300 / 200 | 711 | 57 % | 75 % | 0,655 |

## Qué se concluye

- La búsqueda **híbrida** (embeddings + palabras clave mezclados con RRF) mejora al modo denso solo: sube el
  acierto en primera posición del 36 % al 54 % y el MRR de 0,53 a 0,64.
- **Anteponer el nombre del documento** al texto de cada fragmento ayuda cuando la pregunta nombra a la
  aseguradora (Caser, ING, ASISA), porque el fragmento no suele repetir ese nombre. Con él, el híbrido llega al
  86 % y la puntuación del primer resultado empieza a separar las preguntas sin respuesta (AUC 0,93).
- El **tamaño del fragmento** no muestra un efecto claro: las diferencias están dentro del margen de error.
- Cortar por artículos ayuda algo en @5 pero no en MRR, y el patrón es específico de leyes; queda como opción.

## Límites (importante)

- Con 28 preguntas los intervalos de confianza son anchos (unos ±15 puntos). Muchas diferencias entre
  configuraciones **no son estadísticamente concluyentes**; la mejora del híbrido frente al denso sí se mantiene
  en todos los tamaños probados.
- Las preguntas las redacté yo leyendo los documentos, y las optimizaciones se eligieron mirando estas mismas
  preguntas: el 86 % está algo inflado. Falta un segundo conjunto de preguntas, que no se haya usado para ajustar,
  para confirmarlo.
- Esto mide **solo la recuperación** (¿llega el fragmento correcto al agente?). No mide aún si la respuesta
  generada es correcta, si la cita respalda la respuesta ni si el sistema se niega cuando debe.
- Fallos que siguen en el modo final: intereses de demora (q06), definición de joyas de Caser (q12), teléfono de
  asistencia 24 h de Caser (q14) y definición de franquicia (q19).
- Siguiente mejora a probar: reordenar los 20 primeros con un modelo cross-encoder (por ejemplo
  `bge-reranker-v2-m3`), que obliga a descargar otro modelo.

# Evaluación del agente completo

Agente LangGraph (`src/agente.py`) con Gemini `gemini-3.5-flash-lite`, búsqueda híbrida y verificación de la cita.
Reproducir: `python -m src.evaluacion_agente` (tarda ~12 min con el plan gratuito). Todas las respuestas
están en [`respuestas_agente.md`](respuestas_agente.md).

| Medida | Resultado (IC 95 %) |
|---|---|
| Preguntas con respuesta: la búsqueda trae la evidencia | 24/28 = 86 % (69–94) |
| Preguntas con respuesta: el agente contesta | 26/28 = 93 % (77–98) |
| Preguntas con respuesta: cita el fragmento esperado (automático, estricto) | 22/28 = 79 % (60–90) |
| Preguntas sin respuesta: rechaza | 6/6 = 100 % (61–100) |

## Revisión manual de las respuestas (hecha por el autor del proyecto, conviene que otra persona la repita)

- De las 26 respuestas dadas, **las 26 coinciden con lo que dicen los documentos**. Ninguna cita es inventada
  (la verificación literal lo garantiza) y ninguna respuesta contradice el texto citado.
- Los 4 casos en que la métrica estricta dice «cita incorrecta» (q06, q07, q14, q26) son falsos negativos de la
  métrica: el agente citó un pasaje equivalente de **otro documento** (la pregunta no nombraba ninguno) o de otra
  página del mismo. No se ha modificado la métrica para no ajustarla después de ver los resultados.
- Las 2 preguntas con respuesta que rechazó (q12 joyas de Caser, q19 franquicia) coinciden con fallos de la
  búsqueda: el fragmento correcto no llegó al agente. El fallo es **rechazar**, no inventar.
- q13 (robo de joyas en vivienda secundaria) está bien citada pero la respuesta es imprecisa.
- Al revisar apareció un **error en el conjunto de preguntas**: la respuesta esperada de q18 (regla de equidad)
  estaba mal escrita por mí; el agente había citado correctamente la definición de ING. Ya está corregida.

## Límites

- 34 preguntas redactadas por el autor; el modelo y la búsqueda se ajustaron mirando esas mismas preguntas.
- Con temperatura 0 el modelo es casi, no del todo, determinista: una repetición puede dar pequeñas diferencias.
- No se mide aún la calidad del texto de la respuesta con un modelo juez ni con revisión independiente.
