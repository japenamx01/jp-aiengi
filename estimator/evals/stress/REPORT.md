# Stress test del CAG — Reporte (sesión 6)

> Baseline cuantitativo del CAG conversacional de la sesión 5 antes de pasar a
> RAG. Todos los números salen de `evals/stress/results.csv` (900 filas) y se
> regeneran con `uv run python -m evals.stress.analyze evals/stress/results.csv`.

## 0. Parámetros de la corrida

| Ajuste | Valor |
|---|---|
| Fecha | 2026-11-17 |
| Modelo | `gpt-4o-mini-2024-07-18` en el 100 % de las llamadas (el fallback no se activó) |
| Escenarios | `growing`, `pivot`, `contradiction` (20 turnos cada uno) |
| Tamaños de adjunto | 0, 5, 20, 50, 100 KB de texto extraído (PDF, solo en el turno 1) |
| Repeticiones por celda | 3 → 45 sesiones, **900 turnos, 0 errores** |
| Ventana | `MAX_TURNS=6` (turno en curso + 5 anteriores) |
| Recorte de adjuntos | `MAX_ATTACHMENT_CHARS=20.000` |
| Presupuestos | latencia LLM ≤ 8000 ms · coste por turno ≤ 0,005 USD |
| Concurrencia | 6 sesiones en paralelo (cada sesión es secuencial) |
| Duración / coste total | 37 min · **1,46 USD** (el extractor de metadata = 16 %) |

**Qué mide este CAG y qué no tiene.** A diferencia de la versión de la clase,
este servicio no tiene anclas, summarizer, tier dinámico ni caché semántica. El
evento `turn_observed` mantiene los 13 campos del enunciado
(`anchors_count = summary_chars = 0`, `last_resolved_tier = None`) para que el
CSV sea comparable, y añade `turn_cost_usd` (estimación + extractor) y
`turn_latency_ms`. Un hecho del turno *k* solo sobrevive al turno *N* por tres
vías:

1. su turno sigue en la ventana (*N − k* ≤ 5);
2. el extractor lo guardó en `ProjectMetadata`, que solo tiene nombre, equipo,
   tecnologías y alcance;
3. una respuesta posterior lo repite.

**Decisiones del runner** (justificación en `run.py`):

- La observación viaja en la respuesta, no se parsea del log.
- El turno 1 lleva una referencia única por sesión (*cache-bust*); sin ella, las
  repeticiones 2 y 3 serían aciertos de la caché exacta con latencia y coste ≈ 0.
- Las métricas viven en `evals/stress/metrics.py` porque leen
  `TurnObservation` + el snapshot de `GET /sessions/{id}`, no un
  `EstimationResult`, y este repo no tiene todavía `evals/metrics.py`.

## 1. Tabla resumen (escenario × tamaño de adjunto)

`n` = 60 turnos por celda (3 sesiones × 20). La latencia es la de la llamada de
estimación medida en el servidor. *Recall* es la fracción de los hechos dichos
antes del turno N que siguen en la sesión (match exacto, case-insensitive).

| Escenario | KB | P50 ms | P95 ms | $ / sesión | Hit exacta | Hit semántica | Latencia ≤ 8 s | Nombre del turno 1 | Recall (ventana + memoria) | Recall (solo memoria) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contradiction | 0 | 8791 | 18441 | 0.0293 | 0 % | n/a | 42 % | 100 % | 0.91 | 0.41 |
| contradiction | 5 | 7742 | 27016 | 0.0278 | 0 % | n/a | 52 % | 100 % | 0.91 | 0.39 |
| contradiction | 20 | 9688 | 21663 | 0.0364 | 0 % | n/a | 27 % | 100 % | 0.90 | 0.52 |
| contradiction | 50 | 9507 | 17159 | 0.0341 | 0 % | n/a | 23 % | 100 % | 0.88 | 0.42 |
| contradiction | 100 | 9956 | 61001 | 0.0381 | 0 % | n/a | 17 % | 100 % | 0.89 | 0.41 |
| growing | 0 | 11492 | 19945 | 0.0312 | 0 % | n/a | 10 % | 100 % | 0.92 | 0.59 |
| growing | 5 | 13927 | 56994 | 0.0391 | 0 % | n/a | 2 % | 100 % | 0.91 | 0.58 |
| growing | 20 | 12148 | 20902 | 0.0381 | 0 % | n/a | 3 % | 100 % | 0.90 | 0.61 |
| growing | 50 | 11814 | 22679 | 0.0376 | 0 % | n/a | 10 % | 100 % | 0.90 | 0.58 |
| growing | 100 | 14423 | 22669 | 0.0432 | 0 % | n/a | 2 % | 100 % | 0.90 | 0.59 |
| pivot | 0 | 8650 | 15324 | 0.0249 | 0 % | n/a | 43 % | 100 % | 0.94 | 0.71 |
| pivot | 5 | 7639 | 10287 | 0.0247 | 0 % | n/a | 62 % | 100 % | 0.94 | 0.82 |
| pivot | 20 | 8185 | 17105 | 0.0285 | 0 % | n/a | 47 % | 100 % | 0.88 | 0.63 |
| pivot | 50 | 6495 | 13729 | 0.0252 | 0 % | n/a | 78 % | 100 % | 0.94 | 0.80 |
| pivot | 100 | 6918 | 12277 | 0.0272 | 0 % | n/a | 73 % | 100 % | 0.95 | 0.81 |

**Cómo leer las columnas de caché y presupuestos:**

- **Hit exacta = 0 %** es intencional (*cache-bust*). Además, la clave cubre todo
  el array `messages`, así que en una conversación real solo acierta si se
  reproduce la conversación literal. La caché semántica no existe en este CAG.
- **Presupuesto de coste:** no falló ningún turno (máximo 0,0039 USD).
- **Presupuesto de latencia:** solo lo cumple el 33 % de los turnos.
- **Colas de latencia:** 17 de 900 llamadas (1,9 %) tardaron entre 40 y 87 s, con
  `tokens_in` normales. Es compatible con un timeout de 30 s seguido de un
  reintento del Router (`LLM_TIMEOUT=30`, `LLM_RETRIES=2`); no tiene relación con
  el tamaño del contexto. Esas colas son las que disparan los P95 de 50–60 s.

## 2. Las tres curvas (en tabla)

### 2a. Latencia vs tamaño de contexto

Todas las filas (n = 900), agrupadas por `tokens_in`:

| `tokens_in` | n | Mediana ms | P95 ms | Mediana `tokens_out` | Mediana ms / token de salida |
|---|---:|---:|---:|---:|---:|
| 0–2999 | 40 | 6097 | 9941 | 474 | 13.1 |
| 3000–4999 | 258 | 7306 | 12277 | 544 | 13.1 |
| 5000–6999 | 371 | 10690 | 18512 | 831 | 12.7 |
| 7000–9999 | 219 | 13500 | 23089 | 888 | 13.1 |
| 10000+ | 12 | 13397 | 60324 | 1490 | 13.6 |

Correlación de Pearson, sin las 17 colas de más de 30 s:

| Par | r |
|---|---:|
| latencia ↔ `tokens_out` | 0.84 |
| latencia ↔ `tokens_in` | 0.55 |

El coste por token generado es plano (≈ 13 ms) en todo el rango. La latencia
sube con el contexto sobre todo porque, con más contexto, el modelo escribe
estimaciones más largas.

### 2b. Coste vs turno (0 KB, media de 3 sesiones, USD)

| Turno | growing acumulado | growing del turno | pivot acumulado | pivot del turno | contradiction acumulado | contradiction del turno |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0007 | 0.00066 | 0.0006 | 0.00063 | 0.0006 | 0.00060 |
| 3 | 0.0024 | 0.00094 | 0.0023 | 0.00089 | 0.0021 | 0.00079 |
| 6 | 0.0062 | 0.00141 | 0.0055 | 0.00113 | 0.0052 | 0.00119 |
| 7 | 0.0076 | 0.00147 | 0.0068 | 0.00123 | 0.0065 | 0.00129 |
| 10 | 0.0125 | 0.00168 | 0.0108 | 0.00141 | 0.0107 | 0.00149 |
| 15 | 0.0217 | 0.00193 | 0.0179 | 0.00145 | 0.0191 | 0.00185 |
| 20 | 0.0312 | 0.00181 | 0.0249 | 0.00140 | 0.0293 | 0.00213 |

Contexto que viaja en cada turno (media de `tokens_in`):

| Turno | 0 KB | 5 KB | 20 KB | 50 KB | 100 KB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1491 | 2753 | 5990 | 5990 | 5990 |
| 6 | 4374 | 6143 | 9456 | 9234 | 9617 |
| 7 | 4562 | 4992 | 5029 | 4793 | 5255 |
| 10 | 5243 | 5408 | 5479 | 5205 | 5901 |
| 20 | 6531 | 6522 | 6420 | 5863 | 6626 |

Hay dos regímenes:

- **Turnos 1–6:** el contexto crece unos 580 tokens por turno, así que el coste
  acumulado crece de forma cuadrática.
- **Desde el turno 7:** la ventana está llena y el contexto solo crece unos 150
  tokens por turno (respuestas más largas y metadata más grande), así que el coste
  acumulado se vuelve casi lineal.

Con adjunto, la curva es un **diente de sierra**: el turno 7 expulsa el turno 1
con sus 20.000 caracteres (9617 → 5255 tokens en la celda de 100 KB).

### 2c. Recall de hechos vs N (0 KB, 3 escenarios × 3 repeticiones, n = 9 por fila)

| Turno N | Hechos dichos antes de N | Nombre del turno 1 | Recall ventana + memoria | Solo ventana | Solo memoria (`ProjectMetadata`) |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 100 % | 1.00 | 1.00 | 1.00 |
| 6 | 5 | 100 % | 1.00 | 1.00 | 0.69 |
| 7 | 6 | 100 % | 1.00 | 1.00 | 0.72 |
| 8 | 7 | 100 % | 0.98 | 0.98 | 0.66 |
| 10 | 8 | 100 % | 0.95 | 0.94 | 0.55 |
| 12 | 10 | 100 % | 0.95 | 0.93 | 0.43 |
| 15 | 13 | 100 % | 0.87 | 0.83 | 0.49 |
| 20 | 18 | 100 % | 0.81 | 0.65 | 0.38 |

Recall (ventana + memoria) por escenario en el turno 20: growing 0.82,
pivot 0.83, contradiction 0.76.

### Casos límite: contradicciones, cambios de stack y adjuntos

**Cambio de hecho** (% de las 15 sesiones de cada escenario):

| Escenario | Turno | Hecho viejo sigue en memoria | Hecho nuevo en la sesión | Hecho nuevo en la respuesta |
|---|---:|---:|---:|---:|
| pivot (React Native → Flutter) | 5 | 100 % | 100 % | 100 % |
| pivot | 20 | **100 %** | 100 % | 27 % |
| contradiction (30.000 → 80.000 EUR) | 8 | 0 % | 100 % | 100 % |
| contradiction | 14 | 0 % | 100 % | 53 % |
| contradiction | 15 | 0 % | **73 %** | 47 % |
| contradiction | 20 | 0 % | **53 %** | 47 % |

**Adjuntos.** El marcador de cabecera está en la primera línea del PDF y el de
cola en la última.

| KB | Caracteres extraídos | `tokens_in` turno 1 | Cabecera en respuesta t1 | Cola en respuesta t1 | Cabecera en sesión t6 | Cabecera en sesión t7 | Cabecera en sesión t20 | Cabecera en respuesta t20 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5515 | 2753 | 78 % | 100 % | 100 % | 78 % | 78 % | 11 % |
| 20 | 20031 | 5990 | 78 % | **0 %** | 100 % | 78 % | 67 % | 0 % |
| 50 | 20031 | 5990 | 89 % | **0 %** | 100 % | 78 % | 78 % | 0 % |
| 100 | 20031 | 5990 | 100 % | **0 %** | 100 % | 100 % | 89 % | 22 % |

## 3. Lectura: dónde empieza a romperse mi CAG

**La degradación dominante es la pérdida silenciosa de memoria, no el coste ni
la latencia.**

- **El nombre del proyecto nunca se pierde** (100 % en los 900 turnos), porque
  vive en `ProjectMetadata`.
- **El resto de hechos aguanta mientras su turno está en la ventana.** El recall
  es 1.00 hasta el turno 7. Cae por primera vez en el turno 8 (0.98) y baja a
  0.87 en el turno 15 y a 0.81 en el 20.
- **Sin la ventana, la memoria solo retiene el 38 % de los hechos** en el turno 20.
  El 81 % se sostiene porque las respuestas del modelo van repitiendo requisitos
  antiguos, no por diseño.
- **Lo que no encaja en el esquema de `ProjectMetadata` muere con la ventana:**
  - **El presupuesto.** Los 80.000 EUR del turno 9 siguen en la sesión el 100 %
    de las veces hasta el turno 14. En el turno 15 (justo cuando el turno 9 sale
    de la ventana) bajan al 73 %, y en el 20 al 53 %.
  - **El contenido de los adjuntos.** El requisito del HSM aparece en el 0–22 % de
    las respuestas del turno 20.
- **Fallos que ningún test detecta:**
  - En `pivot`, `mentioned_technologies` conserva *React Native* en el 100 % de las
    sesiones después de descartarlo: la memoria dice que el stack incluye ambos
    frameworks.
  - Con 20, 50 y 100 KB el extractor recorta a 20.031 caracteres y el requisito
    final del documento nunca llega al modelo (0 % en la respuesta). Las celdas de
    50 y 100 KB son idénticas a la de 20 KB.
- **El coste no es el problema a esta escala.** El turno 20 cuesta 2,2–3,6× lo que
  el turno 1 y una sesión completa 0,025–0,043 USD.
- **La latencia falla el SLA desde el turno 1**, pero por la longitud de la
  respuesta (≈ 13 ms por token de salida), no por el contexto.

**Qué implica para RAG.** El CAG empieza a romperse en el turno 8 y es claramente
insuficiente entre los turnos 10 y 15. El caso límite que justifica saltar a RAG
es **un hecho que está fuera del esquema de la metadata y se dijo hace más de 5
turnos**: presupuesto, requisitos no funcionales o contenido de adjuntos. Hoy
solo sobrevive si el modelo, por casualidad, lo repite.

RAG cubriría los dos agujeros medidos:

1. **Recuperar turnos antiguos.** Buscar "¿qué presupuesto acordamos?" sobre el
   historial completo eliminaría la caída de 1.00 a 0.53 del hecho actualizado.
2. **Trocear adjuntos en vez de truncarlos.** Hoy se pierde todo lo que pasa de
   los 20.000 caracteres (cola en la respuesta: 0 % desde 20 KB). Con RAG, un
   documento de 100 KB costaría lo mismo que uno de 5 KB, unos 2.750 `tokens_in`
   frente a los 5.990 actuales, recuperando solo los fragmentos relevantes.

El argumento de RAG aquí no es el coste, que es bajo, sino el recall. Hay que
compararlo contra este baseline:

- recall de hechos en el turno 20 = 0.81 (ventana + memoria) y 0.38 (solo memoria);
- recall del hecho actualizado = 53 %;
- recall del final de un adjunto de más de 20 KB = 0 %.

Además, cualquier mejora de memoria tiene que deduplicar hechos sustituidos,
como *React Native* en el escenario `pivot`.

## 4. Cuatro afirmaciones que defiendo

1. Mi CAG empieza a degradarse entre los turnos 8 y 15 (recall 0.98 → 0.87 →
   0.81 en el 20) porque `MAX_TURNS=6` expulsa los turnos literales y
   `ProjectMetadata` solo retiene el 38 % de los hechos (no tiene campos para
   presupuesto, requisitos no funcionales ni adjuntos).
2. El coste por turno crece en O(turnos) hasta llenar la ventana (+580
   `tokens_in` por turno en los turnos 1–6) y casi plano después (+150 por
   turno), porque cada turno reinyecta system + metadata + 5 turnos completos.
   El turno 20 cuesta entre 2,2× (pivot) y 3,6× (contradiction) lo que el
   turno 1.
3. El cuello de botella de la latencia es la generación, no el contexto:
   ≈ 13 ms por token de salida en todos los tramos; r = 0.84 con `tokens_out`
   frente a 0.55 con `tokens_in`. Las colas de 40–87 s (1,9 %) son
   timeouts/reintentos del proveedor.
4. Para recortar el contexto un 50 % sin perder recall, atacaría primero el
   texto de los adjuntos que se queda en el historial. En el turno 6 aporta unos
   5.200 de los 9.600 `tokens_in` (55 %) y en el turno 20 casi nunca llega a la
   respuesta (0–22 %), así que su contribución al recall tardío es mínima.
   Después, el system prompt (≈ 1.400 tokens fijos por turno).

## 5. Reproducibilidad

```bash
cd estimator
docker compose up -d && curl -sf http://localhost:8000/health
uv run python -m evals.stress.fixtures.build_pdfs      # PDFs deterministas (gitignored)
uv run python -m evals.stress.run --http http://localhost:8000 \
    --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 \
    --repeats 3 --concurrency 6 --latency-budget-ms 8000 --cost-budget-usd 0.005 \
    --output evals/stress/results.csv
uv run python -m evals.stress.analyze evals/stress/results.csv   # tablas de este reporte
uv run pytest tests/test_stress_metrics.py tests/test_stress_runner.py
```

El `results.csv` de esta corrida se comitea junto al reporte: es el segundo
entregable. Una nueva corrida dará números distintos, porque el modelo no es
determinista, pero con la misma forma.
