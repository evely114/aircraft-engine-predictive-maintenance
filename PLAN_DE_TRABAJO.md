# 📋 Plan de Trabajo — Aircraft Engine Predictive Maintenance

## Stack definitivo
- **Proyecto principal:** Mantenimiento predictivo motores de avión (NASA C-MAPSS — 4 subconjuntos)
- **Plan B:** Riesgo crediticio (Home Credit Default Risk)
- **Modelo:** Regresión Logística (baseline) → XGBoost + Optuna + SHAP
- **Producto:** Streamlit (3 pestañas: Predicción · Rendimiento · Valor de negocio)

---

## ✅ FASE 0 — Setup inicial

- [x] Crear repositorio en GitHub: `aircraft-engine-predictive-maintenance`
- [x] Estructura de carpetas completa
- [x] README inicial
- [x] Descargar dataset NASA C-MAPSS (FD001, FD002, FD003, FD004)
- [x] Entorno virtual e instalación de dependencias
- [x] Primer commit

---

## ✅ FASE 1 — EDA (notebooks/01_eda.ipynb)

- [x] Cargar FD001 y revisar estructura
- [x] Calcular RUL (Remaining Useful Life)
- [x] Diccionario de sensores con significado físico real
- [x] Visualización de degradación de sensores
- [x] Sensores con varianza cero identificados (10 sensores eliminados)
- [x] Correlación de sensores con el RUL
- [x] Resumen del EDA
- [x] Commit: `feat: EDA notebook completo`

---

## ✅ FASE 2 — Preprocesamiento (notebooks/02_preprocesamiento.ipynb)

- [x] Cargar los 4 subconjuntos (FD001 + FD002 + FD003 + FD004)
- [x] Motor IDs únicos entre subconjuntos (offset 0/1000/2000/3000)
- [x] KMeans para detectar 6 condiciones operativas automáticamente
- [x] Normalización por condición operativa (no global)
- [x] Features de medias móviles (ventana 10 ciclos)
- [x] Target binario: RUL < 30 → EN RIESGO
- [x] Split train/test estratificado (80/20)
- [x] Guardar datos procesados en data/processed/
- [x] Commit: `feat: preprocessing with 4 subsets, KMeans, condition normalization`

**Resultado:** 709 motores · 160.359 registros · 6 condiciones · 28 features

---

## ✅ FASE 3 — Modelado (notebooks/03_modelado.ipynb)

- [x] Cargar datos procesados
- [x] Baseline — Regresión Logística (AUC-ROC: 0.9881)
- [x] XGBoost con scale_pos_weight (AUC-ROC: 0.9922)
- [x] Optuna — optimización bayesiana (30 pruebas · CV 5 folds)
- [x] Modelo final XGBoost + Optuna (AUC-ROC: 0.9936, Recall: 91.5%)
- [x] Gráfico comparativo de modelos
- [x] Curva ROC y matriz de confusión
- [x] SHAP — feature importance global (beeswarm)
- [x] SHAP — explicación individual (waterfall)
- [x] Expected Value Framework (+$145.3M valor neto)
- [x] Guardar modelo en models/xgboost_model.pkl
- [x] Commit: `feat: modeling complete - XGBoost + Optuna + SHAP + Expected Value`

---

## ✅ FASE 4 — App Streamlit (app/app.py)

- [x] Diseño oscuro profesional (tema aeronáutico)
- [x] Pestaña 1 — Predicción:
  - [x] Botones de escenario rápido (Motor nuevo / Motor en riesgo)
  - [x] Sliders de 5 sensores críticos con nombres descriptivos
  - [x] Resultado visual EN RIESGO / SEGURO con probabilidad
  - [x] Sensores críticos en rojo cuando hay riesgo
  - [x] Recomendaciones en rojo cuando hay alerta
  - [x] Gráfico SHAP waterfall individual
  - [x] Nota explicativa sobre medias móviles
- [x] Pestaña 2 — Rendimiento del modelo:
  - [x] Métricas AUC-ROC, Recall, Precision, F1
  - [x] Comparativa de los 3 modelos
  - [x] Matriz de confusión con columna resaltada según predicción actual
  - [x] Stack tecnológico
- [x] Pestaña 3 — Valor de negocio:
  - [x] Expected Value +$145.3M
  - [x] Tabla por categoría
  - [x] Gráfico de barras económico
  - [x] Pitch listo para copiar
- [x] Commit: `feat: streamlit app complete - dark theme, 3 tabs, SHAP, Expected Value`

---

## ✅ FASE 5 — Documentación

- [x] README completo con descripción de C-MAPSS
- [x] Componentes del motor y sus sensores
- [x] Métricas actualizadas (4 subconjuntos)
- [x] PITCH.md con estructura 7-10 minutos
- [x] Preguntas frecuentes de entrevista
- [x] PLAN_DE_TRABAJO.md actualizado
- [x] Commit: `docs: complete documentation`

---

## ⏳ FASE 6 — Presentación (jueves)

- [ ] Slides PDF listos (presentacion_aeropredict.pdf — 12 slides)
- [ ] Practicar pitch 2-3 veces en voz alta
- [ ] Demo de la app probada (botón "Motor en riesgo" funciona)
- [ ] GitHub abierto y visible
- [ ] Números memorizados (ver abajo)

---

## 🔢 Números clave para memorizar

| Dato | Valor |
|------|-------|
| Motores en entrenamiento | 709 |
| Registros totales | 160.359 |
| Condiciones operativas | 6 |
| Subconjuntos usados | 4 (FD001+FD002+FD003+FD004) |
| Features del modelo | 28 |
| AUC-ROC final | 0.9936 |
| Recall | 91.5% |
| Precision | 86.8% |
| Fallos detectados | 563 de 600 |
| Valor neto del modelo | +$145.3M |
| Ahorro vs sin modelo | +$445.3M |
| Umbral de riesgo | RUL < 30 ciclos |

---

## 💡 Tips para la presentación del jueves

1. **Abre con el problema humano** — "$50B en mantenimiento, 30% por fallos no planificados"
2. **Demo en vivo primero** — muestra la app antes de explicar el código
3. **Explica el trade-off** — "las métricas bajaron ligeramente al usar 4 subconjuntos porque el problema es más complejo, pero el modelo es mucho más robusto"
4. **Muestra el SHAP** — "s4 lleva ciclos subiendo anormalmente — eso es lo que ve el técnico"
5. **Cierra con dinero** — "$145 millones de valor generado"
6. **No hables de código** — habla de negocio, sensores físicos y valor económico
