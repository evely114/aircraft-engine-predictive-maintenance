# 📋 Plan de Trabajo — Aircraft Engine Predictive Maintenance

## Stack definitivo
- **Proyecto principal:** Mantenimiento predictivo motores de avión (NASA C-MAPSS)
- **Plan B:** Riesgo crediticio (Home Credit Default Risk)
- **Modelo:** Regresión Logística (baseline) → XGBoost + Optuna + SHAP
- **Producto:** Streamlit

---

## ✅ FASE 0 — Setup inicial (HOY)

- [ ] Crear repositorio en GitHub: `aircraft-engine-maintenance`
- [ ] Subir estructura de carpetas, README y requirements.txt
- [ ] Descargar dataset NASA C-MAPSS de Kaggle → `data/raw/`
- [ ] Descargar dataset Home Credit (plan B) → guardarlo aparte
- [ ] Crear entorno virtual e instalar requirements
- [ ] Primer commit: `feat: project structure and README`

---

## 📊 FASE 1 — EDA (notebooks/01_eda.ipynb)

- [ ] Cargar los 4 archivos del dataset (train_FD001 a train_FD004)
- [ ] Entender la estructura: motores, ciclos, sensores
- [ ] Calcular el RUL (Remaining Useful Life) de cada motor
- [ ] Visualizar la degradación de sensores a lo largo del tiempo
- [ ] Identificar qué sensores son más informativos (correlación con RUL)
- [ ] Detectar sensores con varianza cero (inútiles, hay varios)
- [ ] Al menos 6 gráficos con conclusiones en Markdown
- [ ] Commit: `feat: EDA notebook completo`

---

## 🔧 FASE 2 — Preprocesamiento (notebooks/02_preprocesamiento.ipynb)

- [ ] Crear la variable objetivo: `target = 1 si RUL < 30, else 0`
- [ ] Eliminar sensores con varianza cero
- [ ] Normalizar lecturas de sensores (StandardScaler)
- [ ] Feature engineering: medias móviles de sensores (tendencia de degradación)
- [ ] Split train/test estratificado (80/20)
- [ ] Guardar datos procesados en `data/processed/`
- [ ] Commit: `feat: preprocessing and feature engineering`

---

## 🧠 FASE 3 — Modelado (notebooks/03_modelado.ipynb)

### Baseline
- [ ] Entrenar Regresión Logística
- [ ] Evaluar: AUC-ROC, Precision, Recall
- [ ] Anotar resultados como referencia

### Modelo principal
- [ ] Entrenar XGBoost con parámetros por defecto
- [ ] Comparar con baseline
- [ ] ⚡ MAGIA NEGRA: Optuna para optimización de hiperparámetros
- [ ] Cross-validation estratificada de 5 folds
- [ ] Curva ROC y matriz de confusión
- [ ] ⚡ MAGIA NEGRA: SHAP — feature importance global (beeswarm plot)
- [ ] ⚡ MAGIA NEGRA: SHAP — explicación individual de un motor en riesgo
- [ ] Guardar modelo en `models/xgboost_model.pkl`
- [ ] Commit: `feat: baseline + XGBoost + Optuna + SHAP`

---

## 🖥️ FASE 4 — App Streamlit (app/app.py)

- [ ] Subir CSV de un motor nuevo → predicción en tiempo real
- [ ] Indicador visual: motor OK / en riesgo (con probabilidad)
- [ ] Gráfico de degradación de sensores del motor analizado
- [ ] ⚡ MAGIA NEGRA: SHAP waterfall plot — por qué este motor está en riesgo
- [ ] README actualizado con captura de pantalla
- [ ] Commit: `feat: streamlit app working`

---

## 🎤 FASE 5 — Presentación

- [ ] Slide 1: El problema — coste de un fallo de motor no detectado
- [ ] Slide 2: Los datos — qué son los sensores NASA C-MAPSS
- [ ] Slide 3: El modelo — baseline vs XGBoost, por qué el Recall es clave
- [ ] Slide 4: Demo en vivo de la app
- [ ] Slide 5: Próximos pasos (tiempo real, más modelos, API)

---

## ⚡ Momentos de magia negra (presta atención aquí)

| Momento | Técnica | Por qué es avanzado |
|---------|---------|---------------------|
| Notebook 02 | Medias móviles como features | Captura tendencias temporales de degradación |
| Notebook 03 | Optuna | Optimización bayesiana automática de hiperparámetros |
| Notebook 03 | Cross-validation estratificada | Evaluación robusta sin data leakage |
| Notebook 03 | SHAP beeswarm | Feature importance global con dirección del efecto |
| Notebook 03 | SHAP waterfall | Explicación individual motor a motor |
| App | SHAP en tiempo real | Explicabilidad en producción |

---

## 🗓️ Estimación de tiempos

| Fase | Tiempo estimado |
|------|----------------|
| Setup inicial | 1-2 horas |
| EDA | 3-4 horas |
| Preprocesamiento | 3-4 horas |
| Modelado + SHAP + Optuna | 5-7 horas |
| App Streamlit | 3-4 horas |
| Presentación | 2-3 horas |
| **Total** | **~20 horas** |

---

## 💡 Tips para la presentación

1. **Abre con el problema humano** — "Un fallo de motor en vuelo puede costar vidas y millones"
2. **Explica el Recall** — "Preferimos hacer mantenimiento de más que dejar pasar un fallo real"
3. **Muestra el SHAP** — "Este motor está en riesgo porque el sensor 11 lleva 20 ciclos degradándose"
4. **Demo en vivo** — sube un CSV y muestra la predicción en tiempo real
5. **Habla de negocio** — "Cada fallo no detectado cuesta X€ en AOG (Aircraft on Ground)"
