# ✈️ Aircraft Engine Predictive Maintenance

> Sistema de mantenimiento predictivo para motores turbofan con explicabilidad SHAP y análisis de valor de negocio  
> Proyecto Final — Bootcamp de Data Science

---

## 🎯 Problema y objetivo

Las aerolíneas gastan más de 50.000 millones de dólares anuales en mantenimiento de motores. El 30% de ese coste viene de fallos no planificados — motores que fallan sin avisar, aviones que quedan en tierra, vuelos cancelados.

El mantenimiento tradicional funciona por calendario: cada X vuelos, revisas el motor. No importa si está perfecto o a punto de fallar.

**Objetivo:** predecir si un motor turbofan va a fallar en los próximos 30 ciclos operacionales, usando lecturas de sensores en tiempo real — para que el equipo de mantenimiento actúe antes de que ocurra el fallo.

**A quién le sirve:** aerolíneas, empresas MRO (Maintenance, Repair & Overhaul), fabricantes como Boeing, Airbus o Rolls-Royce.

---

## 🖥️ App en producción

La app de Streamlit permite analizar motores en tiempo real:

- **Simulador de sensores** — ajusta los valores de los sensores más críticos y obtén la predicción al instante
- **Predicción con explicación SHAP** — no solo dice si el motor está en riesgo, sino exactamente qué sensores lo causan
- **3 pestañas:** Predicción · Rendimiento del modelo · Valor de negocio

```bash
streamlit run app/app.py
```

---

## 🗂️ Estructura del repositorio

```
aircraft-engine-predictive-maintenance/
├── README.md
├── requirements.txt
├── PLAN_DE_TRABAJO.md
├── PITCH.md
├── .gitignore
├── data/
│   ├── raw/               # Dataset NASA C-MAPSS (train_FD001.txt, etc.)
│   └── processed/         # Datos limpios + gráficos generados
├── notebooks/
│   ├── 01_eda.ipynb           # Análisis exploratorio + diccionario de sensores
│   ├── 02_preprocesamiento.ipynb  # Limpieza, normalización y feature engineering
│   └── 03_modelado.ipynb      # Baseline + XGBoost + Optuna + SHAP + Expected Value
├── src/
│   ├── preprocessing.py   # Funciones de limpieza reutilizables
│   ├── train.py           # Pipeline de entrenamiento
│   └── predict.py         # Predicción sobre nuevos datos
├── models/
│   ├── xgboost_model.pkl  # Modelo entrenado
│   └── feature_names.pkl  # Nombres de las 28 features
└── app/
    └── app.py             # Aplicación Streamlit
```

---

## 📊 Dataset

- **Principal:** [NASA C-MAPSS Turbofan Engine — Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
  Dataset de referencia mundial creado por la NASA. Simula la degradación real de motores turbofan hasta el fallo. 21 sensores físicos por ciclo operacional (temperatura, presión, velocidad de rotación).

- **Plan B:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
  Predicción de riesgo crediticio. 300.000 solicitudes de préstamo con historial financiero completo.

---

## 🧠 Estrategia de Machine Learning

| Etapa | Técnica | Por qué |
|-------|---------|---------|
| Baseline | Regresión Logística | Punto de referencia simple y explicable |
| Modelo principal | XGBoost | Máximo rendimiento en datos tabulares |
| Optimización | Optuna (optimización bayesiana) | Búsqueda automática de hiperparámetros |
| Explicabilidad | SHAP TreeExplainer | Justificar cada predicción individualmente |
| Métricas clave | AUC-ROC, **Recall** | Recall crítico: un fallo no detectado puede costar vidas |

**¿Por qué el Recall es la métrica más importante?**
En aviación, un Falso Negativo (motor en riesgo no detectado) puede costar vidas. Preferimos 8 falsas alarmas antes que dejar pasar un fallo real.

---

## 📈 Resultados

| Modelo | AUC-ROC | Recall | Precision |
|--------|---------|--------|-----------|
| Baseline (Reg. Logística) | 0.9901 | 95.0% | 72.6% |
| XGBoost sin optimizar | 0.9963 | 92.7% | 92.1% |
| **XGBoost + Optuna (final)** | **0.9964** | **93.0%** | **92.4%** |

---

## 💰 Valor de negocio — Expected Value Framework

Basado en la metodología de *Data Science for Business* (Provost & Fawcett):

| Tipo de predicción | Cantidad | Valor unitario | Total |
|-------------------|----------|----------------|-------|
| ✅ Verdaderos Positivos (fallos evitados) | 563 | +$300.000 | +$168.9M |
| ⬜ Verdaderos Negativos | 3.476 | $0 | $0 |
| ⚠️ Falsos Positivos (falsas alarmas) | 51 | -$100.000 | -$5.1M |
| 🔴 Falsos Negativos (no detectados) | 37 | -$500.000 | -$18.5M |
| **TOTAL** | | | **+$145.3M** |

**Ahorro vs no tener sistema predictivo: +$445.3M**

---

## 🏗️ Decisiones técnicas

- **Target binario:** RUL < 30 ciclos → EN RIESGO (umbral de negocio decidido por nosotros)
- **Feature engineering:** medias móviles de 10 ciclos para capturar tendencias de degradación
- **Sensores eliminados:** 10 sensores con varianza cero detectados en el EDA
- **Desbalanceo:** `scale_pos_weight = 5.88` en XGBoost (14.5% de casos en riesgo)
- **Validación:** cross-validation estratificada de 5 folds para resultados robustos
- **Optuna:** 30 pruebas de optimización bayesiana con CV incluida

---

## 🚀 Cómo ejecutar

```bash
# 1. Clonar el repositorio
git clone https://github.com/evely114/aircraft-engine-predictive-maintenance.git
cd aircraft-engine-predictive-maintenance

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el dataset de Kaggle → colocar en data/raw/

# 4. Ejecutar los notebooks en orden
# notebooks/01_eda.ipynb
# notebooks/02_preprocesamiento.ipynb
# notebooks/03_modelado.ipynb

# 5. Lanzar la app
streamlit run app/app.py
```

---

## 👤 Autor

**Evely Adrianza**
Bootcamp Data Science — Módulo 3
[LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/evely114)
