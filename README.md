# ✈️ Aircraft Engine Predictive Maintenance

> Modelo de predicción de fallos en motores de avión con explicabilidad SHAP  
> Proyecto Final — Bootcamp de Data Science

---

## 🎯 Problema y objetivo

Las aerolíneas gastan millones en mantenimiento no planificado. Un motor que falla en vuelo es una catástrofe. El mantenimiento preventivo tradicional cambia piezas por calendario, no por necesidad real.

**Objetivo:** predecir si un motor de avión va a fallar en los próximos 30 ciclos operacionales, usando datos de sensores en tiempo real — para que el equipo de mantenimiento actúe antes de que ocurra el fallo.

**A quién le sirve:** aerolíneas, empresas MRO (Maintenance, Repair & Overhaul), fabricantes como Boeing, Airbus o Rolls-Royce.

---

## 🗂️ Estructura del repositorio

```
aircraft-engine-maintenance/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               # Datos originales NASA C-MAPSS
│   └── processed/         # Datos limpios listos para modelar
├── notebooks/
│   ├── 01_eda.ipynb           # Análisis exploratorio de sensores
│   ├── 02_preprocesamiento.ipynb  # Limpieza y feature engineering
│   └── 03_modelado.ipynb      # Baseline + XGBoost + SHAP
├── src/
│   ├── preprocessing.py   # Funciones de limpieza reutilizables
│   ├── train.py           # Pipeline de entrenamiento
│   └── predict.py         # Predicción sobre nuevos motores
├── models/
│   └── xgboost_model.pkl  # Modelo entrenado
└── app/
    └── app.py             # Aplicación Streamlit
```

---

## 📊 Dataset

- **Principal:** [NASA C-MAPSS Turbofan Engine — Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
  Dataset de referencia mundial creado por la NASA. Simula la degradación real de motores turbofan hasta el fallo. 26 variables de sensores por ciclo operacional.

- **Plan B:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
  Predicción de riesgo crediticio. 300.000 solicitudes de préstamo con historial financiero completo.

---

## 🧠 Estrategia de Machine Learning

### Proyecto principal — Aviación

| Etapa | Técnica | Por qué |
|-------|---------|---------|
| Baseline | Regresión Logística | Punto de referencia simple y explicable |
| Modelo principal | XGBoost | Máximo rendimiento en datos tabulares |
| Optimización | Optuna | Tuning automático de hiperparámetros |
| Explicabilidad | SHAP | Justificar cada predicción individualmente |
| Métricas clave | AUC-ROC, Precision, **Recall** | Recall crítico: un fallo no detectado puede costar vidas |

### Plan B — Riesgo crediticio

| Etapa | Técnica | Por qué |
|-------|---------|---------|
| Baseline | Regresión Logística | Benchmark inicial |
| Modelo principal | XGBoost | Estándar en entornos financieros |
| Explicabilidad | SHAP | Requerido por regulación Basilea III |
| Métricas clave | AUC-ROC, Gini, KS Statistic | Lenguaje estándar en banca |

**¿Por qué el Recall es la métrica más importante en aviación?**
En este problema, un falso negativo (no detectar un fallo real) es catastrófico. Un falso positivo (mantenimiento innecesario) solo cuesta dinero. El modelo está optimizado para minimizar fallos no detectados.

---

## 🚀 Cómo ejecutar

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/aircraft-engine-maintenance.git
cd aircraft-engine-maintenance
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset
Descarga el dataset desde [Kaggle NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) y coloca los archivos en `data/raw/`.

### 4. Ejecutar los notebooks en orden
```
notebooks/01_eda.ipynb
notebooks/02_preprocesamiento.ipynb
notebooks/03_modelado.ipynb
```

### 5. Lanzar la app
```bash
streamlit run app/app.py
```

---

## 📈 Resultados

| Métrica | Baseline (Logística) | XGBoost |
|---------|---------------------|---------|
| AUC-ROC | _pendiente_ | _pendiente_ |
| Recall | _pendiente_ | _pendiente_ |
| Precision | _pendiente_ | _pendiente_ |

> Los resultados se actualizarán una vez completado el entrenamiento.

---

## 🏗️ Decisiones técnicas

- **Definición del target:** un motor se etiqueta como "en riesgo" si le quedan menos de 30 ciclos para el fallo (RUL < 30)
- **Datos desbalanceados:** uso de `scale_pos_weight` en XGBoost para compensar la menor proporción de motores en riesgo
- **Optimización:** Optuna para búsqueda automática de hiperparámetros óptimos
- **Validación:** cross-validation estratificada de 5 folds para resultados robustos
- **Explicabilidad:** SHAP TreeExplainer — qué sensores contribuyen más al riesgo de cada motor

---

## 👤 Autor

**[Tu nombre]**  
Bootcamp Data Science — Módulo 3  
[LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/tu-usuario)
