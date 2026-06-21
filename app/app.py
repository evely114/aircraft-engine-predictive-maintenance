"""
Aircraft Engine Predictive Maintenance — App de Streamlit
============================================================
Predice si un motor de avión va a fallar en los próximos 30 ciclos
usando XGBoost, con explicabilidad SHAP para cada predicción.

Para ejecutar:
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Configuración de la página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# ──────────────────────────────────────────────
# Cargar el modelo entrenado (se cachea para no recargar en cada interacción)
# ──────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('models/xgboost_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return modelo, feature_names

modelo, feature_names = cargar_modelo()

# ──────────────────────────────────────────────
# Cabecera
# ──────────────────────────────────────────────
st.title("✈️ Predictor de Mantenimiento de Motores de Avión")
st.markdown("""
Modelo de **XGBoost + SHAP** entrenado sobre el dataset NASA C-MAPSS Turbofan Engine.  
Predice si un motor va a fallar en los próximos **30 ciclos operacionales**.
""")

# Métricas del modelo (fijas, calculadas en el notebook de modelado)
col1, col2, col3 = st.columns(3)
col1.metric("AUC-ROC", "0.996")
col2.metric("Recall", "93.0%")
col3.metric("Precision", "92.4%")

st.divider()

# ──────────────────────────────────────────────
# Sidebar — Entrada de datos del motor
# ──────────────────────────────────────────────
st.sidebar.header("📊 Datos del motor")
st.sidebar.markdown("Introduce las lecturas de sensores del motor a evaluar.")

modo = st.sidebar.radio(
    "¿Cómo quieres introducir los datos?",
    ["Subir archivo CSV", "Usar un ejemplo de prueba"]
)

datos_motor = None

if modo == "Subir archivo CSV":
    archivo = st.sidebar.file_uploader(
        "Sube un CSV con las lecturas de sensores",
        type=['csv'],
        help="El archivo debe tener las mismas columnas que el dataset de entrenamiento."
    )
    if archivo is not None:
        datos_motor = pd.read_csv(archivo)

else:
    st.sidebar.markdown("Genera un motor de ejemplo con sliders:")
    
    # Sliders simplificados para los sensores más importantes (según SHAP)
    s11 = st.sidebar.slider("Sensor s11 (temperatura escape)", -3.0, 3.0, 0.0, 0.1)
    s12 = st.sidebar.slider("Sensor s12 (presión)", -3.0, 3.0, 0.0, 0.1)
    s4 = st.sidebar.slider("Sensor s4 (temperatura)", -3.0, 3.0, 0.0, 0.1)
    s7 = st.sidebar.slider("Sensor s7 (presión)", -3.0, 3.0, 0.0, 0.1)
    s15 = st.sidebar.slider("Sensor s15 (presión aceite)", -3.0, 3.0, 0.0, 0.1)
    
    # Construimos un registro completo: los sensores elegidos + el resto en 0 (valor medio normalizado)
    fila = {col: 0.0 for col in feature_names}
    fila['s11'] = s11
    fila['s11_media_movil'] = s11
    fila['s12'] = s12
    fila['s12_media_movil'] = s12
    fila['s4'] = s4
    fila['s4_media_movil'] = s4
    fila['s7'] = s7
    fila['s7_media_movil'] = s7
    fila['s15'] = s15
    fila['s15_media_movil'] = s15
    
    datos_motor = pd.DataFrame([fila])

# ──────────────────────────────────────────────
# Predicción y resultados
# ──────────────────────────────────────────────
if datos_motor is not None:
    
    # Aseguramos que las columnas coinciden con las que el modelo espera
    try:
        datos_motor = datos_motor[feature_names]
    except KeyError as e:
        st.error(f"El archivo no tiene las columnas correctas. Faltan: {e}")
        st.stop()

    # Predicción
    proba = modelo.predict_proba(datos_motor)[:, 1]
    prediccion = modelo.predict(datos_motor)

    st.header("🔍 Resultado de la predicción")

    # Si hay varios motores (CSV con múltiples filas), mostramos una tabla resumen
    if len(datos_motor) > 1:
        resumen = pd.DataFrame({
            'Motor (fila)': range(len(datos_motor)),
            'Probabilidad de riesgo': (proba * 100).round(1),
            'Predicción': ['⚠️ EN RIESGO' if p == 1 else '✅ SEGURO' for p in prediccion]
        })
        st.dataframe(resumen, use_container_width=True)
        idx_seleccionado = st.selectbox(
            "Selecciona un motor para ver la explicación SHAP:",
            range(len(datos_motor))
        )
    else:
        idx_seleccionado = 0

    # Resultado del motor seleccionado
    prob_motor = proba[idx_seleccionado] * 100
    es_riesgo = prediccion[idx_seleccionado] == 1

    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        if es_riesgo:
            st.error(f"⚠️ **EN RIESGO**\n\nProbabilidad de fallo: **{prob_motor:.1f}%**")
            st.markdown("**Recomendación:** programar mantenimiento antes de los próximos 30 ciclos.")
        else:
            st.success(f"✅ **SEGURO**\n\nProbabilidad de fallo: **{prob_motor:.1f}%**")
            st.markdown("**Recomendación:** mantenimiento de rutina, sin urgencia.")
        
        st.progress(float(prob_motor / 100))

    with col_der:
        st.markdown("### 🧠 Explicación SHAP — por qué el modelo decidió esto")
        
        # Calculamos los SHAP values para este motor concreto
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(datos_motor)

        fig, ax = plt.subplots(figsize=(8, 5))
        explanation = shap.Explanation(
            values=shap_values[idx_seleccionado],
            base_values=explainer.expected_value,
            data=datos_motor.iloc[idx_seleccionado],
            feature_names=datos_motor.columns.tolist()
        )
        shap.plots.waterfall(explanation, show=False, max_display=8)
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Las barras rojas empujan la predicción hacia 'EN RIESGO'. Las barras azules empujan hacia 'SEGURO'.")

else:
    st.info("👈 Sube un archivo CSV o usa los sliders en la barra lateral para generar una predicción.")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption("""
**Aircraft Engine Predictive Maintenance** · Proyecto Final Bootcamp Data Science  
Modelo: XGBoost optimizado con Optuna · Dataset: NASA C-MAPSS Turbofan Engine
""")
