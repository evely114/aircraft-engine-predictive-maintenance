"""
Vuelca los datos COMPLETOS del motor #69 tal y como los calcula app.py,
tanto para la pestaña Predicción (simulación ciclo a ciclo) como para
la pestaña Degradación del motor (proyección de deterioro adicional).

Ejecutar desde la raíz del repo (donde están las carpetas models/ y data/):
    python dump_motor69.py [ciclo_pausa]

ciclo_pausa (opcional): el ciclo en el que "pausarías" la simulación antes
de mirar la pestaña Degradación. Si no lo pasas, usa el ciclo de la
primera alerta.
"""
import sys
import joblib
import numpy as np
import pandas as pd

MOTOR_ID = 69
UMBRAL_ALERTA = 15   # % — el mismo default que usa app.py
UMBRAL_RIESGO = 50   # %

modelo = joblib.load('models/xgboost_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')
print(f"Modelo cargado con {len(feature_names)} features:", feature_names)

df = pd.read_csv(f'data/processed/motores_demo/motor_{MOTOR_ID}.csv')
X = df[feature_names]
df['prob'] = modelo.predict_proba(X)[:, 1] * 100

ciclo_nasa = int(df[df['target'] == 1]['ciclo'].min())
ciclos_totales = int(df['ciclo'].max())

alerta = df[df['prob'] >= UMBRAL_ALERTA]
riesgo = df[df['prob'] >= UMBRAL_RIESGO]
c_alerta = int(alerta.iloc[0]['ciclo']) if len(alerta) else None
p_alerta = float(alerta.iloc[0]['prob']) if len(alerta) else None
c_riesgo = int(riesgo.iloc[0]['ciclo']) if len(riesgo) else None
p_riesgo = float(riesgo.iloc[0]['prob']) if len(riesgo) else None

print("\n=== PESTAÑA PREDICCIÓN — MOTOR #69 ===")
print(f"Ciclos totales:            {ciclos_totales}")
print(f"NASA certifica el fallo:   ciclo {ciclo_nasa}")
if c_alerta is not None:
    print(f"Primera alerta (>= {UMBRAL_ALERTA}%):  ciclo {c_alerta}  ({p_alerta:.2f}%)  "
          f"-> antelación {ciclo_nasa - c_alerta} vuelos")
if c_riesgo is not None:
    print(f"Riesgo alto (>= {UMBRAL_RIESGO}%):      ciclo {c_riesgo}  ({p_riesgo:.2f}%)  "
          f"-> antelación {ciclo_nasa - c_riesgo} vuelos")
print(f"Prob. en ciclo NASA ({ciclo_nasa}):   {df[df['ciclo']==ciclo_nasa]['prob'].values[0]:.2f}%")
print(f"Prob. en el último ciclo ({ciclos_totales}): {df.iloc[-1]['prob']:.2f}%")

print("\n--- Tabla completa ciclo a ciclo ---")
print(df[['ciclo', 'prob', 'target']].to_string(index=False))
df[['ciclo', 'prob', 'target']].to_csv('motor69_prediccion_completa.csv', index=False)
print("\n(Guardada también en motor69_prediccion_completa.csv)")

# ── Pestaña Degradación del motor ──
# Misma lógica que app.py: interpola cada feature desde su valor actual
# hasta el valor "en riesgo" (según SHAP_DIRECCION) en 100 pasos, y mira
# en qué % de deterioro cruza el umbral de riesgo.
SHAP_DIRECCION = {
    's2_norm': +1, 's3_norm': +1, 's4_norm': +1, 's7_norm': +1,
    's8_norm': +1, 's9_norm': +1, 's11_norm': +1, 's12_norm': -1,
    's13_norm': +1, 's14_norm': +1, 's15_norm': +1, 's17_norm': +1,
    's20_norm': -1, 's21_norm': -1,
    's2_norm_mm': +1, 's3_norm_mm': +1, 's4_norm_mm': +1, 's7_norm_mm': -1,
    's8_norm_mm': +1, 's9_norm_mm': +1, 's11_norm_mm': +1, 's12_norm_mm': -1,
    's13_norm_mm': -1, 's14_norm_mm': +1, 's15_norm_mm': +1, 's17_norm_mm': +1,
    's20_norm_mm': -1, 's21_norm_mm': -1,
}

ciclo_pausa = int(sys.argv[1]) if len(sys.argv) > 1 else (c_alerta or 1)
fila_base = df[df['ciclo'] == ciclo_pausa].iloc[0]

pasos = np.arange(0, 101)
t = pasos / 100.0
datos_curva = {}
for col in feature_names:
    dir_f = SHAP_DIRECCION.get(col, 1)
    val_actual = float(fila_base[col])
    val_riesgo = 2.5 * dir_f
    datos_curva[col] = val_actual + t * (val_riesgo - val_actual)

df_curva = pd.DataFrame(datos_curva, columns=feature_names)
probas_curva = modelo.predict_proba(df_curva)[:, 1] * 100

print(f"\n=== PESTAÑA DEGRADACIÓN DEL MOTOR — pausado en ciclo {ciclo_pausa} ===")
print(f"Prob. actual en ese ciclo: {probas_curva[0]:.2f}%")
cruce = np.argmax(probas_curva >= UMBRAL_ALERTA)
if probas_curva[cruce] >= UMBRAL_ALERTA:
    print(f"Entra en ALERTA (>= {UMBRAL_ALERTA}%) con un {cruce}% de deterioro adicional")
cruce_riesgo = np.argmax(probas_curva >= UMBRAL_RIESGO)
if probas_curva[cruce_riesgo] >= UMBRAL_RIESGO:
    print(f"Entra en RIESGO ALTO (>= {UMBRAL_RIESGO}%) con un {cruce_riesgo}% de deterioro adicional")

pd.DataFrame({'deterioro_%': pasos, 'prob_%': probas_curva}).to_csv(
    f'motor69_degradacion_ciclo{ciclo_pausa}.csv', index=False)
print(f"(Guardada también en motor69_degradacion_ciclo{ciclo_pausa}.csv)")
