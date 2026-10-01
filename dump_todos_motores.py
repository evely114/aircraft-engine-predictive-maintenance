"""
Vuelca los datos COMPLETOS de los 5 motores del simulador (16, 56, 69, 84, 92),
tanto para la pestaña Predicción como para la pestaña Degradación del motor —
y para Degradación usa, en cada motor, el ciclo REAL donde la app se pausa
sola (Primera Alerta), no un ciclo elegido a mano.

Ejecutar desde la raíz del repo (donde están models/ y data/):
    python dump_todos_motores.py
"""
import joblib
import numpy as np
import pandas as pd

MOTORES = [16, 56, 69, 84, 92]
UMBRAL_ALERTA = 15   # % — default de app.py
UMBRAL_RIESGO = 50   # %

modelo = joblib.load('models/xgboost_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')
print(f"Modelo cargado con {len(feature_names)} features\n")

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

resumen = []

for motor_id in MOTORES:
    print("=" * 70)
    print(f"MOTOR #{motor_id}")
    print("=" * 70)

    df = pd.read_csv(f'data/processed/motores_demo/motor_{motor_id}.csv')
    X = df[feature_names]
    df['prob'] = modelo.predict_proba(X)[:, 1] * 100

    ciclos_totales = int(df['ciclo'].max())
    ciclo_nasa = int(df[df['target'] == 1]['ciclo'].min()) if 1 in df['target'].values else None

    alerta = df[df['prob'] >= UMBRAL_ALERTA]
    riesgo = df[df['prob'] >= UMBRAL_RIESGO]
    c_alerta = int(alerta.iloc[0]['ciclo']) if len(alerta) else None
    p_alerta = float(alerta.iloc[0]['prob']) if len(alerta) else None
    c_riesgo = int(riesgo.iloc[0]['ciclo']) if len(riesgo) else None
    p_riesgo = float(riesgo.iloc[0]['prob']) if len(riesgo) else None

    antelacion_alerta = (ciclo_nasa - c_alerta) if (ciclo_nasa and c_alerta) else None
    antelacion_riesgo = (ciclo_nasa - c_riesgo) if (ciclo_nasa and c_riesgo) else None

    print(f"Ciclos totales:          {ciclos_totales}")
    print(f"NASA certifica el fallo: ciclo {ciclo_nasa}")
    if c_alerta is not None:
        print(f"Primera alerta (auto-pausa, >= {UMBRAL_ALERTA}%): ciclo {c_alerta} ({p_alerta:.2f}%) "
              f"-> antelación {antelacion_alerta} vuelos")
    if c_riesgo is not None:
        print(f"Riesgo alto (>= {UMBRAL_RIESGO}%): ciclo {c_riesgo} ({p_riesgo:.2f}%) "
              f"-> antelación {antelacion_riesgo} vuelos")

    # Degradación del motor, calculada EXACTAMENTE en el ciclo de auto-pausa (primera alerta)
    deter_alerta = deter_riesgo = None
    if c_alerta is not None:
        fila_base = df[df['ciclo'] == c_alerta].iloc[0]
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

        cruce_a = np.argmax(probas_curva >= UMBRAL_ALERTA)
        if probas_curva[cruce_a] >= UMBRAL_ALERTA:
            deter_alerta = int(cruce_a)
        cruce_r = np.argmax(probas_curva >= UMBRAL_RIESGO)
        if probas_curva[cruce_r] >= UMBRAL_RIESGO:
            deter_riesgo = int(cruce_r)

        print(f"\nDegradación (pausado en ciclo {c_alerta}, prob. actual {probas_curva[0]:.2f}%):")
        print(f"  Entra en ALERTA con:      {deter_alerta}% de deterioro adicional")
        print(f"  Entra en RIESGO ALTO con: {deter_riesgo}% de deterioro adicional")

    df[['ciclo', 'prob', 'target']].to_csv(f'motor{motor_id}_prediccion_completa.csv', index=False)

    resumen.append({
        'motor': motor_id,
        'ciclos_totales': ciclos_totales,
        'ciclo_nasa': ciclo_nasa,
        'ciclo_alerta': c_alerta,
        'prob_alerta': round(p_alerta, 2) if p_alerta is not None else None,
        'antelacion_alerta_vuelos': antelacion_alerta,
        'ciclo_riesgo_alto': c_riesgo,
        'antelacion_riesgo_vuelos': antelacion_riesgo,
        'deterioro_adicional_alerta_%': deter_alerta,
        'deterioro_adicional_riesgo_%': deter_riesgo,
    })
    print()

print("=" * 70)
print("RESUMEN — LOS 5 MOTORES DEL SIMULADOR")
print("=" * 70)
resumen_df = pd.DataFrame(resumen)
print(resumen_df.to_string(index=False))
resumen_df.to_csv('resumen_5_motores.csv', index=False)
print("\n(Guardado también en resumen_5_motores.csv)")

media_antelacion = resumen_df['antelacion_alerta_vuelos'].mean()
print(f"\nAntelación media de los 5 motores (primera alerta): {media_antelacion:.1f} vuelos")
