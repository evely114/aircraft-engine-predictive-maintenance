import joblib
feature_names = joblib.load('models/feature_names.pkl')
print(f"Número de features: {len(feature_names)}")
slope = [f for f in feature_names if f.endswith('_slope')]
print(f"Features '_slope' (deberían ser 0): {len(slope)}")
print(feature_names)

modelo = joblib.load('models/xgboost_model.pkl')
try:
    n_in = modelo.n_features_in_
    print(f"n_features_in_ del modelo: {n_in}")
    print("COINCIDE con feature_names" if n_in == len(feature_names) else "NO COINCIDE — modelo y feature_names están desincronizados")
except AttributeError:
    print("No se pudo leer n_features_in_ directamente (revisa con .get_booster().feature_names si es Booster crudo)")
