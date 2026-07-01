# Pitch de presentación — Aircraft Engine Predictive Maintenance
## Bootcamp Data Science — Proyecto Final

---

## 🎯 Estructura (7-10 minutos)

---

### APERTURA — El problema (1 minuto)

> *"Cada año, las aerolíneas gastan más de 50.000 millones de dólares en mantenimiento de motores. El 30% de ese coste viene de fallos no planificados — motores que fallan sin avisar, aviones que quedan en tierra, vuelos cancelados, vidas en riesgo.*
>
> *El mantenimiento tradicional funciona por calendario: cada X vuelos, revisas el motor. No importa si está perfecto o a punto de fallar.*
>
> *¿Y si el motor te avisara antes de fallar?"*

---

### SOLUCIÓN — El proyecto (1.5 minutos)

> *"Construí un sistema de mantenimiento predictivo que analiza las lecturas de sensores de un motor turbofan en tiempo real y predice si ese motor va a fallar en los próximos 30 vuelos.*
>
> *Los datos vienen del dataset NASA C-MAPSS — el benchmark de referencia mundial para PHM en aviación. Entrenamos con los 4 subconjuntos — 709 motores y 160.000 registros — cubriendo 6 condiciones operativas distintas y 2 tipos de fallo.*
>
> *El sistema tiene tres componentes: un modelo XGBoost optimizado con Optuna, explicabilidad SHAP para cada predicción individual, y una app Streamlit donde cualquier técnico puede analizar un motor sin saber Machine Learning."*

---

### DEMO EN VIVO — La app (2-3 minutos)

*[Abres la app de Streamlit — pestaña Predicción]*

> *"Empezamos con un motor nuevo — 0% de riesgo, todos los sensores en verde.*
>
> *Pulso 'Motor en riesgo' y el modelo detecta inmediatamente el peligro. Pero lo más importante es esto:"*

*[Señalas el gráfico SHAP]*

> *"El modelo me dice exactamente por qué: s15 — ratio de bypass — y s11 — presión del compresor de alta presión — están empujando hacia riesgo. No es una caja negra. Es una explicación que un técnico puede actuar.*
>
> *Ahora la parte más potente — pulso 'Simular motor real'."*

*[Arranca la simulación del motor #69]*

> *"Esto es un motor real de la NASA. El motor #69 — vivió 362 vuelos antes de fallar. El modelo no sabe cuándo va a fallar: va leyendo los sensores vuelo a vuelo, en tiempo real.*
>
> *La línea roja marca el ciclo 333 — el momento en que la NASA certifica que este motor entró en zona de peligro.*
>
> *Fíjense aquí — en el ciclo 291, mucho antes del fallo, el modelo ya está disparando la alarma. Lo detectó con 42 vuelos de antelación. 42 vuelos de margen para programar el mantenimiento antes de que ocurra el fallo."*

*[Al llegar al final aparece la nota azul automáticamente]*

*[Seleccionas el motor #16 en el selector y arrancas nueva simulación]*

> *"Ahora vamos con un motor diferente — el motor #16, que vivió solo 209 vuelos. Este es más interesante que el #69.*
>
> *Fíjense — en el ciclo 120 el modelo detecta una anomalía y dispara la alarma. Pero luego los sensores se recuperan y vuelve a SEGURO. Lo mismo ocurre en el ciclo 160. Solo cuando llega al ciclo 180 y la degradación es sostenida, el modelo mantiene la alarma hasta el fallo.*
>
> *Esto demuestra tres cosas a la vez:*
> *Primero — el modelo no es alarmista. Detecta anomalías pero vuelve a SEGURO si los sensores se recuperan.*
> *Segundo — distingue picos temporales de degradación sostenida. Exactamente para lo que fue diseñado.*
> *Tercero — robustez. Funciona bien incluso con motores con patrones complejos.*
>
> *Eso es exactamente lo que queremos en aviación — un sistema que no grita lobo a la primera señal rara, sino que distingue ruido de degradación real."*

*[Al llegar al final aparece la nota azul con los datos del motor #16 automáticamente]*

*[Cambias a la pestaña Degradación del motor]*

> *"Este gráfico responde a: ¿cuánto deterioro aguanta este motor antes de entrar en zona crítica? Desde el estado actual, el modelo proyecta que con un 39% de deterioro adicional entraría en riesgo. A partir de ahí la degradación es rápida y el fallo es casi inevitable.*
>
> *Importante: esto es predicción del modelo, no datos NASA. El modelo XGBoost evalúa escenarios hipotéticos de deterioro — no reproduce grabaciones reales."*

*[Cambias a la pestaña Valor de negocio]*

> *"La tercera pestaña — la que más le interesa a negocio — muestra que este modelo genera un valor neto estimado de 145 millones de dólares en el conjunto de test. Con un ahorro de 445 millones frente a no tener ningún sistema predictivo."*

---

### TÉCNICA — Cómo está construido (2 minutos)

> *"El pipeline tiene cinco pasos.*
>
> *Primero, feature engineering: el dataset NASA solo da valores de sensores y ciclos. Nosotros creamos dos variables nuevas — el target binario (RUL < 30 ciclos = EN RIESGO) y la media móvil de 10 ciclos para cada sensor, para capturar tendencias de degradación, no solo valores puntuales. En total, 28 features: 14 sensores normalizados y sus 14 medias móviles.*
>
> *Segundo, preprocesamiento: usamos KMeans con k=6 para detectar automáticamente las 6 condiciones operativas de FD002 y FD004, y normalizamos cada sensor dentro de su condición. Así el modelo aprende degradación real y no diferencias de régimen de vuelo.*
>
> *Tercero, el modelo: baseline con Regresión Logística (AUC 0.9881, Recall 95.7%, Precision 69.7%), luego XGBoost sin optimizar (AUC 0.9922, Recall 94.4%, Precision 77.7%). El problema: demasiadas falsas alarmas — el 22% de las alarmas eran falsas.*
>
> *Por eso usamos Optuna — optimización bayesiana que buscó automáticamente la mejor combinación de hiperparámetros para equilibrar Recall y Precision sin sacrificar uno por el otro. Resultado final: AUC 0.9940, Recall 91.2%, Precision 88.2% — de cada 100 alarmas, 88 son reales.*
>
> *Cuarto, SHAP TreeExplainer para explicabilidad individual — cada predicción viene con los sensores que la causan.*
>
> *Quinto, Streamlit como producto y Expected Value Framework para cuantificar el impacto económico."*

---

### RESULTADOS — Los números (1 minuto)

> *"Resultados del modelo final — XGBoost + Optuna:*
>
> - *AUC-ROC: 0.9940*
> - *Recall: 91.2% — 91 de cada 100 fallos reales detectados*
> - *Precision: 88.2% — 88 de cada 100 alarmas son reales*
> - *F1-Score: 89.7%*
> - *Detección anticipada: 42 vuelos antes del fallo certificado por NASA*
> - *Valor económico neto: +$145 millones en el conjunto de test*
> - *Ahorro vs sin modelo: +$445 millones"*

---

### CIERRE (30 segundos)

> *"Este proyecto demuestra el flujo completo de producción: problema de negocio, EDA, feature engineering, modelado, explicabilidad y producto final. El repositorio está en GitHub con toda la documentación. Estaré encantada de responder preguntas."*

---

## 🎤 Preguntas frecuentes

**"¿Por qué XGBoost y no red neuronal?"**
> *"XGBoost es superior en datos tabulares, compatible con SHAP y mucho más eficiente computacionalmente. Una LSTM capturaría mejor las secuencias temporales pero perdería la explicabilidad — crítica en aviación donde hay que justificar cada decisión de mantenimiento."*

**"¿Por qué bajaron las métricas con los 4 subconjuntos?"**
> *"Con FD001 el modelo aprendía 1 condición y 1 tipo de fallo. Con los 4 subconjuntos enfrenta 6 condiciones y 2 tipos de fallo. Una bajada de 0.003 en AUC-ROC a cambio de un modelo que generaliza a condiciones reales es un trade-off completamente aceptable."*

**"¿Por qué KMeans para las condiciones operativas?"**
> *"FD002 y FD004 tienen 6 condiciones de vuelo no etiquetadas. Sin normalizarlas, el modelo aprendería diferencias entre condiciones en vez de degradación real. KMeans las detecta automáticamente con k=6 y normalizamos cada sensor dentro de su condición."*

**"¿Por qué usaste Optuna?"**
> *"Sin optimizar, el modelo tenía Precision del 77% — de cada 100 alarmas, 22 eran falsas. En aviación eso genera desconfianza en el sistema. Optuna encontró el equilibrio: bajó un poco el Recall (94% → 91%) pero la Precision subió de 77% a 88%. Un modelo más confiable en producción."*

**"¿Qué es el target y de dónde sale?"**
> *"La NASA no etiqueta si un motor va a fallar — solo da los valores de sensores y los ciclos. Nosotros calculamos el RUL (Vida Útil Restante) restando el ciclo actual al ciclo máximo de cada motor, y creamos el target binario: si RUL < 30 ciclos → EN RIESGO. Esos 30 ciclos son la ventana de mantenimiento — el tiempo mínimo para actuar."*

**"¿Los 600 fallos son los mismos que los 709 motores de entrenamiento?"**
> *"No. Los 709 motores son para entrenar el modelo. Los 600 son los casos en riesgo del conjunto de test — 4.127 predicciones en total, de las cuales 600 tenían target=1 (EN RIESGO) y 3.527 estaban seguros."*

**"¿Cómo escalarías a producción?"**
> *"Stream de datos en tiempo real desde sensores IoT, alertas automáticas al equipo de mantenimiento y pipeline de reentrenamiento continuo con nuevos datos de flota."*

**"¿Qué mejorarías?"**
> *"Transfer learning entre tipos de motores, integración con sensores IoT en tiempo real y un modelo de series temporales como LSTM para capturar mejor la secuencia de degradación."*

---

## 📋 Números clave — memorizar

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.9940 |
| Recall | 91.2% |
| Precision | 88.2% |
| F1-Score | 89.7% |
| Motores entrenamiento | 709 |
| Registros totales | 160.000 |
| Fallos detectados | 563 de 600 |
| Ciclos de antelación | 42 vuelos |
| Valor neto modelo | +$145M |
| Ahorro vs sin modelo | +$445M |

---

## ✅ Checklist antes de la presentación

- [ ] App funcionando — botón "Motor en riesgo" muestra EN RIESGO
- [ ] Simulación motor real probada de inicio a fin
- [ ] Nota de 42 ciclos aparece al llegar al ciclo 362
- [ ] GitHub abierto en otra pestaña
- [ ] Practicar demo 2-3 veces
- [ ] Números memorizados (tabla arriba)
