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

### DEMO EN VIVO — La app (2 minutos)

*[Abres la app de Streamlit]*

> *"Tenemos tres pestañas. En Predicción, pulso 'Motor en riesgo' y el modelo detecta inmediatamente el riesgo.*
>
> *Pero lo más importante es esto:"*

*[Señalas el gráfico SHAP]*

> *"El modelo me dice exactamente por qué: el sensor s4 — temperatura de la turbina de baja presión — lleva ciclos subiendo de forma anormal. Y s11, la presión del compresor de alta presión, también muestra degradación. Eso es lo que un técnico necesita saber para actuar.*
>
> *La tercera pestaña — la que más le interesa a negocio — muestra que este modelo genera un valor estimado de 145 millones de dólares en el conjunto de test."*

---

### TÉCNICA — Cómo está construido (2 minutos)

> *"El pipeline tiene cinco pasos.*
>
> *Primero, preprocesamiento: combinamos los 4 subconjuntos del NASA C-MAPSS — 709 motores, 160.000 registros. Usamos KMeans para detectar las 6 condiciones operativas automáticamente y normalizamos cada sensor dentro de su condición, para que el modelo aprenda degradación real y no diferencias de régimen de vuelo.*
>
> *Segundo, el modelo: baseline con Regresión Logística (AUC-ROC 0.9881), luego XGBoost, y finalmente Optuna para optimización bayesiana — 30 combinaciones con cross-validation estratificada de 5 folds.*
>
> *Tercero, SHAP para explicabilidad individual. Cuarto, Streamlit como producto. Quinto, Expected Value Framework para cuantificar el impacto económico."*

---

### RESULTADOS — Los números (1 minuto)

> *"Resultados del modelo final con los 4 subconjuntos:*
>
> - *AUC-ROC: 0.9936*
> - *Recall: 91.5% — 91 de cada 100 fallos reales detectados*
> - *Precision: 86.8%*
> - *Valor económico: +$145 millones de dólares en el conjunto de test*
> - *Ahorro vs sin modelo: +$445 millones*"*

---

### CIERRE (30 segundos)

> *"Este proyecto demuestra el flujo completo de producción: problema de negocio, EDA, feature engineering avanzado, modelado, explicabilidad y producto final. El repositorio está en GitHub con toda la documentación. Estaré encantada de responder preguntas."*

---

## 🎤 Preguntas frecuentes

**"¿Por qué XGBoost y no red neuronal?"**
> *"XGBoost es superior en datos tabulares, compatible con SHAP y mucho más eficiente computacionalmente. Una LSTM capturaría mejor las secuencias temporales pero perdería la explicabilidad — crítica en aviación."*

**"¿Por qué bajaron las métricas al añadir los 4 subconjuntos?"**
> *"Con FD001 el modelo aprendía 1 condición y 1 tipo de fallo. Con los 4 subconjuntos enfrenta 6 condiciones y 2 tipos de fallo. Una bajada de 0.003 en AUC-ROC a cambio de un modelo que generaliza a condiciones reales es un trade-off completamente aceptable."*

**"¿Por qué KMeans para las condiciones operativas?"**
> *"FD002 y FD004 tienen 6 condiciones de vuelo. Sin normalizarlas, el modelo aprendería diferencias entre condiciones en vez de degradación. KMeans las detecta automáticamente y normalizamos cada sensor dentro de su condición."*

**"¿Cómo escalarías a producción?"**
> *"Stream de datos en tiempo real, alertas automáticas y pipeline de reentrenamiento continuo."*

**"¿Qué mejorarías?"**
> *"Transfer learning entre tipos de motores e integración con sensores IoT en tiempo real."*

---

## 📋 Checklist antes del jueves

- [ ] App funcionando — botón "Motor en riesgo" muestra EN RIESGO
- [ ] GitHub abierto en otra pestaña
- [ ] Practicar demo 2-3 veces
- [ ] Números memorizados: 0.9936 · 91.5% · 709 motores · 160K registros · $145M · $445M
