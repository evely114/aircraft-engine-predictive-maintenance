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
> *Hasta aquí he estado moviendo sensores yo misma — es un escenario hipotético para entender cómo razona el modelo. Ahora la parte más potente — pulso 'Simular motor real'. Esto ya no es un escenario que yo invento: es la trayectoria real de un motor del dataset NASA, con su desenlace ya conocido, para comprobar si el modelo lo hubiera detectado a tiempo."*

*[Arranca la simulación del motor #69]*

> *"Esto es un motor real de la NASA. El motor #69 — vivió 362 vuelos antes de fallar. El modelo no sabe cuándo va a fallar: va leyendo los sensores vuelo a vuelo, en tiempo real.*
>
> *La línea roja marca el ciclo 333 — el momento en que la NASA certifica que este motor entró en zona de peligro.*
>
> *Fíjense aquí — en el ciclo 282, mucho antes del fallo, el modelo ya está disparando la alarma. Lo detectó con 51 vuelos de antelación. 51 vuelos de margen para programar el mantenimiento antes de que ocurra el fallo."*

*[Al llegar al final aparece la nota azul automáticamente]*

*[Seleccionas el motor #16 en el selector y arrancas nueva simulación]*

> *"Ahora vamos con un motor diferente — el motor #16, que vivió solo 209 vuelos. Aquí el patrón de fallo es distinto al del #69, y mucho más ruidoso.*
>
> *Lo dejo correr en automático, y el simulador va parando solo en cada momento clave. Primera parada: ciclo 110 — una señal aislada del 17.9%, un solo ciclo, y enseguida vuelve a bajar. Esa es la primera alerta real: 70 vuelos de antelación sobre el fallo certificado.*
>
> *Sigo, y la siguiente parada es el ciclo 121 — ahí sí hay un salto grande: de 14% a 60% en un solo paso, entrando directo en riesgo alto.*
>
> *Y aquí es donde se pone interesante: entre los ciclos 122 y 159 la probabilidad sube y baja repetidamente — llega a tocar zona segura varias veces, en los ciclos 127, 135, 143, 149, 152 y 158 — con otro pico por encima del 50% en el ciclo 137, que es donde el simulador vuelve a pararse. A partir del ciclo 160 las oscilaciones se hacen más frecuentes y más altas — 83% en el ciclo 161, 64% en el 163, 60% en el 167, 76% en el 176 — y desde el 177 ya no vuelve a bajar del 50%. El simulador pausa en cada uno de esos picos nuevos. Solo al final, en el ciclo 180, la NASA certifica el fallo, con el modelo ya en 96.8% de probabilidad.*
>
> *Fíjense también en este otro pico, en el ciclo 133 — casi llega al 44%, se acerca mucho al umbral de riesgo alto, pero no lo cruza, así que el simulador no se detiene ahí. Eso es intencional: el sistema solo pausa y alerta cuando de verdad se cruza esa línea roja del 50% — así evitamos saturar al técnico de falsas alarmas por cada sube-y-baja dentro de zona de alerta. Es la diferencia entre 'vigilar de cerca' y 'actuar ya'.*
>
> *Esto demuestra tres cosas a la vez: primero, que el modelo reacciona de forma instantánea ante una anomalía puntual, sin necesitar ver un patrón gradual para dispararse; segundo, que no se deja engañar por el ruido intermedio — cada vez que la probabilidad vuelve a subir después de haber bajado por debajo de riesgo alto, el sistema lo marca de nuevo, sin perder de vista el problema una vez que empezó; y tercero, que distingue ruido de señal — no confunde un pico que se queda en zona de alerta con una entrada real en riesgo alto.*
>
> *Eso es exactamente lo que queremos en aviación — un sistema que funciona tanto con fallos súbitos como con degradación ruidosa e intermitente, no uno afinado para un solo tipo de patrón."*

*[Cada pausa automática del simulador muestra su propia nota — primera alerta, riesgo alto, y finalmente el fallo certificado por la NASA]*

*[Con la simulación pausada en el ciclo 110 — la primera alerta — cambias a la pestaña Degradación del motor]*

> *"Este gráfico ya no es una proyección hipotética — es la trayectoria REAL de probabilidad de este motor, de principio a fin, calculada con el mismo modelo. El tramo ya recorrido se ve atenuado y el punto naranja marca dónde estamos pausados ahora mismo, en el ciclo 110, con un 17.9%. Fíjense en lo que viene justo después, apenas 11 ciclos más adelante: el salto al 60%. Este motor parece tranquilo, pero está a un paso del quiebre.*
>
> *A diferencia del motor #69, aquí no hace falta imaginar nada — es literalmente lo que le pasó a este motor, ciclo a ciclo."*

*[Cambias a la pestaña Valor de negocio]*

> *"La tercera pestaña — la que más le interesa a negocio — muestra que este modelo genera un valor neto estimado de 962 millones de dólares en el conjunto de test. Con un ahorro de más de 3.000 millones frente a no tener ningún sistema predictivo."*

---

### TÉCNICA — Cómo está construido (2 minutos)

> *"El pipeline tiene cinco pasos.*
>
> *Primero, feature engineering: el dataset NASA solo da valores de sensores y ciclos. Nosotros creamos dos variables nuevas — el target binario (RUL < 30 ciclos = EN RIESGO) y la media móvil de 10 ciclos para cada sensor, para capturar tendencias de degradación, no solo valores puntuales. En total, 28 features: 14 sensores normalizados y sus 14 medias móviles.*
>
> *Segundo, preprocesamiento: usamos KMeans con k=6 para detectar automáticamente las 6 condiciones operativas de FD002 y FD004, y normalizamos cada sensor dentro de su condición. Así el modelo aprende degradación real y no diferencias de régimen de vuelo.*
>
> *Tercero, el modelo: baseline con Regresión Logística (AUC 0.9881, Recall 95.7%, Precision 69.8%), luego XGBoost sin optimizar (AUC 0.9922, Recall 94.4%, Precision 77.8%). El problema: demasiadas falsas alarmas — más de un 22% de las alarmas eran falsas.*
>
> *Por eso usamos Optuna — optimización bayesiana que buscó automáticamente la mejor combinación de hiperparámetros para equilibrar Recall y Precision sin sacrificar uno por el otro. Resultado final: AUC 0.9934, Recall 93.2%, Precision 82.7% — de cada 100 alarmas, 83 son reales.*
>
> *Cuarto, SHAP TreeExplainer para explicabilidad individual — cada predicción viene con los sensores que la causan.*
>
> *Quinto, Streamlit como producto y Expected Value Framework para cuantificar el impacto económico."*

---

### RESULTADOS — Los números (1 minuto)

> *"Resultados del modelo final — XGBoost + Optuna:*
>
> - *AUC-ROC: 0.9934*
> - *Recall: 93.2% — 93 de cada 100 fallos reales detectados*
> - *Precision: 82.7% — 83 de cada 100 alarmas son reales*
> - *F1-Score: 87.7%*
> - *Detección anticipada: 60 vuelos de media antes del fallo certificado por NASA (5 motores del simulador)*
> - *Valor económico neto: +$962.3 millones en el conjunto de test*
> - *Ahorro vs sin modelo: +$3.089,3 millones"*

---

### CIERRE (30 segundos)

> *"Este proyecto demuestra el flujo completo de producción: problema de negocio, EDA, feature engineering, modelado, explicabilidad y producto final. El repositorio está en GitHub con toda la documentación. Estaré encantada de responder preguntas."*

---

## 🎤 Preguntas frecuentes

**"¿Para qué sirven los dos modos de la app — manual y automático?"**
> *"No compiten, se complementan. El modo manual, con los sliders y los presets 'Motor nuevo' / 'Motor en riesgo', es una foto fija de un escenario hipotético — sirve para explorar cómo razona el modelo, qué sensor pesa más en cada decisión, apoyado en el SHAP waterfall. Su valor es pedagógico: enseña por qué el modelo decide lo que decide.*
> *El modo automático, 'Simular motor real', es distinto: reproduce ciclo a ciclo la trayectoria real de un motor del dataset NASA, con su desenlace ya conocido, y compara la predicción del modelo contra la verdad de terreno en cada ciclo. Eso aporta tres cosas que el manual no puede dar: validación con datos reales — no una suposición mía —, la dimensión temporal — de ahí sale la métrica de antelación (60 vuelos de media entre los 5 motores del simulador), que es imposible de mostrar con un slider estático —, y una narrativa mucho más convincente para una audiencia no técnica: ver a un motor real degradarse y al modelo detectarlo a tiempo se recuerda mucho más que una barra SHAP."*

**"¿Por qué XGBoost y no red neuronal?"**
> *"XGBoost es superior en datos tabulares, compatible con SHAP y mucho más eficiente computacionalmente. Una LSTM capturaría mejor las secuencias temporales pero perdería la explicabilidad — crítica en aviación donde hay que justificar cada decisión de mantenimiento."*

**"¿Por qué bajaron las métricas con los 4 subconjuntos?"**
> *"Con FD001 el modelo aprendía 1 condición y 1 tipo de fallo. Con los 4 subconjuntos enfrenta 6 condiciones y 2 tipos de fallo. Una bajada de 0.003 en AUC-ROC a cambio de un modelo que generaliza a condiciones reales es un trade-off completamente aceptable."*

**"¿Por qué KMeans para las condiciones operativas?"**
> *"FD002 y FD004 tienen 6 condiciones de vuelo no etiquetadas. Sin normalizarlas, el modelo aprendería diferencias entre condiciones en vez de degradación real. KMeans las detecta automáticamente con k=6 y normalizamos cada sensor dentro de su condición."*

**"¿Por qué usaste Optuna?"**
> *"Sin optimizar, el modelo tenía Precision del 77.8% — de cada 100 alarmas, más de 22 eran falsas. En aviación eso genera desconfianza en el sistema. Optuna encontró una mejor combinación de hiperparámetros: subió la Precision de 77.8% a 82.7% prácticamente sin sacrificar Recall (94.4% → 93.2%). Menos falsas alarmas, casi sin perder capacidad de detección."*

**"¿Qué es el target y de dónde sale?"**
> *"La NASA no etiqueta si un motor va a fallar — solo da los valores de sensores y los ciclos. Nosotros calculamos el RUL (Vida Útil Restante) restando el ciclo actual al ciclo máximo de cada motor, y creamos el target binario: si RUL < 30 ciclos → EN RIESGO. Esos 30 ciclos son la ventana de mantenimiento — el tiempo mínimo para actuar."*

**"¿Los casos en riesgo son los mismos motores que los 709 de entrenamiento?"**
> *"No. Los 709 motores son para entrenar el modelo. En el conjunto de test hay 32.072 predicciones en total, de las cuales 4.254 tenían target=1 (EN RIESGO) y 27.818 estaban seguros. De esos 4.254 casos en riesgo, el modelo detectó 3.965 — el Recall del 93.2% que ya mencionamos."*

**"¿Cómo escalarías a producción?"**
> *"Stream de datos en tiempo real desde sensores IoT, alertas automáticas al equipo de mantenimiento y pipeline de reentrenamiento continuo con nuevos datos de flota."*

**"¿Qué mejorarías?"**
> *"Transfer learning entre tipos de motores, integración con sensores IoT en tiempo real y un modelo de series temporales como LSTM para capturar mejor la secuencia de degradación."*

---

## 📋 Números clave — memorizar

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.9934 |
| Recall | 93.2% |
| Precision | 82.7% |
| F1-Score | 87.7% |
| Motores entrenamiento | 709 |
| Registros totales | 160.000 |
| Fallos detectados | 3.965 de 4.254 |
| Ciclos de antelación | 60 vuelos de media (5 motores del simulador) |
| Valor neto modelo | +$962,3M |
| Ahorro vs sin modelo | +$3.089,3M |

---

## 🔧 Los 5 motores del simulador — datos verificados ciclo a ciclo

*Antelación = ciclo de la Primera Alerta (probabilidad ≥ 15%) hasta el ciclo donde la NASA certifica el fallo.*

| Motor | Ciclos totales | NASA certifica | Primera alerta (ciclo · %) | Antelación |
|---|---|---|---|---|
| #16 | 209 | 180 | 110 · 17.9% | 70 vuelos |
| #56 | 275 | 246 | 191 · 16.1% | 55 vuelos |
| #69 | 362 | 333 | 282 · 16.7% | 51 vuelos |
| #84 | 267 | 238 | 152 · 27.7% | 86 vuelos |
| #92 | 341 | 312 | 274 · 22.9% | 38 vuelos |

**Media: 60 vuelos de antelación.**

Detalle del motor #16 (el más ruidoso — usado en la demo en vivo): primera alerta aislada en el ciclo 110 (17.9%), salto a riesgo alto en el ciclo 121 (60.2%), nuevo pico por encima del 50% en el ciclo 137 tras haber tocado zona segura varias veces (ciclos 127, 135, 143, 149, 152, 158), y desde el ciclo 160 oscilaciones más frecuentes y altas (161: 83%, 163: 64%, 167: 60%, 176: 76%, 177-179 siempre por encima del 50%) hasta la certificación de la NASA en el ciclo 180 (96.8%).

---

## ✅ Checklist antes de la presentación

- [ ] App funcionando — botón "Motor en riesgo" muestra EN RIESGO
- [ ] Simulación motor real probada de inicio a fin (motor #69 y motor #16)
- [ ] AUTO pausa solo en cada pico real de riesgo alto y en el fallo NASA — comprobado
- [ ] GitHub abierto en otra pestaña
- [ ] Practicar demo 2-3 veces
- [ ] Números memorizados (tabla arriba)
