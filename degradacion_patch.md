# Parche — Pestaña "Degradación del motor" usa datos reales en modo Simular motor real

Busca cada bloque de "ANTES" en tu `app.py` (con Ctrl+F) y sustitúyelo por el de "DESPUÉS".
Son 3 cambios, todos dentro de la sección de la pestaña Degradación (tab4).

---

## Cambio 1 — el aviso de arriba (ahora es condicional)

**ANTES:**
```python
    st.markdown('''
    <div style="font-size:13px;color:#64748b;line-height:1.8;margin-bottom:10px">
        Este gráfico simula cómo evoluciona la <span style="color:#0ea5e9">probabilidad de fallo</span>
        a medida que los sensores se <strong style="color:#e2e8f0">degradan progresivamente</strong>
        desde su estado actual hasta el máximo deterioro.
        Mueve los sliders en Predicción y vuelve aquí para ver cómo cambia la trayectoria.
    </div>
    <div style="background:#0f1829;border:1px solid #1e2d4a;border-left:3px solid #f59e0b;
         border-radius:6px;padding:8px 14px;margin-bottom:16px;font-size:11px;color:#64748b;line-height:1.7">
        🤖 <strong style="color:#f59e0b">Predicción del modelo, no datos NASA.</strong>
        El modelo XGBoost evalúa escenarios hipotéticos de deterioro a partir del estado actual de los sensores —
        no reproduce grabaciones reales del dataset.
    </div>
    ''', unsafe_allow_html=True)

    col_curva, col_info = st.columns([3, 1], gap="large")

    with col_curva:
        st.markdown('<div class="panel-title">Curva de degradación progresiva — 0% a 100% de deterioro</div>', unsafe_allow_html=True)
```

**DESPUÉS:**
```python
    _modo_deg = st.session_state.get('modo_preset', 'manual')
    _es_real_deg = (_modo_deg == 'real')

    if _es_real_deg:
        st.markdown('''
        <div style="font-size:13px;color:#64748b;line-height:1.8;margin-bottom:10px">
            Este gráfico muestra cómo evolucionó realmente la
            <span style="color:#0ea5e9">probabilidad de fallo</span> de este motor
            desde el ciclo donde tienes pausada la simulación hasta su último ciclo.
        </div>
        <div style="background:#0a1a0e;border:1px solid #1e2d4a;border-left:3px solid #22c55e;
             border-radius:6px;padding:8px 14px;margin-bottom:16px;font-size:11px;color:#64748b;line-height:1.7">
            📊 <strong style="color:#22c55e">Datos reales del motor NASA.</strong>
            No es una proyección — es la trayectoria real de los ciclos que le quedan a este motor,
            calculada con el mismo modelo.
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div style="font-size:13px;color:#64748b;line-height:1.8;margin-bottom:10px">
            Este gráfico simula cómo evoluciona la <span style="color:#0ea5e9">probabilidad de fallo</span>
            a medida que los sensores se <strong style="color:#e2e8f0">degradan progresivamente</strong>
            desde su estado actual hasta el máximo deterioro.
            Mueve los sliders en Predicción y vuelve aquí para ver cómo cambia la trayectoria.
        </div>
        <div style="background:#0f1829;border:1px solid #1e2d4a;border-left:3px solid #f59e0b;
             border-radius:6px;padding:8px 14px;margin-bottom:16px;font-size:11px;color:#64748b;line-height:1.7">
            🤖 <strong style="color:#f59e0b">Predicción del modelo, no datos NASA.</strong>
            El modelo XGBoost evalúa escenarios hipotéticos de deterioro a partir del estado actual de los sensores —
            no reproduce grabaciones reales del dataset.
        </div>
        ''', unsafe_allow_html=True)

    col_curva, col_info = st.columns([3, 1], gap="large")

    with col_curva:
        _titulo_curva = ('Curva real de degradación — ciclos que le quedan al motor' if _es_real_deg
                          else 'Curva de degradación progresiva — 0% a 100% de deterioro')
        st.markdown(f'<div class="panel-title">{_titulo_curva}</div>', unsafe_allow_html=True)
```

---

## Cambio 2 — el cálculo de la curva (aquí está el núcleo del cambio)

**ANTES:**
```python
        n_pasos = 100
        pasos_eje = list(range(0, 101, 1))  # 0% a 100% de degradación
        t_array = np.array(pasos_eje) / 100.0

        modo_curva = st.session_state.get('modo_preset', 'manual')
        fila_actual = slider_to_fila(st.session_state, modo=modo_curva)
        fila_base = fila_actual.iloc[0]

        datos_curva = {}
        for col_f in feature_names:
            dir_f = SHAP_DIRECCION.get(col_f, 1)
            val_actual_f = float(fila_base[col_f])
            val_riesgo_f = 2.5 * dir_f
            datos_curva[col_f] = val_actual_f + t_array * (val_riesgo_f - val_actual_f)
        df_curva = pd.DataFrame(datos_curva, columns=feature_names)
        probas_curva = (modelo.predict_proba(df_curva)[:, 1] * 100).tolist()

        # Suavizar curva con media móvil
        ventana_suav = 8
        probas_suav = []
        for i in range(len(probas_curva)):
            inicio = max(0, i - ventana_suav + 1)
            probas_suav.append(float(np.mean(probas_curva[inicio:i+1])))
        probas_curva = probas_suav

        prob_actual = probas_curva[0]
        ciclo_actual = 0  # punto de partida
```

**DESPUÉS:**
```python
        modo_curva = _modo_deg

        if _es_real_deg:
            # ── Motor real: usamos los ciclos que YA conocemos (dataset histórico
            #    completo), en vez de una extrapolación hipotética. ──
            sim_ciclo_deg = st.session_state.get('sim_ciclo', 0)
            df_restante = motor_demo_df.iloc[sim_ciclo_deg:].reset_index(drop=True)
            X_restante = df_restante[feature_names]
            probas_curva = (modelo.predict_proba(X_restante)[:, 1] * 100).tolist()
            # eje x = ciclos reales transcurridos desde el punto de pausa (0, 1, 2, ...)
            pasos_eje = list(range(len(probas_curva)))
            # ciclo NASA real, para marcarlo en el gráfico
            _ciclo_actual_num_deg = int(df_restante.iloc[0]['ciclo'])
            _ciclos_nasa_deg = motor_demo_df[motor_demo_df['target'] == 1]['ciclo']
            ciclo_nasa_deg = int(_ciclos_nasa_deg.min()) if len(_ciclos_nasa_deg) else None
        else:
            # ── Manual / presets: proyección hipotética (igual que antes) ──
            n_pasos = 100
            pasos_eje = list(range(0, 101, 1))  # 0% a 100% de degradación
            t_array = np.array(pasos_eje) / 100.0

            fila_actual = slider_to_fila(st.session_state, modo=modo_curva)
            fila_base = fila_actual.iloc[0]

            datos_curva = {}
            for col_f in feature_names:
                dir_f = SHAP_DIRECCION.get(col_f, 1)
                val_actual_f = float(fila_base[col_f])
                val_riesgo_f = 2.5 * dir_f
                datos_curva[col_f] = val_actual_f + t_array * (val_riesgo_f - val_actual_f)
            df_curva = pd.DataFrame(datos_curva, columns=feature_names)
            probas_curva = (modelo.predict_proba(df_curva)[:, 1] * 100).tolist()

        # Suavizar curva con media móvil
        ventana_suav = 8
        probas_suav = []
        for i in range(len(probas_curva)):
            inicio = max(0, i - ventana_suav + 1)
            probas_suav.append(float(np.mean(probas_curva[inicio:i+1])))
        probas_curva = probas_suav

        prob_actual = probas_curva[0]
        ciclo_actual = 0  # punto de partida
```

---

## Cambio 3 — ejes y etiqueta del gráfico (eje X e xlim dependen del modo)

**ANTES:**
```python
        ax4.set_xlabel('% de degradación de sensores', fontsize=11)
        ax4.set_ylabel('Probabilidad de fallo (%)', fontsize=11)
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
```

**DESPUÉS:**
```python
        ax4.set_xlabel('Ciclos reales transcurridos' if _es_real_deg else '% de degradación de sensores', fontsize=11)
        ax4.set_ylabel('Probabilidad de fallo (%)', fontsize=11)
        ax4.set_xlim(0, max(pasos_eje) if pasos_eje else 100)
        ax4.set_ylim(0, 100)
```

Y un poco más abajo, las etiquetas de zona (`ax4.text(98, 7, 'SEGURO', ...)` etc.) usan la coordenada `98` fija — con el eje real esa `x` puede quedarse fuera de rango si el motor tiene pocos ciclos restantes. Sustituye ese bloque:

**ANTES:**
```python
        ax4.text(98, 7,  'SEGURO',      ha='right', va='center', fontsize=9, color='#22c55e', fontweight='600', alpha=0.7)
        ax4.text(98, 30, 'ALERTA',      ha='right', va='center', fontsize=9, color='#f59e0b', fontweight='600', alpha=0.7)
        ax4.text(98, 70, 'EN RIESGO',   ha='right', va='center', fontsize=9, color='#ef4444', fontweight='600', alpha=0.7)
```

**DESPUÉS:**
```python
        _x_etq = (max(pasos_eje) * 0.98) if pasos_eje else 98
        ax4.text(_x_etq, 7,  'SEGURO',      ha='right', va='center', fontsize=9, color='#22c55e', fontweight='600', alpha=0.7)
        ax4.text(_x_etq, 30, 'ALERTA',      ha='right', va='center', fontsize=9, color='#f59e0b', fontweight='600', alpha=0.7)
        ax4.text(_x_etq, 70, 'EN RIESGO',   ha='right', va='center', fontsize=9, color='#ef4444', fontweight='600', alpha=0.7)
```

---

## Cambio 4 — la nota dinámica y la tarjeta "Entra en riesgo con X%" (texto en ciclos, no %)

**ANTES (nota bajo el gráfico):**
```python
        if ciclo_umbral_nota is not None and prob_actual < umbral_alerta_pct:
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #0ea5e9;border-left:4px solid #0ea5e9;
                 border-radius:8px;padding:12px 18px;margin-top:8px">
                <div style="font-size:11px;color:#94a3b8;line-height:1.9">
                    Este motor aguanta hasta un <strong style="color:#f59e0b">{ciclo_umbral_nota}% de deterioro</strong>
                    antes de entrar en zona de riesgo.
                    A partir de ahí, la degradación es rápida y el fallo es casi inevitable.
                </div>
            </div>
            """, unsafe_allow_html=True)
```

**DESPUÉS:**
```python
        if ciclo_umbral_nota is not None and prob_actual < umbral_alerta_pct:
            _texto_umbral = (f"{ciclo_umbral_nota} ciclos más" if _es_real_deg
                              else f"un <strong style=\"color:#f59e0b\">{ciclo_umbral_nota}% de deterioro</strong>")
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #0ea5e9;border-left:4px solid #0ea5e9;
                 border-radius:8px;padding:12px 18px;margin-top:8px">
                <div style="font-size:11px;color:#94a3b8;line-height:1.9">
                    Este motor aguanta {_texto_umbral}
                    antes de entrar en zona de riesgo.
                    A partir de ahí, la degradación es rápida y el fallo es casi inevitable.
                </div>
            </div>
            """, unsafe_allow_html=True)
```

**ANTES (tarjeta lateral "Entra en riesgo con"):**
```python
        elif ciclo_umbral:
            st.markdown(f'''
            <div style="background:#1a0808;border:1px solid #ef4444;border-left:4px solid #ef4444;
                 border-radius:8px;padding:14px;margin-bottom:12px">
                <div style="font-size:10px;color:#ef4444;font-weight:600;letter-spacing:0.1em;text-transform:uppercase">
                    Entra en riesgo con
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:#ef4444">
                    {ciclo_umbral}% deterioro
                </div>
                <div style="font-size:10px;color:#7f1d1d;margin-top:4px">
                    nivel de degradación crítico
                </div>
            </div>
            ''', unsafe_allow_html=True)
```

**DESPUÉS:**
```python
        elif ciclo_umbral:
            _etiqueta_arriba = "Entra en riesgo en" if _es_real_deg else "Entra en riesgo con"
            _valor_grande = f"{ciclo_umbral} ciclos" if _es_real_deg else f"{ciclo_umbral}% deterioro"
            st.markdown(f'''
            <div style="background:#1a0808;border:1px solid #ef4444;border-left:4px solid #ef4444;
                 border-radius:8px;padding:14px;margin-bottom:12px">
                <div style="font-size:10px;color:#ef4444;font-weight:600;letter-spacing:0.1em;text-transform:uppercase">
                    {_etiqueta_arriba}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:#ef4444">
                    {_valor_grande}
                </div>
                <div style="font-size:10px;color:#7f1d1d;margin-top:4px">
                    nivel de degradación crítico
                </div>
            </div>
            ''', unsafe_allow_html=True)
```

---

## Resumen de qué cambia en la demo

- **Modo manual / "Motor nuevo" / "Motor en riesgo"** (sliders): sigue igual, curva hipotética 0-100% de deterioro — porque ahí no existe un "futuro real".
- **Modo "Simular motor real"**: ahora la curva y la tarjeta usan los ciclos reales que le quedan al motor (dataset NASA completo), con el eje en "ciclos reales transcurridos" y el mensaje "Entra en riesgo en X ciclos" en vez de "con X% de deterioro". Coincide exactamente con lo que verás luego en la pestaña Predicción si sigues avanzando la simulación.

Prueba primero con el motor #16 pausado en el ciclo 110 — deberías ver que "entra en riesgo alto" en torno a 11 ciclos más (121 - 110 = 11), coincidiendo con el salto real que ya vimos en la tabla.
