import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.facecolor'] = '#0a0e1a'
matplotlib.rcParams['axes.facecolor'] = '#0f1829'
matplotlib.rcParams['axes.edgecolor'] = '#1e2d4a'
matplotlib.rcParams['text.color'] = '#e2e8f0'
matplotlib.rcParams['xtick.color'] = '#64748b'
matplotlib.rcParams['ytick.color'] = '#64748b'
matplotlib.rcParams['axes.labelcolor'] = '#64748b'
matplotlib.rcParams['grid.color'] = '#1e2d4a'
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

st.set_page_config(
    page_title="AeroPredict — Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0a0e1a; }

.stTabs [data-baseweb="tab-list"] {
    background: #0a0e1a;
    border-bottom: 1px solid #1e2d4a;
    padding: 0 16px;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #475569;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 12px 18px;
}
.stTabs [aria-selected="true"] {
    color: #0ea5e9 !important;
    border-bottom: 2px solid #0ea5e9 !important;
    background: transparent !important;
}

.alert-danger {
    background: #1a0808;
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.alert-safe {
    background: #0a1a0e;
    border: 1px solid #22c55e;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.alert-icon { font-size: 2rem; }
.alert-status { font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; font-weight: 600; }
.alert-prob { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; line-height: 1; }
.alert-sub { font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px; opacity: 0.6; }
.alert-actions { font-size: 11px; margin-top: 8px; line-height: 1.8; }

.panel-title {
    font-size: 10px;
    font-weight: 600;
    color: #0ea5e9;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
}

.mrow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #111827;
}
.mrow:last-child { border-bottom: none; }
.mname { font-size: 12px; color: #64748b; }
.mval { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #e2e8f0; }
.mval.good { color: #22c55e; }
.mval.warn { color: #f59e0b; }
.mval.bad  { color: #ef4444; }
.mval.blue { color: #0ea5e9; }

.status-banner {
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    border-left: 3px solid;
}

.pitch-box {
    background: #0a1628;
    border: 1px solid #0ea5e9;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.8;
    font-style: italic;
}
.pitch-box strong { color: #0ea5e9; font-style: normal; }

div[data-testid="stMetric"] {
    background: #0f1829;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 14px 16px;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #0ea5e9 !important;
}
div[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stButton > button {
    background: #0f1829;
    border: 1px solid #1e2d4a;
    color: #64748b;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 6px;
}
.stButton > button:hover { border-color: #0ea5e9; color: #0ea5e9; }

hr { border-color: #1e2d4a !important; }
section[data-testid="stSidebar"] { background: #0f1829; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('models/xgboost_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return modelo, feature_names

@st.cache_data
def cargar_datos_test():
    X = pd.read_csv('data/processed/X_test.csv')
    y = pd.read_csv('data/processed/y_test.csv').values.flatten()
    return X, y

@st.cache_data
def cargar_motor_demo():
    return pd.read_csv('data/processed/motor_demo.csv')

modelo, feature_names = cargar_modelo()
X_test_data, y_test_data = cargar_datos_test()
motor_demo_df = cargar_motor_demo()

# ── Mapeo de sliders a features del modelo ──
SLIDER_FEATURES = ['sl_s11', 'sl_s4', 'sl_s12', 'sl_s7', 'sl_s15']
SENSOR_NAMES    = ['s11',    's4',    's12',    's7',    's15']

def slider_to_fila(valores_slider: dict, modo: str = 'manual') -> pd.DataFrame:
    """
    Convierte valores de sliders en una fila con todas las features del modelo.
    En modo 'riesgo' o 'nuevo', rellena las features no controladas por slider
    con su valor correcto según SHAP_DIRECCION.
    """
    if modo == 'riesgo':
        fila = {col: float(SHAP_DIRECCION.get(col, 1)) * 2.5 for col in feature_names}
    elif modo == 'nuevo':
        fila = {col: float(SHAP_DIRECCION.get(col, 1)) * -2.0 for col in feature_names}
    else:
        fila = {col: 0.0 for col in feature_names}

    # Sobreescribir con los valores reales de los sliders
    for k, s in zip(SLIDER_FEATURES, SENSOR_NAMES):
        v = valores_slider.get(k, 0.0)
        for suffix in [f'{s}_norm', s, f'{s}_norm_mm', f'{s}_media_movil']:
            if suffix in fila:
                fila[suffix] = v
    return pd.DataFrame([fila])[feature_names]

# Dirección de riesgo de cada feature según SHAP values reales (calculados desde +2.5)
# +1 → alto = riesgo,  -1 → bajo = riesgo
SHAP_DIRECCION = {
    's2_norm':     +1, 's3_norm':     +1, 's4_norm':     +1, 's7_norm':     +1,
    's8_norm':     +1, 's9_norm':     +1, 's11_norm':    +1, 's12_norm':    -1,
    's13_norm':    +1, 's14_norm':    +1, 's15_norm':    +1, 's17_norm':    +1,
    's20_norm':    -1, 's21_norm':    -1,
    's2_norm_mm':  +1, 's3_norm_mm':  +1, 's4_norm_mm':  +1, 's7_norm_mm':  -1,
    's8_norm_mm':  +1, 's9_norm_mm':  +1, 's11_norm_mm': +1, 's12_norm_mm': -1,
    's13_norm_mm': -1, 's14_norm_mm': +1, 's15_norm_mm': +1, 's17_norm_mm': +1,
    's20_norm_mm': -1, 's21_norm_mm': -1,
}

def calcular_preset_shap(_modelo, _feature_names, modo: str):
    """Devuelve los valores de slider para cada preset usando las direcciones SHAP reales."""
    preset = {}
    for k, s in zip(SLIDER_FEATURES, SENSOR_NAMES):
        feats = [f for f in _feature_names if f.startswith(s + '_') or f == s]
        if not feats:
            preset[k] = 2.5 if modo == 'riesgo' else -2.0
            continue
        suma = sum(SHAP_DIRECCION.get(f, 1) * (2 if f.endswith('_mm') else 1) for f in feats)
        dir_sensor = +1 if suma >= 0 else -1
        if modo == 'riesgo':
            preset[k] = 2.5 * dir_sensor
        else:
            preset[k] = -2.0 * dir_sensor
    return preset

# Callback para resetear modo cuando el usuario mueve sliders manualmente
def reset_modo():
    st.session_state['modo_preset'] = 'manual'
    st.session_state.pop('fila_preset', None)

# ── HEADER ──
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown("""
    <div style="padding:8px 0 16px">
        <div style="font-size:10px;font-weight:600;color:#0ea5e9;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:4px">
            ✈ AeroPredict · Engine Health Monitor
        </div>
        <div style="font-size:1.5rem;font-weight:600;color:#e2e8f0;line-height:1.2">
            Predictive Maintenance System
        </div>
        <div style="font-size:12px;color:#475569;margin-top:4px">
            NASA C-MAPSS Turbofan Dataset · XGBoost + Optuna + SHAP · Bootcamp Data Science
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style="text-align:right;padding-top:16px">
        <span style="background:#0a2a1a;border:1px solid #22c55e;border-radius:20px;
              padding:4px 12px;font-size:10px;font-weight:600;color:#22c55e;letter-spacing:0.08em">
            ● SISTEMA ACTIVO
        </span>
        <div style="font-size:10px;color:#334155;margin-top:6px;text-align:right">
            AUC-ROC: 0.9940 · Recall: 91%
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:0 0 4px'>", unsafe_allow_html=True)

tab1, tab4, tab2, tab3 = st.tabs(["✈  Predicción", "📈  Degradación del motor", "◈  Rendimiento del modelo", "◆  Valor de negocio"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICCIÓN
# ══════════════════════════════════════════════
with tab1:

    # Inicializar session_state
    for k in SLIDER_FEATURES:
        if k not in st.session_state:
            st.session_state[k] = 0.0

    # Si hay simulación activa, actualizar fila_preset ANTES de calcular datos_motor
    if st.session_state.get('simular_motor', False):
        sim_ciclo_pre = min(st.session_state.get('sim_ciclo', 0), len(motor_demo_df) - 1)
        st.session_state['sim_ciclo'] = sim_ciclo_pre
        # Inicializar historial solo si acaba de empezar (no existía antes)
        if 'sim_historial' not in st.session_state:
            st.session_state['sim_historial'] = []
        fila_sim_pre = motor_demo_df.iloc[sim_ciclo_pre][feature_names].to_dict()
        st.session_state['fila_preset'] = fila_sim_pre
        st.session_state['modo_preset'] = 'real'
        # Sincronizar sliders visualmente
        slider_map_pre = {'sl_s11': 's11_norm', 'sl_s4': 's4_norm',
                         'sl_s12': 's12_norm', 'sl_s7': 's7_norm', 'sl_s15': 's15_norm'}
        for k, f in slider_map_pre.items():
            st.session_state[k] = float(np.clip(fila_sim_pre.get(f, 0), -3.0, 3.0))

    # Construir datos del motor con valores actuales de sliders
    modo_actual = st.session_state.get('modo_preset', 'manual')
    if modo_actual != 'manual' and 'fila_preset' in st.session_state:
        datos_motor = pd.DataFrame([st.session_state['fila_preset']])[feature_names]
    else:
        datos_motor = slider_to_fila(st.session_state, modo='manual')

    proba = modelo.predict_proba(datos_motor)[:, 1]
    prob_pct = float(proba[0] * 100)
    es_riesgo = bool(prob_pct >= 15.0)
    st.session_state['prob_pct'] = prob_pct
    st.session_state['es_riesgo'] = es_riesgo

    # Guardar en historial si simulación activa
    if st.session_state.get('simular_motor', False):
        sim_ciclo_actual = st.session_state.get('sim_ciclo', 0)
        historial = st.session_state.get('sim_historial', [])
        ciclo_num_actual = int(motor_demo_df.iloc[sim_ciclo_actual]['ciclo']) if sim_ciclo_actual < len(motor_demo_df) else sim_ciclo_actual
        # Actualizar o añadir
        if len(historial) <= sim_ciclo_actual:
            historial.append({'ciclo': ciclo_num_actual, 'prob': prob_pct})
        else:
            historial[sim_ciclo_actual] = {'ciclo': ciclo_num_actual, 'prob': prob_pct}
        st.session_state['sim_historial'] = historial


    # ── FILA SUPERIOR: RESULTADO + MÉTRICAS ──
    col_alert, col_m2 = st.columns([3, 1], gap="small")

    with col_alert:
        if es_riesgo:
            st.markdown(f"""
            <div class="alert-danger">
                <div class="alert-icon">⚠</div>
                <div>
                    <div class="alert-status" style="color:#ef4444">EN RIESGO</div>
                    <div class="alert-prob" style="color:#ef4444">{prob_pct:.1f}<span style="font-size:1.2rem">%</span></div>
                    <div class="alert-sub" style="color:#7f1d1d">probabilidad de fallo detectada</div>
                </div>
            </div>
            <div style="margin-top:8px;display:flex;gap:8px">
                <div style="flex:1;background:#3d0000;border:1px solid #ef4444;border-left:4px solid #ef4444;
                     border-radius:6px;padding:10px 14px;font-size:12px;color:#ff6b6b;font-weight:600">
                    🔧 Programar mantenimiento en menos de 30 ciclos
                </div>
                <div style="flex:1;background:#3d0000;border:1px solid #ef4444;border-left:4px solid #ef4444;
                     border-radius:6px;padding:10px 14px;font-size:12px;color:#ff6b6b;font-weight:600">
                    🔍 Revisar sensores de presión HPC y temperatura LPT
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-safe">
                <div class="alert-icon">✓</div>
                <div>
                    <div class="alert-status" style="color:#22c55e">SEGURO</div>
                    <div class="alert-prob" style="color:#22c55e">{prob_pct:.1f}<span style="font-size:1.2rem">%</span></div>
                    <div class="alert-sub" style="color:#14532d">operación normal confirmada</div>
                    <div class="alert-actions" style="color:#86efac">
                        ▸ Seguimiento rutinario sin urgencia<br>
                        ▸ Próxima revisión según calendario estándar
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_m2:
        st.metric("Estado", "⚠ EN RIESGO" if es_riesgo else "✓ SEGURO")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FILA INFERIOR: SENSORES | SHAP ──
    col_sensores, col_shap = st.columns([1, 1], gap="large")

    with col_sensores:
        st.markdown('<div class="panel-title">Panel de control de sensores</div>', unsafe_allow_html=True)

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("✓ Motor nuevo", use_container_width=True):
                preset = calcular_preset_shap(modelo, feature_names, modo='nuevo')
                for k, v in preset.items():
                    st.session_state[k] = v
                fila_preset = slider_to_fila(preset, modo='nuevo')
                st.session_state['fila_preset'] = fila_preset.to_dict('records')[0]
                st.session_state['modo_preset'] = 'nuevo'
                st.session_state.pop('motor_real', None)
                st.rerun()
        with col_b2:
            if st.button("⚠ Motor en riesgo", use_container_width=True):
                preset = calcular_preset_shap(modelo, feature_names, modo='riesgo')
                for k, v in preset.items():
                    st.session_state[k] = v
                fila_preset = slider_to_fila(preset, modo='riesgo')
                st.session_state['fila_preset'] = fila_preset.to_dict('records')[0]
                st.session_state['modo_preset'] = 'riesgo'
                st.session_state.pop('motor_real', None)
                st.rerun()
        with col_b3:
            if st.button("🎬 Simular motor real", use_container_width=True):
                st.session_state['simular_motor'] = True
                st.session_state['sim_ciclo'] = 0
                st.session_state['sim_historial'] = []
                st.session_state.pop('motor_real', None)
                st.rerun()

        # Simulación motor real
        if st.session_state.get('simular_motor', False):
            sim_ciclo = st.session_state.get('sim_ciclo', 0)
            total_ciclos = len(motor_demo_df)

            fila_sim = motor_demo_df.iloc[sim_ciclo][feature_names].to_dict()
            target_sim = int(motor_demo_df.iloc[sim_ciclo]['target'])
            ciclo_num = int(motor_demo_df.iloc[sim_ciclo]['ciclo'])

            # Actualizar sliders
            slider_map = {'sl_s11': 's11_norm', 'sl_s4': 's4_norm',
                         'sl_s12': 's12_norm', 'sl_s7': 's7_norm', 'sl_s15': 's15_norm'}
            for k, f in slider_map.items():
                st.session_state[k] = float(np.clip(fila_sim.get(f, 0), -3.0, 3.0))

            # Guardar fila completa
            st.session_state['fila_preset'] = fila_sim
            st.session_state['modo_preset'] = 'real'

            color_sim = "#ef4444" if target_sim == 1 else "#22c55e"
            bg_sim = "#1a0808" if target_sim == 1 else "#0a1a0e"
            estado_sim = "⚠ EN RIESGO" if target_sim == 1 else "✓ SEGURO"

            st.markdown(f'''
            <div style="background:{bg_sim};border:1px solid {color_sim};border-left:4px solid {color_sim};
                 border-radius:6px;padding:10px 14px;margin-top:8px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-size:10px;color:{color_sim};font-weight:600;letter-spacing:0.1em;text-transform:uppercase">
                            🎬 Motor NASA #69 — Ciclo {ciclo_num} de {int(motor_demo_df["ciclo"].max())}
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{color_sim};font-weight:700;margin-top:2px">
                            Realidad: {estado_sim}
                        </div>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:700;color:{color_sim}">
                        {int((sim_ciclo/total_ciclos)*100)}% completado
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Barra de progreso y controles
            st.progress(sim_ciclo / (total_ciclos - 1))
            col_prev, col_next, col_auto, col_stop = st.columns(4)
            with col_prev:
                if st.button("◀", use_container_width=True, disabled=sim_ciclo==0):
                    st.session_state['sim_ciclo'] = max(0, sim_ciclo - 10)
                    st.session_state['auto_play'] = False
                    st.rerun()
            with col_next:
                if st.button("▶", use_container_width=True, disabled=sim_ciclo>=total_ciclos-1):
                    st.session_state['sim_ciclo'] = min(total_ciclos - 1, sim_ciclo + 10)
                    st.session_state['auto_play'] = False
                    st.rerun()
            with col_auto:
                auto_label = "⏸ Pausar" if st.session_state.get('auto_play', False) else "▶▶ Auto"
                if st.button(auto_label, use_container_width=True):
                    st.session_state['auto_play'] = not st.session_state.get('auto_play', False)
                    st.rerun()
            with col_stop:
                if st.button("✕ Salir", use_container_width=True):
                    st.session_state['simular_motor'] = False
                    st.session_state['auto_play'] = False
                    st.session_state.pop('fila_preset', None)
                    st.session_state['modo_preset'] = 'manual'
                    st.session_state.pop('sim_historial', None)
                    st.session_state['sim_ciclo'] = 0
                    for k in SLIDER_FEATURES:
                        st.session_state[k] = 0.0
                    st.rerun()

            # Auto-avance
            if st.session_state.get('auto_play', False):
                if sim_ciclo < total_ciclos - 1:
                    import time
                    time.sleep(0.05)
                    st.session_state['sim_ciclo'] = sim_ciclo + 5
                    st.rerun()
                else:
                    st.session_state['auto_play'] = False
            # Gráfico evolución + predicción vs realidad
            historial = st.session_state.get('sim_historial', [])
            if len(historial) > 1:
                ciclos_h = [h['ciclo'] for h in historial]
                probs_h  = [h['prob']  for h in historial]

                def _col_h(p):
                    if p < 15:   return '#22c55e'
                    elif p < 50: return '#f59e0b'
                    else:        return '#ef4444'

                ciclo_riesgo_real = int(motor_demo_df[motor_demo_df['target']==1]['ciclo'].min())

                prob_actual_color = _col_h(probs_h[-1])
                st.markdown(f'<p style="font-size:11px;color:#64748b;margin-bottom:4px">🤖 Prob. fallo actual: <span style="color:{prob_actual_color};font-weight:700">{probs_h[-1]:.1f}%</span> &nbsp;·&nbsp; 📋 NASA fallo real en ciclo <span style="color:#ffffff;font-weight:700">{ciclo_riesgo_real}</span></p>', unsafe_allow_html=True)

                plt.close('all')
                fig_sim, ax_sim = plt.subplots(figsize=(6, 3.0))
                fig_sim.patch.set_facecolor('#0a0e1a')
                ax_sim.set_facecolor('#0f1829')
                max_ciclo = int(motor_demo_df['ciclo'].max())

                # Área sombreada roja = zona de fallo real NASA
                ax_sim.axvspan(ciclo_riesgo_real, max_ciclo, alpha=0.15, color='#ef4444', label='Zona fallo NASA')
                ax_sim.axvline(ciclo_riesgo_real, color='#ef4444', linewidth=1.5, linestyle='--', alpha=0.7)
                ax_sim.text(ciclo_riesgo_real+2, 97, f'Fallo NASA c.{ciclo_riesgo_real}', fontsize=6.5, color='#ef4444', alpha=0.8)

                # Línea predicción coloreada
                for i in range(len(ciclos_h)-1):
                    ax_sim.plot(ciclos_h[i:i+2], probs_h[i:i+2],
                               color=_col_h(probs_h[i]), linewidth=2, alpha=0.95)

                # Punto actual
                ax_sim.scatter([ciclos_h[-1]], [probs_h[-1]],
                              color=_col_h(probs_h[-1]), s=60, zorder=5)

                ax_sim.axhline(15, color='#f59e0b', linewidth=0.8, linestyle=':', alpha=0.5)
                ax_sim.set_xlim(1, max_ciclo)
                ax_sim.set_ylim(-3, 108)
                ax_sim.set_xlabel('Ciclo', fontsize=8, color='#64748b')
                ax_sim.set_ylabel('Prob. fallo (%)', fontsize=8, color='#64748b')
                ax_sim.tick_params(colors='#64748b', labelsize=7)
                ax_sim.spines['top'].set_visible(False)
                ax_sim.spines['right'].set_visible(False)
                for spine in ['bottom', 'left']:
                    ax_sim.spines[spine].set_color('#1e2d4a')
                fig_sim.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.22)
                st.pyplot(fig_sim, use_container_width=True, clear_figure=True)
                plt.close('all')

            # Nota explicativa al finalizar la simulación
            sim_idx_check = st.session_state.get('sim_ciclo', 0)
            total_ciclos_check = len(motor_demo_df)
            if sim_idx_check >= total_ciclos_check - 1:
                st.markdown("""
                <div style="background:#0a1628;border:1px solid #0ea5e9;border-left:4px solid #0ea5e9;
                     border-radius:8px;padding:14px 18px;margin:10px 0 6px 0">
                    <div style="font-size:10px;font-weight:600;color:#0ea5e9;letter-spacing:0.12em;
                         text-transform:uppercase;margin-bottom:6px">✈ Simulación completada — Motor NASA #69</div>
                    <div style="font-size:12px;color:#94a3b8;line-height:1.8">
                        El modelo ha reproducido los <strong style="color:#e2e8f0">362 ciclos de vida</strong>
                        de este motor real del dataset NASA C-MAPSS.<br>
                        La NASA certifica el fallo en el <strong style="color:#ef4444">ciclo 333</strong>.
                        Nuestro modelo lo detectó en el ciclo <strong style="color:#f59e0b">~291</strong> —
                        <strong style="color:#22c55e">42 ciclos de antelación</strong>, equivalente a
                        <strong style="color:#22c55e">42 vuelos de margen</strong> para programar mantenimiento
                        antes de que ocurra el fallo.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Panel predicción vs realidad
            sim_idx = st.session_state.get('sim_ciclo', 0)
            target_actual = int(motor_demo_df.iloc[sim_idx]['target'])
            pred_actual = prob_pct >= 15.0
            col_pred, col_real = st.columns(2)
            with col_pred:
                color_p = "#ef4444" if pred_actual else "#22c55e"
                bg_p = "#1a0808" if pred_actual else "#0a1a0e"
                txt_p = "⚠ EN RIESGO" if pred_actual else "✓ SEGURO"
                st.markdown(f'''
                <div style="background:{bg_p};border:1px solid {color_p};border-radius:8px;padding:8px;text-align:center">
                    <div style="font-size:9px;color:#475569;text-transform:uppercase">🤖 Predicción</div>
                    <div style="font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{color_p}">{txt_p}</div>
                    <div style="font-size:10px;color:{color_p}">{prob_pct:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
            with col_real:
                color_r = "#ef4444" if target_actual else "#22c55e"
                bg_r = "#1a0808" if target_actual else "#0a1a0e"
                txt_r = "⚠ EN RIESGO" if target_actual else "✓ SEGURO"
                acierto = "✓ CORRECTO" if pred_actual == bool(target_actual) else "✗ ERROR"
                color_a = "#22c55e" if pred_actual == bool(target_actual) else "#ef4444"
                st.markdown(f'''
                <div style="background:{bg_r};border:1px solid {color_r};border-radius:8px;padding:8px;text-align:center">
                    <div style="font-size:9px;color:#475569;text-transform:uppercase">📋 Realidad NASA</div>
                    <div style="font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{color_r}">{txt_r}</div>
                    <div style="font-size:10px;color:{color_a};font-weight:600">{acierto}</div>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown("---")

        # ── Umbrales configurables ──
        with st.expander("⚙ Configurar umbrales de alerta", expanded=False):
            st.markdown('''<div style="font-size:11px;color:#64748b;margin-bottom:8px">
            Ajusta cuándo un sensor se marca como degradado (🟡) o en fallo (🔴).
            Valores más bajos = más sensible. Valores más altos = más conservador.
            </div>''', unsafe_allow_html=True)
            umbral_amarillo = st.slider("🟡 Umbral degradación (amarillo)", 
                                        min_value=0.01, max_value=0.5, value=0.05, step=0.01,
                                        help="SHAP mínimo para marcar sensor en amarillo")
            umbral_rojo = st.slider("🔴 Umbral fallo activo (rojo)", 
                                    min_value=0.1, max_value=1.0, value=0.3, step=0.05,
                                    help="SHAP mínimo para marcar sensor en rojo (solo cuando motor en riesgo)")

        st.markdown("**🌡 Temperatura y presión**")

        # Colorear sensores según SHAP values reales
        _explainer_p = shap.TreeExplainer(modelo)
        _sv_p = _explainer_p.shap_values(datos_motor)[0]
        def _shap_s(s):
            idxs = [i for i, f in enumerate(feature_names) if (f.startswith(s+'_') or f == s) and f != 'ciclo']
            return sum(_sv_p[i] for i in idxs)
        def _col(s):
            v = _shap_s(s)
            if es_riesgo and v > umbral_rojo:    return "#ef4444"
            elif v > umbral_amarillo:            return "#f59e0b"
            else:                                return "#e2e8f0"
        def _ale(s):
            v = _shap_s(s)
            if es_riesgo and v > umbral_rojo:    return " 🔴"
            elif v > umbral_amarillo:            return " 🟡"
            else:                                return ""

        sim_activa = st.session_state.get('simular_motor', False)

        st.markdown(f'<p style="font-size:13px;color:{_col("s11")};margin-bottom:-10px;font-weight:500">Presión estática HPC · s11 {_ale("s11")} <span title="Presión estática de salida del compresor de alta presión" style="cursor:help;color:#475569;font-size:11px">❓</span></p>', unsafe_allow_html=True)
        if sim_activa:
            v11 = float(st.session_state.get('sl_s11', 0.0))
            st.markdown(f'<div style="background:#0f1829;border:1px solid #1e2d4a;border-radius:6px;padding:6px 12px;font-family:JetBrains Mono,monospace;color:#0ea5e9;font-size:13px">{v11:.2f}</div>', unsafe_allow_html=True)
        else:
            st.slider("s11", -3.0, 3.0, step=0.1, key='sl_s11', label_visibility="collapsed", on_change=reset_modo)

        st.markdown(f'<p style="font-size:13px;color:{_col("s4")};margin-bottom:-10px;font-weight:500">Temperatura salida LPT · s4 {_ale("s4")} <span title="Temperatura de salida de la turbina de baja presión" style="cursor:help;color:#475569;font-size:11px">❓</span></p>', unsafe_allow_html=True)
        if sim_activa:
            v4 = float(st.session_state.get('sl_s4', 0.0))
            st.markdown(f'<div style="background:#0f1829;border:1px solid #1e2d4a;border-radius:6px;padding:6px 12px;font-family:JetBrains Mono,monospace;color:#0ea5e9;font-size:13px">{v4:.2f}</div>', unsafe_allow_html=True)
        else:
            st.slider("s4", -3.0, 3.0, step=0.1, key='sl_s4', label_visibility="collapsed", on_change=reset_modo)

        st.markdown(f'<p style="font-size:13px;color:{_col("s12")};margin-bottom:-10px;font-weight:500">Ratio flujo combustible · s12 {_ale("s12")} <span title="Ratio de flujo de combustible respecto a Ps30" style="cursor:help;color:#475569;font-size:11px">❓</span></p>', unsafe_allow_html=True)
        if sim_activa:
            v12 = float(st.session_state.get('sl_s12', 0.0))
            st.markdown(f'<div style="background:#0f1829;border:1px solid #1e2d4a;border-radius:6px;padding:6px 12px;font-family:JetBrains Mono,monospace;color:#0ea5e9;font-size:13px">{v12:.2f}</div>', unsafe_allow_html=True)
        else:
            st.slider("s12", -3.0, 3.0, step=0.1, key='sl_s12', label_visibility="collapsed", on_change=reset_modo)

        st.markdown("**⚙ Velocidad y bypass**")
        st.markdown(f'<p style="font-size:13px;color:{_col("s7")};margin-bottom:-10px;font-weight:500">Presión salida HPC · s7 {_ale("s7")} <span title="Presión de salida del compresor de alta presión" style="cursor:help;color:#475569;font-size:11px">❓</span></p>', unsafe_allow_html=True)
        if sim_activa:
            v7 = float(st.session_state.get('sl_s7', 0.0))
            st.markdown(f'<div style="background:#0f1829;border:1px solid #1e2d4a;border-radius:6px;padding:6px 12px;font-family:JetBrains Mono,monospace;color:#0ea5e9;font-size:13px">{v7:.2f}</div>', unsafe_allow_html=True)
        else:
            st.slider("s7", -3.0, 3.0, step=0.1, key='sl_s7', label_visibility="collapsed", on_change=reset_modo)

        st.markdown(f'<p style="font-size:13px;color:{_col("s15")};margin-bottom:-10px;font-weight:500">Ratio de bypass · s15 {_ale("s15")} <span title="Ratio de bypass del motor" style="cursor:help;color:#475569;font-size:11px">❓</span></p>', unsafe_allow_html=True)
        if sim_activa:
            v15 = float(st.session_state.get('sl_s15', 0.0))
            st.markdown(f'<div style="background:#0f1829;border:1px solid #1e2d4a;border-radius:6px;padding:6px 12px;font-family:JetBrains Mono,monospace;color:#0ea5e9;font-size:13px">{v15:.2f}</div>', unsafe_allow_html=True)
        else:
            st.slider("s15", -3.0, 3.0, step=0.1, key='sl_s15', label_visibility="collapsed", on_change=reset_modo)


        st.markdown("---")
        archivo = st.file_uploader("O sube un CSV con datos reales", type=['csv'],
                                  help="El CSV debe tener las mismas columnas que el dataset de entrenamiento")
        if archivo:
            datos_csv = pd.read_csv(archivo)
            st.success(f"✓ {len(datos_csv)} registros cargados")

    with col_shap:
        st.markdown('<div class="panel-title">Explicabilidad SHAP — factores de la predicción</div>', unsafe_allow_html=True)
        st.caption("Barras rojas → empujan hacia EN RIESGO · Barras azules → empujan hacia SEGURO")

        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(datos_motor)
        # Renombrar features para el público: s11_norm → s11(val) · s11_norm_mm → s11(tend)
        def _renombrar(f):
            if f.endswith('_norm_mm'):
                return f.replace('_norm_mm', ' (tendencia)')
            elif f.endswith('_norm'):
                return f.replace('_norm', ' (valor)')
            return f
        feature_names_legibles = [_renombrar(f) for f in datos_motor.columns.tolist()]
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=float(explainer.expected_value),
            data=datos_motor.iloc[0],
            feature_names=feature_names_legibles
        )
        # Limpiar cualquier figura anterior antes de crear la nueva
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#0a0e1a')
        shap.plots.waterfall(explanation, show=False, max_display=min(8, len(feature_names)))
        fig.set_size_inches(7, 5)
        st.pyplot(fig, use_container_width=True, clear_figure=True)
        plt.close('all')

        st.markdown("""
        <div style="background:#0a1628;border:1px solid #1e2d4a;border-left:3px solid #0ea5e9;
             border-radius:6px;padding:10px 14px;margin-top:10px;font-size:11px;color:#64748b;line-height:1.7">
            <span style="color:#0ea5e9;font-weight:600">ℹ El simulador</span> controla los 5 sensores más críticos.
            El gráfico SHAP muestra cómo el modelo evalúa las 28 features reales —
            incluyendo <span style="color:#94a3b8">medias móviles</span> que capturan
            la tendencia de degradación de cada sensor en los últimos 10 ciclos,
            no solo su valor puntual.
        </div>
        """, unsafe_allow_html=True)



        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Tendencia del sensor — valor puntual vs media móvil</div>', unsafe_allow_html=True)
        st.caption("Compara cómo el valor instantáneo y la tendencia de los últimos 10 ciclos divergen durante la degradación")

        # Selector de sensor
        sensor_opciones = {
            's11 — Presión estática HPC': 's11',
            's4  — Temperatura salida LPT': 's4',
            's12 — Ratio flujo combustible': 's12',
            's7  — Presión salida HPC': 's7',
            's15 — Ratio de bypass': 's15',
        }
        sensor_label = st.selectbox("Selecciona sensor", list(sensor_opciones.keys()),
                                    label_visibility="collapsed")
        sensor_sel = sensor_opciones[sensor_label]

        # Valor actual del slider de ese sensor
        slider_key = f'sl_{sensor_sel}'
        val_actual = float(st.session_state.get(slider_key, 0.0))

        # Simular 50 ciclos: degradación progresiva desde motor nuevo hasta valor actual
        n_ciclos = 50
        ciclos_sim = list(range(1, n_ciclos + 1))

        # Valor puntual: empieza en -2 (nuevo) y llega al valor actual con algo de ruido
        np.random.seed(42)
        val_inicio = -2.0 if val_actual >= 0 else 2.0
        valores_puntuales = np.linspace(val_inicio, val_actual, n_ciclos) + np.random.normal(0, 0.08, n_ciclos)

        # Media móvil de ventana 10
        ventana = 10
        valores_mm = []
        for i in range(n_ciclos):
            inicio = max(0, i - ventana + 1)
            valores_mm.append(float(np.mean(valores_puntuales[inicio:i+1])))

        fig5, ax5 = plt.subplots(figsize=(7, 3.5))

        # Zona de riesgo del sensor
        umbral_sensor = 1.5 if val_actual >= 0 else -1.5
        if val_actual >= 0:
            ax5.axhspan(umbral_sensor, 3.5, alpha=0.06, color='#ef4444')
            ax5.axhline(umbral_sensor, color='#ef4444', linewidth=0.8, linestyle=':', alpha=0.5)
        else:
            ax5.axhspan(-3.5, umbral_sensor, alpha=0.06, color='#ef4444')
            ax5.axhline(umbral_sensor, color='#ef4444', linewidth=0.8, linestyle=':', alpha=0.5)

        ax5.plot(ciclos_sim, valores_puntuales, color='#0ea5e9', linewidth=1.5,
                alpha=0.6, linestyle='--', label=f'{sensor_sel}_norm  (valor puntual)')
        ax5.plot(ciclos_sim, valores_mm, color='#f59e0b', linewidth=2.5,
                label=f'{sensor_sel}_norm_mm  (media móvil 10 ciclos)')

        # Punto actual
        ax5.scatter([n_ciclos], [val_actual], color='#ef4444' if es_riesgo else '#22c55e',
                   s=80, zorder=5)

        ax5.set_xlabel('Ciclo', fontsize=9)
        ax5.set_ylabel('Valor normalizado', fontsize=9)
        ax5.set_xlim(1, n_ciclos)
        ax5.legend(fontsize=8, loc='upper left')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

        # Etiqueta zona riesgo
        ax5.text(n_ciclos - 1, umbral_sensor + (0.15 if val_actual >= 0 else -0.25),
                'zona riesgo', ha='right', fontsize=8, color='#ef4444', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        st.markdown('''
        <div style="background:#0a1628;border:1px solid #1e2d4a;border-left:3px solid #f59e0b;
             border-radius:6px;padding:10px 14px;font-size:11px;color:#64748b;line-height:1.7;margin-top:6px">
            <span style="color:#f59e0b;font-weight:600">⚡ Clave:</span>
            La <span style="color:#f59e0b">media móvil (naranja)</span> suaviza el ruido y revela la tendencia real de degradación.
            El modelo usa <strong style="color:#e2e8f0">ambas señales</strong> para distinguir una anomalía puntual
            de una degradación sostenida.
        </div>
        ''', unsafe_allow_html=True)

        # Gráfico de evolución durante simulación
        if st.session_state.get('simular_motor', False):
            historial = st.session_state.get('sim_historial', [])
            if len(historial) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="panel-title">📊 Evolución de riesgo — simulación en tiempo real</div>', unsafe_allow_html=True)

                ciclos_h = [h['ciclo'] for h in historial]
                probs_h  = [h['prob']  for h in historial]

                fig_h, ax_h = plt.subplots(figsize=(7, 3))
                ax_h.axhspan(0,  15, alpha=0.08, color='#22c55e')
                ax_h.axhspan(15, 50, alpha=0.06, color='#f59e0b')
                ax_h.axhspan(50, 100, alpha=0.06, color='#ef4444')
                ax_h.axhline(15, color='#f59e0b', linewidth=0.8, linestyle='--', alpha=0.5)
                ax_h.axhline(50, color='#ef4444', linewidth=0.8, linestyle='--', alpha=0.5)

                def _col_h(p):
                    if p < 15:   return '#22c55e'
                    elif p < 50: return '#f59e0b'
                    else:        return '#ef4444'

                for i in range(len(ciclos_h) - 1):
                    ax_h.plot(ciclos_h[i:i+2], probs_h[i:i+2],
                             color=_col_h(probs_h[i]), linewidth=2, alpha=0.9)

                # Punto actual
                ax_h.scatter([ciclos_h[-1]], [probs_h[-1]],
                            color=_col_h(probs_h[-1]), s=60, zorder=5)

                ax_h.set_xlabel('Ciclo', fontsize=9)
                ax_h.set_ylabel('Prob. fallo (%)', fontsize=9)
                ax_h.set_ylim(-5, 100)
                ax_h.set_xlim(1, int(motor_demo_df['ciclo'].max()))
                ax_h.spines['top'].set_visible(False)
                ax_h.spines['right'].set_visible(False)
                # Línea vertical donde empieza el riesgo REAL (RUL < 30)
                ciclo_riesgo_real = int(motor_demo_df[motor_demo_df['target']==1]['ciclo'].min())
                ax_h.axvline(ciclo_riesgo_real, color='#ffffff', linewidth=1.2,
                            linestyle='--', alpha=0.4, label=f'Riesgo real (ciclo {ciclo_riesgo_real})')
                ax_h.text(ciclo_riesgo_real + 3, 85, f'Real: EN RIESGO\n(ciclo {ciclo_riesgo_real})',
                         fontsize=7, color='#ffffff', alpha=0.6)

                ax_h.legend(fontsize=7, loc='upper left')
                ax_h.text(int(motor_demo_df['ciclo'].max())-5, 7,  'SEGURO',    ha='right', fontsize=8, color='#22c55e', alpha=0.7)
                ax_h.text(int(motor_demo_df['ciclo'].max())-5, 30, 'ALERTA',    ha='right', fontsize=8, color='#f59e0b', alpha=0.7)
                ax_h.text(int(motor_demo_df['ciclo'].max())-5, 70, 'EN RIESGO', ha='right', fontsize=8, color='#ef4444', alpha=0.7)
                st.pyplot(fig_h)
                plt.close(fig_h)

                # Panel predicción vs realidad
                sim_idx = st.session_state.get('sim_ciclo', 0)
                target_actual = int(motor_demo_df.iloc[sim_idx]['target'])
                pred_actual = prob_pct >= 15.0

                col_pred, col_real = st.columns(2)
                with col_pred:
                    color_p = "#ef4444" if pred_actual else "#22c55e"
                    bg_p = "#1a0808" if pred_actual else "#0a1a0e"
                    txt_p = "⚠ EN RIESGO" if pred_actual else "✓ SEGURO"
                    st.markdown(f'''
                    <div style="background:{bg_p};border:1px solid {color_p};border-radius:8px;padding:10px;text-align:center">
                        <div style="font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:0.1em">🤖 Predicción modelo</div>
                        <div style="font-family:JetBrains Mono,monospace;font-size:14px;font-weight:700;color:{color_p};margin-top:4px">{txt_p}</div>
                        <div style="font-size:11px;color:{color_p};opacity:0.8">{prob_pct:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with col_real:
                    color_r = "#ef4444" if target_actual else "#22c55e"
                    bg_r = "#1a0808" if target_actual else "#0a1a0e"
                    txt_r = "⚠ EN RIESGO" if target_actual else "✓ SEGURO"
                    acierto = "✓ CORRECTO" if pred_actual == bool(target_actual) else "✗ ERROR"
                    color_a = "#22c55e" if pred_actual == bool(target_actual) else "#ef4444"
                    st.markdown(f'''
                    <div style="background:{bg_r};border:1px solid {color_r};border-radius:8px;padding:10px;text-align:center">
                        <div style="font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:0.1em">📋 Realidad NASA</div>
                        <div style="font-family:JetBrains Mono,monospace;font-size:14px;font-weight:700;color:{color_r};margin-top:4px">{txt_r}</div>
                        <div style="font-size:11px;color:{color_a};font-weight:600">{acierto}</div>
                    </div>
                    ''', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — RENDIMIENTO
# ══════════════════════════════════════════════
with tab2:
    if 'prob_pct' in st.session_state:
        color = "#ef4444" if st.session_state['es_riesgo'] else "#22c55e"
        bg = "#1a0808" if st.session_state['es_riesgo'] else "#0a1a0e"
        estado = "EN RIESGO" if st.session_state['es_riesgo'] else "SEGURO"
        st.markdown(f"""
        <div class="status-banner" style="background:{bg};border-color:{color};color:{color}">
            Motor analizado → <strong>{estado}</strong> &nbsp;·&nbsp; Probabilidad: {st.session_state['prob_pct']:.1f}%
        </div>
        """, unsafe_allow_html=True)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("AUC-ROC", "0.9940")
    col_r2.metric("Recall", "91.4%", help="% de fallos reales detectados — métrica crítica")
    col_r3.metric("Precision", "88.2%")
    col_r4.metric("F1-Score", "89.7%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="panel-title">Comparativa de modelos</div>', unsafe_allow_html=True)
        for nombre, tipo, auc, rec, prec, cls in [
            ("Regresión Logística", "Baseline", "0.9901", "95.0%", "72.6%", "warn"),
            ("XGBoost", "Sin optimizar", "0.9963", "92.7%", "92.1%", "good"),
            ("XGBoost + Optuna", "Modelo final ★", "0.9940", "91.4%", "88.2%", "blue"),
        ]:
            st.markdown(f"""
            <div class="mrow">
                <div>
                    <div style="font-size:13px;color:#e2e8f0;font-weight:500">{nombre}</div>
                    <div style="font-size:10px;color:#475569;margin-top:2px">{tipo}</div>
                </div>
                <div style="text-align:right">
                    <div class="mval {cls}">AUC {auc}</div>
                    <div style="font-size:10px;color:#475569;margin-top:2px">R:{rec} · P:{prec}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">¿Por qué priorizamos el Recall?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px;color:#64748b;line-height:1.8;">
        En aviación, un <span style="color:#ef4444">Falso Negativo</span> — motor en riesgo no detectado —
        puede costar vidas y millones.<br><br>
        Un <span style="color:#f59e0b">Falso Positivo</span> — falsa alarma —
        solo genera un mantenimiento innecesario.<br><br>
        Por eso optimizamos para <span style="color:#22c55e">maximizar el Recall</span>.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel-title">Matriz de confusión — conjunto de test</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        cm = np.array([[3476, 51], [37, 563]])
        cell_bg = [['#0a1a0e', '#1a0808'], ['#1a0808', '#0a1a0e']]
        for i in range(2):
            for j in range(2):
                ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color=cell_bg[i][j], zorder=0))
        ax2.set_xlim(-0.5, 1.5); ax2.set_ylim(-0.5, 1.5)
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Pred: Seguro', 'Pred: Riesgo'], fontsize=10)
        ax2.set_yticklabels(['Real: Seguro', 'Real: Riesgo'], fontsize=10)
        cell_colors = [['#22c55e', '#ef4444'], ['#ef4444', '#22c55e']]
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=18, fontweight='700', color=cell_colors[i][j], zorder=1)
        if 'es_riesgo' in st.session_state:
            col_pred = 1 if st.session_state['es_riesgo'] else 0
            rect = plt.Rectangle((col_pred-0.5, -0.5), 1, 2,
                                 fill=False, edgecolor='#0ea5e9', linewidth=2.5, linestyle='--', zorder=2)
            ax2.add_patch(rect)
        ax2.set_title('Columna azul = predicción del motor actual', fontsize=9, color='#64748b', pad=10)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Stack tecnológico</div>', unsafe_allow_html=True)
        for k, v in [("Dataset","NASA C-MAPSS FD001"),("Modelo","XGBoost + Optuna (bayesiano)"),
                     ("Explicabilidad","SHAP TreeExplainer"),("Validación","CV estratificada · 5 folds")]:
            st.markdown(f'<div class="mrow"><span class="mname">{k}</span><span class="mval">{v}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — VALOR DE NEGOCIO
# ══════════════════════════════════════════════
with tab3:
    if 'prob_pct' in st.session_state:
        color = "#ef4444" if st.session_state['es_riesgo'] else "#22c55e"
        bg = "#1a0808" if st.session_state['es_riesgo'] else "#0a1a0e"
        msg = "EN RIESGO — coste estimado si falla: $500K" if st.session_state['es_riesgo'] else "SEGURO — sin coste adicional previsto"
        st.markdown(f"""
        <div class="status-banner" style="background:{bg};border-color:{color};color:{color}">
            Motor analizado → <strong>{msg}</strong>
        </div>
        """, unsafe_allow_html=True)

    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Valor neto del modelo", "+$145.3M")
    col_v2.metric("Ahorro vs sin modelo", "+$445.3M")
    col_v3.metric("Fallos evitados", "563 / 600", delta="Recall 91.4%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_tbl, col_chart = st.columns([1, 2], gap="large")

    with col_tbl:
        st.markdown('<div class="panel-title">Expected Value por categoría</div>', unsafe_allow_html=True)
        for nombre, detalle, total, cls in [
            ("✅ Verdaderos Positivos", "563 casos · +$300K c/u", "+$168.9M", "good"),
            ("⬜ Verdaderos Negativos", "3.476 casos · $0", "$0", ""),
            ("⚠️ Falsos Positivos", "51 casos · -$100K c/u", "-$5.1M", "warn"),
            ("🔴 Falsos Negativos", "37 casos · -$500K c/u", "-$18.5M", "bad"),
        ]:
            st.markdown(f"""
            <div class="mrow">
                <div>
                    <div style="font-size:12px;color:#e2e8f0;font-weight:500">{nombre}</div>
                    <div style="font-size:10px;color:#475569;margin-top:2px">{detalle}</div>
                </div>
                <div class="mval {cls}" style="font-size:14px">{total}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Fuente metodológica</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:11px;color:#475569;line-height:1.8;">
            <strong style="color:#64748b">Expected Value Framework</strong><br>
            Provost & Fawcett — Data Science for Business<br><br>
            Valores estimados basados en costes reales de la industria aeronáutica:
            AOG, reparaciones mayores e indemnizaciones.
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="panel-title">Impacto económico</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        conceptos = ['TP ×563\nfallos evitados', 'FP ×51\nfalsas alarmas', 'FN ×37\nno detectados', 'TOTAL\nneto']
        valores = [168_900_000, -5_100_000, -18_500_000, 145_300_000]
        colores = ['#22c55e', '#f59e0b', '#ef4444', '#0ea5e9']
        barras = ax3.bar(conceptos, [v/1e6 for v in valores], color=colores, alpha=0.85, width=0.5)
        ax3.axhline(0, color='#1e2d4a', linewidth=1)
        ax3.set_ylabel('Millones de dólares ($M)', fontsize=10)
        ax3.set_ylim(-40, 210)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        for barra, valor in zip(barras, valores):
            h = barra.get_height()
            ax3.text(barra.get_x() + barra.get_width()/2,
                    h + (4 if h >= 0 else -10),
                    f'${valor/1e6:+.1f}M',
                    ha='center', va='bottom', fontweight='bold', fontsize=10, color='#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Pitch para la presentación</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pitch-box">
    "El modelo genera un valor estimado de <strong>$145 millones</strong> detectando
    <strong>563 de 600 fallos</strong> antes de que ocurran — un Recall del 93%.
    Comparado con no tener ningún sistema predictivo, el ahorro estimado es de
    <strong>$445 millones de dólares</strong>."
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — DEGRADACIÓN POR CICLO
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
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

        # Simular degradación progresiva de los sensores
        n_pasos = 100
        pasos_eje = list(range(0, 101, 1))  # 0% a 100% de degradación
        probas_curva = []

        modo_curva = st.session_state.get('modo_preset', 'manual')
        fila_actual = slider_to_fila(st.session_state, modo=modo_curva)

        for pct in pasos_eje:
            t = pct / 100.0  # 0.0 a 1.0
            fila_deg = fila_actual.copy()
            for col_f in feature_names:
                dir_f = SHAP_DIRECCION.get(col_f, 1)
                val_actual_f = float(fila_actual[col_f].iloc[0])
                val_riesgo_f = 2.5 * dir_f
                fila_deg[col_f] = val_actual_f + t * (val_riesgo_f - val_actual_f)
            p = float(modelo.predict_proba(fila_deg)[:, 1][0] * 100)
            probas_curva.append(p)

        # Suavizar curva con media móvil
        ventana_suav = 8
        probas_suav = []
        for i in range(len(probas_curva)):
            inicio = max(0, i - ventana_suav + 1)
            probas_suav.append(float(np.mean(probas_curva[inicio:i+1])))
        probas_curva = probas_suav

        prob_actual = probas_curva[0]
        ciclo_actual = 0  # punto de partida

        fig4, ax4 = plt.subplots(figsize=(9, 5))

        # Zonas seguro / alerta / riesgo
        ax4.axhspan(0, 15, alpha=0.08, color='#22c55e')
        ax4.axhspan(15, 50, alpha=0.06, color='#f59e0b')
        ax4.axhspan(50, 100, alpha=0.06, color='#ef4444')
        ax4.axhline(15, color='#f59e0b', linewidth=1, linestyle='--', alpha=0.6, label='Umbral alerta (15%)')
        ax4.axhline(50, color='#ef4444', linewidth=1, linestyle='--', alpha=0.6, label='Umbral riesgo alto (50%)')

        # Curva degradación — verde/amarillo/rojo
        def _color_curva(p):
            if p < 15:   return '#22c55e'  # verde — seguro
            elif p < 50: return '#f59e0b'  # amarillo — alerta
            else:        return '#ef4444'  # rojo — riesgo alto

        colores_linea = [_color_curva(p) for p in probas_curva]
        for i in range(len(pasos_eje) - 1):
            ax4.plot(pasos_eje[i:i+2], probas_curva[i:i+2],
                    color=colores_linea[i], linewidth=2.5, alpha=0.9)

        # Punto actual
        ax4.scatter([0], [prob_actual],
                   color='#f59e0b', s=120, zorder=5, label=f'Estado actual ({prob_actual:.1f}%)')
        ax4.annotate(f'{prob_actual:.1f}%',
                    xy=(0, prob_actual),
                    xytext=(10, prob_actual + 5),
                    color='#f59e0b', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1.5))

        ax4.set_xlabel('% de degradación de sensores', fontsize=11)
        ax4.set_ylabel('Probabilidad de fallo (%)', fontsize=11)
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        ax4.legend(fontsize=9, loc='upper left')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        # Etiquetas zona
        ax4.text(98, 7,  'SEGURO',      ha='right', va='center', fontsize=9, color='#22c55e', fontweight='600', alpha=0.7)
        ax4.text(98, 30, 'ALERTA',      ha='right', va='center', fontsize=9, color='#f59e0b', fontweight='600', alpha=0.7)
        ax4.text(98, 70, 'EN RIESGO',   ha='right', va='center', fontsize=9, color='#ef4444', fontweight='600', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # Nota dinámica debajo del gráfico
        ciclo_umbral_nota = None
        for i, p in enumerate(probas_curva):
            if p >= 15:
                ciclo_umbral_nota = pasos_eje[i]
                break

        if ciclo_umbral_nota is not None and prob_actual < 15:
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
        elif prob_actual >= 15:
            st.markdown(f"""
            <div style="background:#1a0808;border:1px solid #ef4444;border-left:4px solid #ef4444;
                 border-radius:8px;padding:12px 18px;margin-top:8px">
                <div style="font-size:11px;color:#94a3b8;line-height:1.9">
                    Este motor <strong style="color:#ef4444">ya se encuentra en zona de riesgo</strong>
                    con su estado actual ({prob_actual:.1f}%).
                    Cualquier deterioro adicional acelera el camino al fallo.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="panel-title">Lectura del gráfico</div>', unsafe_allow_html=True)

        # Ciclo donde cruza el umbral
        ciclo_umbral = None
        for i, (c, p) in enumerate(zip(pasos_eje, probas_curva)):
            if p >= 15:
                ciclo_umbral = c
                break

        if prob_actual >= 15:
            st.markdown(f'''
            <div style="background:#1a0808;border:1px solid #ef4444;border-left:4px solid #ef4444;
                 border-radius:8px;padding:14px;margin-bottom:12px">
                <div style="font-size:10px;color:#ef4444;font-weight:600;letter-spacing:0.1em;text-transform:uppercase">
                    Estado crítico
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:#ef4444">
                    Ya en riesgo ⚠
                </div>
                <div style="font-size:10px;color:#7f1d1d;margin-top:4px">
                    mantenimiento urgente requerido
                </div>
            </div>
            ''', unsafe_allow_html=True)
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
        else:
            st.markdown('''
            <div style="background:#0a1a0e;border:1px solid #22c55e;border-left:4px solid #22c55e;
                 border-radius:8px;padding:14px;margin-bottom:12px">
                <div style="font-size:10px;color:#22c55e;font-weight:600;letter-spacing:0.1em;text-transform:uppercase">
                    Estado
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:700;color:#22c55e">
                    Motor muy robusto ✓
                </div>
                <div style="font-size:10px;color:#14532d;margin-top:4px">
                    resiste degradación total sin fallar
                </div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<div class="panel-title" style="margin-top:16px">Probabilidad actual</div>', unsafe_allow_html=True)
        color_prob = "#ef4444" if prob_actual >= 15 else "#22c55e"
        st.markdown(f'''
        <div style="text-align:center;padding:16px 0">
            <div style="font-family:'JetBrains Mono',monospace;font-size:2.5rem;font-weight:700;color:{color_prob}">
                {prob_actual:.1f}%
            </div>
            <div style="font-size:10px;color:#475569;margin-top:4px">estado actual</div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('<div class="panel-title" style="margin-top:8px">Cómo leer la curva</div>', unsafe_allow_html=True)
        st.markdown('''
        <div style="font-size:11px;color:#64748b;line-height:1.9">
            <span style="color:#22c55e">━</span> Seguro · prob &lt; 15%<br>
            <span style="color:#f59e0b">━</span> Alerta · prob 15-50%<br>
            <span style="color:#ef4444">━</span> Riesgo alto · prob &gt; 50%<br>
            <span style="color:#f59e0b">●</span> Estado actual del motor<br><br>
            El eje X muestra el % de degradación<br>
            desde el estado actual (0%) hasta<br>
            el máximo deterioro posible (100%).
        </div>
        ''', unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:10px;color:#1e2d4a;font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;">
AEROPREDICT · BOOTCAMP DATA SCIENCE · XGBOOST + OPTUNA + SHAP · NASA C-MAPSS
</div>
""", unsafe_allow_html=True)
