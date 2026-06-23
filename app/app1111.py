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

modelo, feature_names = cargar_modelo()

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
            AUC-ROC: 0.9964 · Recall: 93%
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:0 0 4px'>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["✈  Predicción", "◈  Rendimiento del modelo", "◆  Valor de negocio"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICCIÓN
# ══════════════════════════════════════════════
with tab1:

    # Inicializar session_state
    for k in ['sl_s11','sl_s4','sl_s12','sl_s7','sl_s15']:
        if k not in st.session_state:
            st.session_state[k] = 0.0

    # Construir datos del motor
    fila = {col: 0.0 for col in feature_names}
    for k, s in [('sl_s11','s11'),('sl_s4','s4'),('sl_s12','s12'),('sl_s7','s7'),('sl_s15','s15')]:
        v = st.session_state.get(k, 0.0)
        fila[s] = v; fila[f'{s}_media_movil'] = v
    datos_motor = pd.DataFrame([fila])[feature_names]

    proba = modelo.predict_proba(datos_motor)[:, 1]
    prob_pct = float(proba[0] * 100)
    es_riesgo = bool(prob_pct >= 15.0)
    st.session_state['prob_pct'] = prob_pct
    st.session_state['es_riesgo'] = es_riesgo

    # ── FILA SUPERIOR: RESULTADO + MÉTRICAS ──
    col_alert, col_m1, col_m2 = st.columns([3, 1, 1], gap="small")

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

    with col_m1:
        st.metric("Ciclo operacional", st.session_state.get('ciclo', 150))

    with col_m2:
        st.metric("Ciclos restantes", "< 30 ⚠" if es_riesgo else "> 30 ✓")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FILA INFERIOR: SENSORES | SHAP ──
    col_sensores, col_shap = st.columns([1, 1], gap="large")

    with col_sensores:
        st.markdown('<div class="panel-title">Panel de control de sensores</div>', unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("✓ Motor nuevo", use_container_width=True):
                for k, v in [('sl_s11',-2.0),('sl_s4',-1.8),('sl_s12',1.5),('sl_s7',-1.5),('sl_s15',-1.6)]:
                    st.session_state[k] = v
                st.rerun()
        with col_b2:
            if st.button("⚠ Motor en riesgo", use_container_width=True):
                for k, v in [('sl_s11',2.5),('sl_s4',2.2),('sl_s12',-2.3),('sl_s7',2.0),('sl_s15',2.1)]:
                    st.session_state[k] = v
                st.rerun()

        st.markdown("---")
        st.markdown("**🌡 Temperatura y presión**")

        # Color rojo en los sensores críticos cuando hay riesgo
        color_s11 = "#ef4444" if es_riesgo else "#e2e8f0"
        color_s4  = "#ef4444" if es_riesgo else "#e2e8f0"
        alerta_s11 = " 🔴" if es_riesgo else ""
        alerta_s4  = " 🔴" if es_riesgo else ""

        st.markdown(f'<p style="font-size:13px;color:{color_s11};margin-bottom:-10px;font-weight:500">Presión estática HPC · s11{alerta_s11}</p>', unsafe_allow_html=True)
        st.slider("s11", -3.0, 3.0, step=0.1, key='sl_s11',
                 help="Presión estática de salida del compresor de alta presión",
                 label_visibility="collapsed")

        st.markdown(f'<p style="font-size:13px;color:{color_s4};margin-bottom:-10px;font-weight:500">Temperatura salida LPT · s4{alerta_s4}</p>', unsafe_allow_html=True)
        st.slider("s4", -3.0, 3.0, step=0.1, key='sl_s4',
                 help="Temperatura de salida de la turbina de baja presión",
                 label_visibility="collapsed")

        st.slider("Ratio flujo combustible · s12", -3.0, 3.0, step=0.1, key='sl_s12',
                 help="Ratio de flujo de combustible respecto a Ps30")

        st.markdown("**⚙ Velocidad y bypass**")
        st.slider("Presión salida HPC · s7", -3.0, 3.0, step=0.1, key='sl_s7',
                 help="Presión de salida del compresor de alta presión")
        st.slider("Ratio de bypass · s15", -3.0, 3.0, step=0.1, key='sl_s15',
                 help="Ratio de bypass del motor")

        ciclo = st.number_input("Ciclo operacional actual", min_value=1, max_value=500, value=150, key='ciclo')

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
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=float(explainer.expected_value),
            data=datos_motor.iloc[0],
            feature_names=datos_motor.columns.tolist()
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        shap.plots.waterfall(explanation, show=False, max_display=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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
    col_r1.metric("AUC-ROC", "0.9964")
    col_r2.metric("Recall", "93.0%", help="% de fallos reales detectados — métrica crítica")
    col_r3.metric("Precision", "92.4%")
    col_r4.metric("F1-Score", "92.7%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="panel-title">Comparativa de modelos</div>', unsafe_allow_html=True)
        for nombre, tipo, auc, rec, prec, cls in [
            ("Regresión Logística", "Baseline", "0.9901", "95.0%", "72.6%", "warn"),
            ("XGBoost", "Sin optimizar", "0.9963", "92.7%", "92.1%", "good"),
            ("XGBoost + Optuna", "Modelo final ★", "0.9964", "93.0%", "92.4%", "blue"),
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
    col_v3.metric("Fallos evitados", "563 / 600", delta="Recall 93%")

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

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:10px;color:#1e2d4a;font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;">
AEROPREDICT · BOOTCAMP DATA SCIENCE · XGBOOST + OPTUNA + SHAP · NASA C-MAPSS
</div>
""", unsafe_allow_html=True)
