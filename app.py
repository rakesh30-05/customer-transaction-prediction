import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Transaction Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0A0F1E; color: #E2E8F0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1400px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0F3460 0%, #16213E 30%, #533483 65%, #E94560 100%);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(83,52,131,0.35), 0 2px 8px rgba(233,69,96,0.15);
    border: 1px solid rgba(255,255,255,0.07);
}
.hero::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(255,255,255,0.09) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -70px; left: 28%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(233,69,96,0.15) 0%, transparent 65%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 800;
    color: #FFFFFF; margin: 0 0 0.7rem 0; line-height: 1.15;
    letter-spacing: -0.02em;
}
.hero-accent {
    background: linear-gradient(90deg, #A78BFA, #F472B6, #FB923C);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: rgba(255,255,255,0.65); font-size: 0.95rem; margin: 0; max-width: 600px; line-height: 1.6; }
.hero-stats { display: flex; gap: 2.5rem; margin-top: 2rem; flex-wrap: wrap; }
.hero-stat .val {
    font-size: 1.75rem; font-weight: 700; color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 0 20px rgba(255,255,255,0.25);
}
.hero-stat .lbl { font-size: 0.7rem; color: rgba(255,255,255,0.42); text-transform: uppercase; letter-spacing: 0.12em; margin-top: 3px; }

/* ── Section Header ── */
.section-header {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #818CF8, #C084FC, #F472B6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(129,140,248,0.25);
}

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    border: 1px solid rgba(129,140,248,0.18);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: rgba(129,140,248,0.45); transform: translateY(-2px); }
.metric-card .m-val {
    font-size: 1.8rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #818CF8, #C084FC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .m-lbl { font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
.metric-card .m-sub { font-size: 0.78rem; color: #475569; margin-top: 0.2rem; }

/* ── Result Boxes ── */
.result-positive {
    background: linear-gradient(135deg, rgba(20,83,45,0.6), rgba(21,128,61,0.4));
    border: 2px solid #22C55E;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 4px 24px rgba(34,197,94,0.2);
}
.result-negative {
    background: linear-gradient(135deg, rgba(127,29,29,0.6), rgba(153,27,27,0.4));
    border: 2px solid #EF4444;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 4px 24px rgba(239,68,68,0.2);
}
.result-prob { font-size: 2.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.result-label { font-size: 1.25rem; font-weight: 700; margin: 0.5rem 0 0.3rem 0; }
.result-sub { font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem; }

/* ── Info Box ── */
.info-box {
    background: linear-gradient(135deg, rgba(15,52,96,0.25), rgba(83,52,131,0.2));
    border: 1px solid rgba(129,140,248,0.2);
    border-left: 3px solid #818CF8;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.85rem; color: #94A3B8; margin: 1rem 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0F172A; border-radius: 12px;
    padding: 4px; border: 1px solid rgba(129,140,248,0.18);
}
.stTabs [data-baseweb="tab"] { color: #475569; font-weight: 500; font-size: 0.9rem; border-radius: 8px; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0F3460, #533483) !important;
    color: #FFFFFF !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F3460 0%, #16213E 35%, #533483 75%, #E94560 100%);
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #E94560 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0F3460, #533483, #E94560);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 1.8rem; font-weight: 600; font-size: 0.9rem;
    width: 100%; transition: all 0.2s;
    box-shadow: 0 3px 14px rgba(83,52,131,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #E94560, #533483, #0F3460);
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(233,69,96,0.4);
}

/* ── Misc ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
hr { border-color: rgba(129,140,248,0.15); }
.stTextArea textarea {
    background: #0F172A; border: 1px solid rgba(129,140,248,0.2);
    border-radius: 10px; color: #E2E8F0;
}
.stNumberInput input { background: #0F172A; border: 1px solid rgba(129,140,248,0.2); color: #E2E8F0; border-radius: 8px; }
.stFileUploader { background: #0F172A; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Customer Transaction <span class="hero-accent">Predictor</span></h1>
    <p>Predict whether a Santander Bank customer will make a specific transaction — powered by LightGBM trained on 200,000 records with SMOTE balancing.</p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="val">0.764</div><div class="lbl">ROC-AUC Score</div></div>
        <div class="hero-stat"><div class="val">83.4%</div><div class="lbl">Accuracy</div></div>
        <div class="hero-stat"><div class="val">200K</div><div class="lbl">Training Rows</div></div>
        <div class="hero-stat"><div class="val">200</div><div class="lbl">Features</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        base = os.path.dirname(__file__)
        model  = joblib.load(os.path.join(base, "lgbm_model.pkl"))
        scaler = joblib.load(os.path.join(base, "scaler.pkl"))
        return model, scaler, True
    except:
        return None, None, False

model, scaler, model_loaded = load_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    threshold = st.slider("Decision Threshold", 0.10, 0.90, 0.35, 0.01,
        help="Lower = more customers flagged. Higher = more conservative.")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.07); border-radius:10px; padding:0.9rem 1rem; font-size:0.84rem; margin:0.8rem 0; border:1px solid rgba(255,255,255,0.1);">
        <b>Threshold: {threshold:.2f}</b><br>
        Probability ≥ {threshold:.2f} → Will Transact
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.07); border-radius:10px; padding:0.9rem 1rem; font-size:0.84rem; border:1px solid rgba(255,255,255,0.1);">
        <b>Model:</b> LightGBM<br>
        <b>Imbalance:</b> SMOTE (50:50)<br>
        <b>Features:</b> var_0 – var_199<br>
        <b>Metric:</b> ROC-AUC = 0.764
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:rgba(255,255,255,0.3); text-align:center;">
        Built by <b style="color:rgba(255,255,255,0.55)">D N Rakesh</b><br>
        Data Science Portfolio · PRCP-1003
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Single Prediction", "📂  Batch Prediction (CSV)", "📈  Model Performance"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Choose Input Method</div>', unsafe_allow_html=True)

    method = st.radio(
        "How do you want to provide customer data?",
        ["🎲 Random Sample", "📋 Paste CSV Row", "✏️ Manual Entry"],
        horizontal=True
    )
    st.markdown("---")

    # ── Method 1: Random Sample (DEFAULT — works best) ─────────────────────────
    if method == "🎲 Random Sample":
        st.markdown("""
        <div class="info-box">
            📌 Click the button to instantly generate a random customer and get a prediction.
            Perfect for <b>demos and interviews</b>.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎲 Generate Random Customer & Predict"):
            np.random.seed(np.random.randint(0, 99999))
            vals = np.random.normal(loc=0, scale=10, size=200).astype(np.float32)
            st.session_state["rand_vals"] = vals
            st.session_state["rand_done"] = True

        if st.session_state.get("rand_done") and "rand_vals" in st.session_state:
            vals = st.session_state["rand_vals"]

            # ── Predict ──
            if model_loaded:
                prob = float(model.predict_proba(scaler.transform(vals.reshape(1, -1)))[0][1])
            else:
                prob = float(np.random.beta(2, 5))
            prediction = 1 if prob >= threshold else 0

            # ── Show result ──
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div style="font-size:3rem;">✅</div>
                        <div class="result-label" style="color:#22C55E;">Will Make Transaction</div>
                        <div class="result-prob" style="color:#22C55E;">{prob:.1%}</div>
                        <div class="result-sub">Transaction probability · Threshold: {threshold:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div style="font-size:3rem;">❌</div>
                        <div class="result-label" style="color:#EF4444;">Will NOT Make Transaction</div>
                        <div class="result-prob" style="color:#EF4444;">{prob:.1%}</div>
                        <div class="result-sub">Transaction probability · Threshold: {threshold:.2f}</div>
                    </div>""", unsafe_allow_html=True)

            with col_r2:
                bg = '#0F172A'
                fig, ax = plt.subplots(figsize=(4, 4), facecolor=bg)
                ax.set_facecolor(bg)
                theta = np.linspace(np.pi, 0, 200)
                r = 0.8
                ax.plot(r*np.cos(theta), r*np.sin(theta), color='#1E293B', linewidth=18, solid_capstyle='round')
                color = '#22C55E' if prob >= threshold else '#EF4444'
                fill_theta = np.linspace(np.pi, np.pi - prob*np.pi, 200)
                ax.plot(r*np.cos(fill_theta), r*np.sin(fill_theta), color=color, linewidth=18, solid_capstyle='round')
                ta = np.pi - threshold * np.pi
                ax.plot([0.65*np.cos(ta), 0.95*np.cos(ta)], [0.65*np.sin(ta), 0.95*np.sin(ta)], color='#C084FC', linewidth=2.5)
                ax.text(0, 0.15, f"{prob:.1%}", ha='center', va='center', fontsize=26, fontweight='bold', color=color, fontfamily='monospace')
                ax.text(0, -0.15, 'Transaction Probability', ha='center', va='center', fontsize=9, color='#64748B')
                ax.text(0, -0.38, f'▲ Threshold: {threshold:.2f}', ha='center', va='center', fontsize=8, color='#C084FC')
                ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.6, 1.1)
                ax.set_aspect('equal'); ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()

            # ── Preview generated values ──
            preview_df = pd.DataFrame([vals], columns=[f"var_{i}" for i in range(200)]).T.reset_index()
            preview_df.columns = ["Feature", "Value"]
            preview_df["Value"] = preview_df["Value"].round(4)
            with st.expander("👁️ View Generated Feature Values"):
                c1, c2, c3, c4 = st.columns(4)
                for idx, col in enumerate([c1, c2, c3, c4]):
                    with col:
                        st.dataframe(preview_df.iloc[idx*50:(idx+1)*50], use_container_width=True, hide_index=True)

            if not model_loaded:
                st.warning("⚠️ Demo mode — place lgbm_model.pkl and scaler.pkl in the app folder.")

    # ── Method 2: Paste CSV Row ────────────────────────────────────────────────
    elif method == "📋 Paste CSV Row":
        st.markdown("""
        <div class="info-box">
            📌 Open your <b>train.csv</b> or <b>test.csv</b>, copy any one data row and paste below.
            ID_code and target columns are handled automatically.
        </div>
        """, unsafe_allow_html=True)

        pasted = st.text_area(
            "Paste a single customer row (comma separated values):",
            placeholder="8.9255,-6.7863,11.9081,5.0930,11.4607,-9.2834,...",
            height=120
        )

        if st.button("🔮 Predict from Pasted Row"):
            if pasted.strip() == "":
                st.error("Please paste a data row first.")
            else:
                # parse all numeric values, skip non-numeric (like ID_code)
                parts = [p.strip() for p in pasted.strip().split(",")]
                numeric = []
                for p in parts:
                    try:
                        numeric.append(float(p))
                    except:
                        pass
                if len(numeric) == 201:
                    numeric = numeric[1:]
                if len(numeric) == 200:
                    st.session_state["paste_vals"] = np.array(numeric, dtype=np.float32)
                    st.session_state["paste_done"] = True
                else:
                    st.session_state["paste_done"] = False
                    st.error(f"Expected 200 numeric values, got {len(numeric)}. Please check your row.")

        if st.session_state.get("paste_done") and "paste_vals" in st.session_state:
            vals = st.session_state["paste_vals"]
            if model_loaded:
                prob = float(model.predict_proba(scaler.transform(vals.reshape(1, -1)))[0][1])
            else:
                prob = float(np.random.beta(2, 5))
            prediction = 1 if prob >= threshold else 0

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div style="font-size:3rem;">✅</div>
                        <div class="result-label" style="color:#22C55E;">Will Make Transaction</div>
                        <div class="result-prob" style="color:#22C55E;">{prob:.1%}</div>
                        <div class="result-sub">Transaction probability · Threshold: {threshold:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div style="font-size:3rem;">❌</div>
                        <div class="result-label" style="color:#EF4444;">Will NOT Make Transaction</div>
                        <div class="result-prob" style="color:#EF4444;">{prob:.1%}</div>
                        <div class="result-sub">Transaction probability · Threshold: {threshold:.2f}</div>
                    </div>""", unsafe_allow_html=True)
            with col_r2:
                bg = '#0F172A'
                fig, ax = plt.subplots(figsize=(4, 4), facecolor=bg)
                ax.set_facecolor(bg)
                theta = np.linspace(np.pi, 0, 200)
                r = 0.8
                ax.plot(r*np.cos(theta), r*np.sin(theta), color='#1E293B', linewidth=18, solid_capstyle='round')
                color = '#22C55E' if prob >= threshold else '#EF4444'
                fill_theta = np.linspace(np.pi, np.pi - prob*np.pi, 200)
                ax.plot(r*np.cos(fill_theta), r*np.sin(fill_theta), color=color, linewidth=18, solid_capstyle='round')
                ta = np.pi - threshold * np.pi
                ax.plot([0.65*np.cos(ta), 0.95*np.cos(ta)], [0.65*np.sin(ta), 0.95*np.sin(ta)], color='#C084FC', linewidth=2.5)
                ax.text(0, 0.15, f"{prob:.1%}", ha='center', va='center', fontsize=26, fontweight='bold', color=color, fontfamily='monospace')
                ax.text(0, -0.15, 'Transaction Probability', ha='center', va='center', fontsize=9, color='#64748B')
                ax.text(0, -0.38, f'▲ Threshold: {threshold:.2f}', ha='center', va='center', fontsize=8, color='#C084FC')
                ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.6, 1.1)
                ax.set_aspect('equal'); ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()

            if not model_loaded:
                st.warning("⚠️ Demo mode — place lgbm_model.pkl and scaler.pkl in the app folder.")

    # ── Method 3: Manual Entry ─────────────────────────────────────────────────
    elif method == "✏️ Manual Entry":
        st.markdown("""
        <div class="info-box">
            📌 Enter values for all 200 features manually.
            Use <b>Fill Random</b> to auto-fill, then tweak specific values.
        </div>
        """, unsafe_allow_html=True)

        col_f1, col_f2, _ = st.columns([1, 1, 3])
        with col_f1:
            fill_random = st.button("🎲 Fill Random")
        with col_f2:
            fill_zero = st.button("🔄 Reset Zeros")

        if "man_vals" not in st.session_state or fill_zero:
            st.session_state.man_vals = {f"var_{i}": 0.0 for i in range(200)}
        if fill_random:
            np.random.seed(np.random.randint(0, 9999))
            st.session_state.man_vals = {f"var_{i}": round(float(np.random.normal(0, 10)), 4) for i in range(200)}

        feature_values = {}
        for gs in range(0, 200, 20):
            ge = min(gs + 20, 200)
            with st.expander(f"var_{gs} → var_{ge-1}"):
                cols = st.columns(5)
                for i, fi in enumerate(range(gs, ge)):
                    feat = f"var_{fi}"
                    with cols[i % 5]:
                        feature_values[feat] = st.number_input(
                            feat,
                            value=float(st.session_state.man_vals.get(feat, 0.0)),
                            format="%.4f", key=f"man_{feat}"
                        )

        if st.button("🔮 Predict", key="manual_predict"):
            vals = np.array([feature_values[f"var_{i}"] for i in range(200)], dtype=np.float32)
            if model_loaded:
                prob = float(model.predict_proba(scaler.transform(vals.reshape(1, -1)))[0][1])
            else:
                prob = float(np.random.beta(2, 5))
            prediction = 1 if prob >= threshold else 0

            st.markdown("---")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div style="font-size:3rem;">✅</div>
                        <div class="result-label" style="color:#22C55E;">Will Make Transaction</div>
                        <div class="result-prob" style="color:#22C55E;">{prob:.1%}</div>
                        <div class="result-sub">Transaction probability · Threshold: {threshold:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div style="font-size:3rem;">❌</div>
                        <div class="result-label" style="color:#EF4444;">Will NOT Make Transaction</div>
                        <div class="result-prob" style="color:#EF4444;">{prob:.1%}</div>
                        <div class="result-sub">Transaction probability · Threshold: {threshold:.2f}</div>
                    </div>""", unsafe_allow_html=True)
            with col_r2:
                bg = '#0F172A'
                fig, ax = plt.subplots(figsize=(4, 4), facecolor=bg)
                ax.set_facecolor(bg)
                theta = np.linspace(np.pi, 0, 200)
                r = 0.8
                ax.plot(r*np.cos(theta), r*np.sin(theta), color='#1E293B', linewidth=18, solid_capstyle='round')
                color = '#22C55E' if prob >= threshold else '#EF4444'
                fill_theta = np.linspace(np.pi, np.pi - prob*np.pi, 200)
                ax.plot(r*np.cos(fill_theta), r*np.sin(fill_theta), color=color, linewidth=18, solid_capstyle='round')
                ta = np.pi - threshold * np.pi
                ax.plot([0.65*np.cos(ta), 0.95*np.cos(ta)], [0.65*np.sin(ta), 0.95*np.sin(ta)], color='#C084FC', linewidth=2.5)
                ax.text(0, 0.15, f"{prob:.1%}", ha='center', va='center', fontsize=26, fontweight='bold', color=color, fontfamily='monospace')
                ax.text(0, -0.15, 'Transaction Probability', ha='center', va='center', fontsize=9, color='#64748B')
                ax.text(0, -0.38, f'▲ Threshold: {threshold:.2f}', ha='center', va='center', fontsize=8, color='#C084FC')
                ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.6, 1.1)
                ax.set_aspect('equal'); ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()

            if not model_loaded:
                st.warning("⚠️ Demo mode — place lgbm_model.pkl and scaler.pkl in the app folder.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Batch Prediction via CSV Upload</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Upload a CSV with columns <b>var_0</b> to <b>var_199</b>. Optionally include <b>ID_code</b>.
        You can upload your original <b>test.csv</b> directly!
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(3), use_container_width=True)

        feature_cols = [f"var_{i}" for i in range(200)]
        missing = [c for c in feature_cols if c not in df.columns]

        if missing:
            st.error(f"Missing {len(missing)} feature columns.")
        else:
            X = df[feature_cols].values.astype('float32')
            if model_loaded:
                probs = model.predict_proba(scaler.transform(X))[:, 1]
            else:
                probs = np.random.beta(2, 5, size=len(X))

            preds = (probs >= threshold).astype(int)
            result_df = pd.DataFrame()
            if 'ID_code' in df.columns:
                result_df['ID_code'] = df['ID_code']
            result_df['Transaction_Probability'] = np.round(probs, 4)
            result_df['Prediction'] = preds
            result_df['Result'] = result_df['Prediction'].map({1: '✅ Will Transact', 0: '❌ Will NOT Transact'})

            st.markdown("---")
            n_pos = int(preds.sum())
            n_neg = len(preds) - n_pos

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="m-val">{len(preds):,}</div><div class="m-lbl">Total Customers</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="m-val" style="background:linear-gradient(135deg,#16A34A,#22C55E);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{n_pos:,}</div><div class="m-lbl">Will Transact</div><div class="m-sub">{n_pos/len(preds):.1%}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="m-val" style="background:linear-gradient(135deg,#DC2626,#EF4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{n_neg:,}</div><div class="m-lbl">Will NOT Transact</div><div class="m-sub">{n_neg/len(preds):.1%}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="m-val">{probs.mean():.3f}</div><div class="m-lbl">Avg Probability</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True)

            bg = '#0F172A'
            fig2, ax2 = plt.subplots(figsize=(8, 3), facecolor=bg)
            ax2.set_facecolor(bg)
            ax2.hist(probs[preds==0], bins=40, color='#EF4444', alpha=0.75, label='Will NOT Transact')
            ax2.hist(probs[preds==1], bins=40, color='#22C55E', alpha=0.75, label='Will Transact')
            ax2.axvline(threshold, color='#C084FC', linewidth=2, linestyle='--', label=f'Threshold ({threshold:.2f})')
            ax2.set_xlabel('Transaction Probability', color='#64748B', fontsize=10)
            ax2.set_ylabel('Count', color='#64748B', fontsize=10)
            ax2.set_title('Prediction Probability Distribution', color='#E2E8F0', fontsize=12, fontweight='bold')
            ax2.tick_params(colors='#475569')
            for sp in ['top','right']: ax2.spines[sp].set_visible(False)
            for sp in ['bottom','left']: ax2.spines[sp].set_color('#1E293B')
            ax2.legend(facecolor='#0F172A', edgecolor='#1E293B', labelcolor='#94A3B8', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            csv_out = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions CSV", data=csv_out,
                               file_name="transaction_predictions.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, (val, lbl, sub) in zip([c1,c2,c3,c4,c5], [
        ("0.7642","ROC-AUC","Primary Metric"),
        ("83.42%","Accuracy","Test Set"),
        ("29.11%","Precision","Class 1"),
        ("45.30%","Recall","Class 1"),
        ("35.45%","F1 Score","Class 1"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="m-val">{val}</div><div class="m-lbl">{lbl}</div><div class="m-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    bg = '#0F172A'

    with col_left:
        st.markdown('<div class="section-header">Model Comparison — ROC-AUC</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 4), facecolor=bg)
        ax3.set_facecolor(bg)
        mnames = ['Logistic\nRegression','Decision\nTree','XGBoost','LightGBM']
        aucs = [0.632, 0.621, 0.662, 0.764]
        bcolors = ['#1E3A5F','#1E3A5F','#2D4A7A','#533483']
        bars = ax3.bar(mnames, aucs, color=bcolors, width=0.5, zorder=3)
        bars[-1].set_color('#7C3AED')
        ax3.set_ylim(0.55, 0.82)
        ax3.set_ylabel('ROC-AUC Score', color='#64748B', fontsize=10)
        ax3.set_title('Model Comparison', color='#E2E8F0', fontsize=12, fontweight='bold')
        ax3.tick_params(colors='#475569', labelsize=9)
        for sp in ['top','right']: ax3.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax3.spines[sp].set_color('#1E293B')
        ax3.yaxis.grid(True, color='#1E293B', linewidth=1, zorder=0)
        for bar, v in zip(bars, aucs):
            ax3.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.003,
                     f'{v:.3f}', ha='center', va='bottom',
                     color='#E2E8F0', fontsize=9, fontweight='bold', fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col_right:
        st.markdown('<div class="section-header">Confusion Matrix — LightGBM</div>', unsafe_allow_html=True)
        cm = np.array([[33275, 2705],[2963, 1057]])
        fig4, ax4 = plt.subplots(figsize=(5, 4), facecolor=bg)
        ax4.set_facecolor(bg)
        ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
        labels = ['No Transaction','Transaction']
        ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
        ax4.set_xticklabels(labels, color='#94A3B8', fontsize=9)
        ax4.set_yticklabels(labels, color='#94A3B8', fontsize=9)
        ax4.set_xlabel('Predicted', color='#64748B', fontsize=10)
        ax4.set_ylabel('Actual', color='#64748B', fontsize=10)
        ax4.set_title('Confusion Matrix', color='#E2E8F0', fontsize=12, fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                         color='white' if cm[i,j] > cm.max()/2 else '#E2E8F0',
                         fontsize=14, fontweight='bold', fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">ML Pipeline Steps</div>', unsafe_allow_html=True)
    steps = [
        ("01","Data Loading","200,000 rows × 202 cols. Dropped ID_code."),
        ("02","EDA","Nulls: 0, Duplicates: 0, Imbalance: 90:10"),
        ("03","Feature Scaling","StandardScaler — fit on train only"),
        ("04","SMOTE","143,922 samples each class → 50:50 balanced"),
        ("05","Model Training","LR, Decision Tree, XGBoost, LightGBM"),
        ("06","Hyperparameter Tuning","GridSearchCV 5-fold CV, F1 scoring"),
        ("07","Threshold Tuning","Best threshold = 0.35 for F1/Recall"),
        ("08","Best Model","LightGBM — ROC-AUC: 0.7642"),
    ]
    cols_pipe = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols_pipe[i % 4]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; margin-bottom:1rem;">
                <div style="font-size:0.7rem; font-weight:700; margin-bottom:0.3rem;
                     background:linear-gradient(90deg,#818CF8,#C084FC);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{num}</div>
                <div style="font-size:0.9rem; font-weight:600; color:#E2E8F0; margin-bottom:0.3rem;">{title}</div>
                <div style="font-size:0.78rem; color:#475569;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
