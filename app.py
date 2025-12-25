import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION "DARK MODE"
# ==========================================
st.set_page_config(
    page_title="MASI Dark Terminal",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injection CSS pour le Thème Sombre "Bloomberg"
st.markdown("""
<style>
    /* Fond principal (Noir Profond) */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Style des cartes (Gris Anthracite) */
    div[data-testid="stMetric"], .css-1r6slb0 {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3d3d3d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Titres en Blanc/Gris Clair */
    h1, h2, h3 {
        color: #e0e0e0 !important;
        font-family: 'Courier New', monospace; /* Police style Terminal */
    }
    
    /* Texte normal */
    p, label {
        color: #b0b0b0 !important;
    }
    
    /* Métriques (Chiffres) */
    div[data-testid="stMetricValue"] {
        color: #00e676; /* Vert Néon par défaut */
        font-family: 'Roboto Mono', monospace;
    }

    /* Slider (Barre de défilement) */
    .stSlider > div > div > div > div {
        background-color: #00e676;
    }

    /* Cacher menu Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Activation du style Matplotlib "Dark Background"
plt.style.use('dark_background')

# ==========================================
# 2. CHARGEMENT BACKEND
# ==========================================
@st.cache_data
def load_data():
    file_name = "Moroccan All Shares Historical Data.csv"
    import os
    if not os.path.exists(file_name):
        return None, None, None

    try:
        df = pd.read_csv(file_name, thousands=',', decimal='.')
        if len(df.columns) < 2:
            df = pd.read_csv(file_name, sep=';')
        df.columns = df.columns.str.strip()
        
        cols = df.columns.tolist()
        col_prix = next((c for c in cols if 'Price' in c or 'Dernier' in c), None)
        col_date = next((c for c in cols if 'Date' in c), None)

        if not col_prix or not col_date:
            return None, None, None

        df[col_date] = pd.to_datetime(df[col_date])
        df = df.sort_values(col_date).set_index(col_date)
        
        if df[col_prix].dtype == object:
            df[col_prix] = df[col_prix].str.replace(',', '').astype(float)

        df['Log_Return'] = np.log(df[col_prix] / df[col_prix].shift(1))
        clean_returns = df['Log_Return'].replace([np.inf, -np.inf], np.nan).dropna()
        
        return df, clean_returns, col_prix

    except Exception:
        return None, None, None

# ==========================================
# 3. INTERFACE TERMINAL
# ==========================================

# En-tête minimaliste
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown("# 🌑 MASI QUANT TERMINAL")
    st.caption("SYSTEME D'ANALYSE ALGORITHMIQUE // INSEA RABAT")
with col_head2:
    st.markdown("### 🟢 LIVE")

st.markdown("---")

df, clean_returns, col_prix = load_data()

if df is None:
    st.error("🚨 SYSTEM FAILURE : DATA FILE NOT FOUND")
    st.stop()

# --- A. TICKERS (KPIs) ---
last_price = df[col_prix].iloc[-1]
prev_price = df[col_prix].iloc[-2]
var_pct = ((last_price - prev_price) / prev_price) * 100
vol_annuelle = clean_returns.std() * np.sqrt(252) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("💎 DERNIER COURS", f"{last_price:,.2f}", f"{var_pct:.2f}%")
c2.metric("⚡ VOLATILITÉ", f"{vol_annuelle:.2f}%")
c3.metric("📊 SMA (20)", "BULLISH" if var_pct > 0 else "BEARISH", delta_color="normal")
c4.metric("💾 DATA POINTS", f"{len(df)}")

st.markdown("<br>", unsafe_allow_html=True)

# --- B. ZONE DE COMMANDE & GRAPHIQUE ---
col_ctrl, col_main = st.columns([1, 3])

with col_ctrl:
    st.markdown("### ⚙️ PARAMÈTRES")
    with st.container():
        st.write("HORIZON PRÉDICTIF")
        horizon = st.slider("", 1, 30, 5) # Slider sans label pour look épuré
        st.info(f"CIBLE : J+{horizon}")
        # MISE A JOUR DU TEXTE SIDEBAR
        st.markdown("MODELE : **ARIMA(2,0,2)**")

with col_main:
    st.markdown("### 🚀 PROJECTION TRAJECTOIRE")
    
    with st.spinner('COMPUTING NEURAL PATH...'):
        # --- MISE A JOUR DU MODELE ICI (2, 0, 2) ---
        model = ARIMA(clean_returns, order=(2,0,2))
        res = model.fit()
        forecast = res.get_forecast(steps=horizon)
        
        pred_ret = forecast.predicted_mean
        pred_price = last_price * np.exp(np.cumsum(pred_ret))
        dates_futur = pd.date_range(df.index[-1], periods=horizon+1, freq='B')[1:]
        
        cible = pred_price.iloc[-1]
        perf = ((cible - last_price)/last_price)*100
        
        # Graphique "Dark Mode"
        fig, ax = plt.subplots(figsize=(10, 4))
        # Le fond est géré par plt.style.use('dark_background')
        fig.patch.set_alpha(0) # Transparent pour fondre dans la page
        ax.patch.set_alpha(0)
        
        # Couleurs Néons
        ax.plot(df.index[-90:], df[col_prix].iloc[-90:], color='#00e676', linewidth=1.5, label='HISTORIQUE') # Vert Matrix
        ax.plot([df.index[-1], dates_futur[0]], [last_price, pred_price.iloc[0]], color='#ff0055', linestyle=':') # Pont
        ax.plot(dates_futur, pred_price, color='#ff0055', linewidth=2, marker='o', markersize=4, label='PRÉVISION') # Rose Cyberpunk
        
        # Cône d'incertitude
        std_err = np.linspace(0.005, 0.02, len(dates_futur)) * last_price
        ax.fill_between(dates_futur, pred_price-std_err, pred_price+std_err, color='#ff0055', alpha=0.2)
        
        # Esthétique
        ax.grid(True, linestyle=':', alpha=0.3, color='#444444')
        ax.legend(loc='upper left', frameon=False, labelcolor='linecolor')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#888888')
        ax.spines['left'].set_color('#888888')
        
        st.pyplot(fig)
        
        # Verdict Terminal Style
        if perf > 0:
            st.success(f"🟢 SIGNAL : LONG (ACHAT) | CIBLE : {cible:,.0f} (+{perf:.2f}%)")
        else:
            st.error(f"🔴 SIGNAL : SHORT (VENTE) | CIBLE : {cible:,.0f} ({perf:.2f}%)")

st.markdown("---")

# --- C. RADAR DE RISQUE (GARCH) ---
st.markdown("### 🛡️ ANALYSE DE STRESS (GARCH)")

try:
    garch = arch_model(clean_returns*100, p=1, q=1)
    res_g = garch.fit(disp='off')
    curr_vol = res_g.conditional_volatility.iloc[-1]
    
    c_g1, c_g2 = st.columns([1, 3])
    
    with c_g1:
        st.markdown("**INDICATEUR DE PEUR**")
        if curr_vol < 1.0:
            st.markdown("# 🟢 LOW")
        elif curr_vol < 1.8:
            st.markdown("# 🟡 MED")
        else:
            st.markdown("# 🔴 HIGH")
        st.caption(f"VIX LOCALE : {curr_vol:.2f}")

    with c_g2:
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        fig2.patch.set_alpha(0)
        ax2.patch.set_alpha(0)
        
        # Courbe orange "Feu"
        ax2.plot(res_g.conditional_volatility.iloc[-180:], color='#ff9100', linewidth=1)
        ax2.fill_between(res_g.conditional_volatility.index[-180:], 0, res_g.conditional_volatility.iloc[-180:], color='#ff9100', alpha=0.2)
        
        ax2.set_title("VOLATILITE IMPLICITE (6 MOIS)", fontsize=10, color='#888888')
        ax2.grid(False)
        ax2.axis('off') # On enlève les axes pour un look "Sparkline" pur
        
        st.pyplot(fig2)

except:
    st.warning("⚠️ GARCH MODULE ERROR")
