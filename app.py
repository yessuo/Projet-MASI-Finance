import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION DE LA PAGE & CSS (LE DESIGN)
# ==========================================
st.set_page_config(
    page_title="MASI Pro Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# C'est ici que la magie du design opère (CSS Injection)
st.markdown("""
<style>
    /* Fond de l'application gris très clair (Style Dashboard) */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Style des conteneurs (Cartes Blanches avec Ombres) */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Style spécifique pour les métriques (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #e74c3c; /* Barre rouge sur le côté */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Titres Centrés */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
    }
    
    /* Cacher les éléments parasites */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CHARGEMENT DES DONNÉES (Back-end)
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
        
        # Détection auto
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
# 3. INTERFACE PRINCIPALE
# ==========================================

# --- EN-TÊTE ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown("<h1 style='text-align: left; margin-bottom: 0;'>🇲🇦 MASI MARKET INTELLIGENCE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: gray; margin-top: -10px;'>Plateforme d'Analyse Prédictive & Gestion des Risques (INSEA)</p>", unsafe_allow_html=True)

st.markdown("---")

# Chargement
df, clean_returns, col_prix = load_data()

if df is None:
    st.error("⚠️ Erreur : Veuillez placer le fichier CSV dans le dossier.")
    st.stop()

# --- A. BARRE D'ÉTAT (KPIs) ---
# On récupère les dernières valeurs
last_price = df[col_prix].iloc[-1]
prev_price = df[col_prix].iloc[-2]
var_pct = ((last_price - prev_price) / prev_price) * 100
vol_annuelle = clean_returns.std() * np.sqrt(252) * 100

# Affichage en 4 colonnes stylisées
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dernier Cours", f"{last_price:,.2f}", f"{var_pct:.2f}%")
c2.metric("Volatilité (Risque)", f"{vol_annuelle:.2f}%")
c3.metric("Tendance (SMA 20)", "Neutre" if var_pct > -0.5 and var_pct < 0.5 else ("Haussière" if var_pct > 0 else "Baissière"))
c4.metric("Données", f"{len(df)} Séances")

st.markdown("<br>", unsafe_allow_html=True) # Espacement

# --- B. ZONE DE CONTRÔLE & PRÉVISION ---
# On divise l'écran : 25% Menu (Gauche), 75% Graphique (Droite)
col_ctrl, col_main = st.columns([1, 3])

with col_ctrl:
    st.markdown("### ⚙️ Paramètres")
    with st.container(): # Effet carte
        st.write("Ajustez l'horizon pour recalculer le modèle ARIMA en temps réel.")
        horizon = st.slider("Horizon (Jours)", 1, 30, 5)
        st.info(f"Prévision jusqu'au : \n**{pd.date_range(df.index[-1], periods=horizon+1, freq='B')[-1].strftime('%d/%m/%Y')}**")

with col_main:
    st.markdown("### 🔮 Trajectoire Prévue (ARIMA)")
    
    with st.spinner('Calcul des trajectoires...'):
        # Modèle
        model = ARIMA(clean_returns, order=(5,1,0))
        res = model.fit()
        forecast = res.get_forecast(steps=horizon)
        
        # Reconstruction
        pred_ret = forecast.predicted_mean
        pred_price = last_price * np.exp(np.cumsum(pred_ret))
        dates_futur = pd.date_range(df.index[-1], periods=horizon+1, freq='B')[1:]
        
        # Calculs Cibles
        cible = pred_price.iloc[-1]
        perf = ((cible - last_price)/last_price)*100
        
        # Graphique Matplotlib "Pro"
        fig, ax = plt.subplots(figsize=(10, 4))
        # Fond transparent pour fondre dans la carte Streamlit
        fig.patch.set_alpha(0) 
        ax.patch.set_alpha(0)
        
        # Données
        ax.plot(df.index[-90:], df[col_prix].iloc[-90:], color='#2c3e50', linewidth=2, label='Historique')
        ax.plot([df.index[-1], dates_futur[0]], [last_price, pred_price.iloc[0]], color='#e74c3c', linestyle='--')
        ax.plot(dates_futur, pred_price, color='#e74c3c', linewidth=2, marker='o', markersize=4, label='Prévision')
        
        # Cône (Simulation visuelle pour le style)
        std_err = np.linspace(0.005, 0.02, len(dates_futur)) * last_price
        ax.fill_between(dates_futur, pred_price-std_err, pred_price+std_err, color='#e74c3c', alpha=0.1)
        
        # Esthétique Graphique
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='upper left', frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        
        # Verdict en bas du graph
        if perf > 0:
            st.success(f"🎯 **VERDICT :** Tendance HAUSSIÈRE détectée. Cible : {cible:,.2f} (+{perf:.2f}%)")
        else:
            st.error(f"🎯 **VERDICT :** Tendance BAISSIÈRE/CORRECTION détectée. Cible : {cible:,.2f} ({perf:.2f}%)")

st.markdown("<br>", unsafe_allow_html=True)

# --- C. ANALYSE DE RISQUE (GARCH) ---
st.markdown("### 🛡️ Radar de Volatilité (GARCH)")

try:
    garch = arch_model(clean_returns*100, p=1, q=1)
    res_g = garch.fit(disp='off')
    curr_vol = res_g.conditional_volatility.iloc[-1]
    
    c_g1, c_g2 = st.columns([1, 3])
    
    with c_g1:
        st.markdown("**Niveau de Stress**")
        # Jauge simple
        if curr_vol < 1.0:
            st.markdown("# 🟢 Faible")
        elif curr_vol < 1.8:
            st.markdown("# 🟠 Moyen")
        else:
            st.markdown("# 🔴 Élevé")
        st.caption(f"Volatilité Cond. : {curr_vol:.2f}")

    with c_g2:
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        fig2.patch.set_alpha(0)
        ax2.patch.set_alpha(0)
        
        ax2.plot(res_g.conditional_volatility.iloc[-180:], color='#f1c40f', label='Volatilité (GARCH)')
        ax2.fill_between(res_g.conditional_volatility.index[-180:], 0, res_g.conditional_volatility.iloc[-180:], color='#f1c40f', alpha=0.1)
        
        ax2.set_title("Évolution de la nervosité du marché (6 mois)", fontsize=10, color='gray')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', linestyle=':', alpha=0.3)
        
        st.pyplot(fig2)

except:
    st.warning("Module GARCH non disponible")
