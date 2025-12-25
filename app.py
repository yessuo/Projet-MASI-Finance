import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. CONFIGURATION & DESIGN (CSS)
# ==========================================
st.set_page_config(
    page_title="MASI Analytics Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection de CSS pour le style "Bloomberg / Pro"
st.markdown("""
<style>
    /* Fond général plus doux */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Titre principal centré et stylisé */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 20px;
    }
    /* Sous-titres */
    h3 {
        color: #34495e;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 10px;
    }
    /* Style des métriques (Gros chiffres) */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #2980b9;
    }
    /* Cacher le menu Streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CHARGEMENT ROBUSTE
# ==========================================
@st.cache_data
def load_data():
    file_name = "Moroccan All Shares Historical Data.csv"
    import os
    if not os.path.exists(file_name):
        return None, f"⚠️ Fichier '{file_name}' introuvable."

    try:
        # Lecture flexible
        df = pd.read_csv(file_name, thousands=',', decimal='.')
        if len(df.columns) < 2:
            df = pd.read_csv(file_name, sep=';')

        df.columns = df.columns.str.strip()
        
        # Détection intelligente des colonnes
        cols = df.columns.tolist()
        col_prix = next((c for c in cols if 'Price' in c or 'Dernier' in c or 'Close' in c), None)
        col_date = next((c for c in cols if 'Date' in c), None)

        if not col_prix or not col_date:
            return None, "⚠️ Colonnes Prix/Date non trouvées."

        df[col_date] = pd.to_datetime(df[col_date])
        df = df.sort_values(col_date).set_index(col_date)
        
        if df[col_prix].dtype == object:
            df[col_prix] = df[col_prix].str.replace(',', '').astype(float)

        df['Log_Return'] = np.log(df[col_prix] / df[col_prix].shift(1))
        clean_returns = df['Log_Return'].replace([np.inf, -np.inf], np.nan).dropna()
        
        return df, clean_returns, col_prix

    except Exception as e:
        return None, str(e)

# ==========================================
# 2. SIDEBAR (MENU LATÉRAL)
# ==========================================
st.sidebar.markdown("## ⚙️ Panneau de Contrôle")
st.sidebar.markdown("---")

horizon = st.sidebar.slider("📅 Horizon de prévision (Jours)", 1, 30, 5)

st.sidebar.markdown("### ℹ️ Détails du Modèle")
st.sidebar.info(
    """
    **Modèle Tendance :** ARIMA (5,1,0)
    **Modèle Risque :** GARCH (1,1)
    **Source :** Bourse de Casablanca
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Projet INSEA - Séries Temporelles")

# ==========================================
# 3. CORPS PRINCIPAL
# ==========================================

# --- TITRE PRINCIPAL ---
st.title("🇲🇦 MASI MARKET PREDICTOR")
st.markdown("<p style='text-align: center; color: grey;'>Intelligence Artificielle appliquée à l'Indice Boursier Marocain</p>", unsafe_allow_html=True)
st.markdown("---")

# Chargement
df, clean_returns, col_prix = load_data()

if df is None:
    st.error(clean_returns)
    st.stop()

# --- A. INDICATEURS CLÉS (KPIs) ---
last_price = df[col_prix].iloc[-1]
prev_price = df[col_prix].iloc[-2]
var_day = ((last_price - prev_price) / prev_price) * 100
vol_annuelle = clean_returns.std() * np.sqrt(252) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Dernier Cours", f"{last_price:,.2f}", f"{var_day:.2f}%")
col2.metric("⚡ Volatilité Hist.", f"{vol_annuelle:.2f}%")
col3.metric("📅 Données", f"{len(df)} Jours")
col4.metric("🔮 Horizon", f"{horizon} Jours")

st.markdown("---")

# --- B. PRÉVISION (ZONE PRINCIPALE) ---
st.markdown("### 🚀 Prévision de la Trajectoire")

try:
    with st.spinner('Le modèle analyse les tendances...'):
        # Calculs ARIMA
        model_arima = ARIMA(clean_returns, order=(5, 1, 0))
        fit_arima = model_arima.fit()
        forecast = fit_arima.get_forecast(steps=horizon)
        
        # Reconstruction Prix
        pred_returns = forecast.predicted_mean
        pred_prices = last_price * np.exp(np.cumsum(pred_returns))
        dates_futur = pd.date_range(df.index[-1], periods=horizon+1, freq='B')[1:]
        
        # Calcul Cible
        prix_cible = pred_prices.iloc[-1]
        variation_prevue = ((prix_cible - last_price) / last_price) * 100
        
        # Layout Graphique + Analyse
        c_graph, c_info = st.columns([3, 1])
        
        with c_graph:
            fig, ax = plt.subplots(figsize=(10, 5))
            # Style du graphique
            ax.set_facecolor('#f8f9fa')
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # Courbes
            ax.plot(df.index[-90:], df[col_prix].iloc[-90:], label="Historique Réel", color='#34495e', linewidth=2)
            ax.plot([df.index[-1], dates_futur[0]], [last_price, pred_prices.iloc[0]], color='#e74c3c', linestyle='--') # Pont
            ax.plot(dates_futur, pred_prices, color='#e74c3c', label=f"Prévision (J+{horizon})", linestyle='--', marker='o')
            
            # Cône de confiance (Simulation visuelle)
            conf_scale = np.linspace(0.005, 0.015, len(dates_futur)) * last_price # Élargissement progressif
            ax.fill_between(dates_futur, pred_prices - conf_scale, pred_prices + conf_scale, color='#e74c3c', alpha=0.1)
            
            ax.legend()
            ax.set_title("Projection Dynamique des Prix", fontsize=12)
            st.pyplot(fig)
            
        with c_info:
            st.markdown("#### 🎯 Objectif")
            st.metric("Prix Cible", f"{prix_cible:,.0f}", f"{variation_prevue:.2f}%")
            
            if variation_prevue > 0:
                st.success("Tendance : **HAUSSIÈRE**")
                st.markdown("Le modèle détecte une dynamique positive à court terme.")
            else:
                st.error("Tendance : **BAISSIÈRE**")
                st.markdown("Le modèle suggère une correction ou une prise de bénéfices.")

except Exception as e:
    st.error(f"Erreur Modèle : {e}")

st.markdown("---")

# --- C. ANALYSE DE RISQUE (GARCH) ---
st.markdown("### 🛡️ Analyse de la Volatilité (Risque)")

try:
    garch = arch_model(clean_returns * 100, p=1, q=1)
    res = garch.fit(disp='off')
    
    # Récupérer la dernière volatilité estimée
    curr_vol = res.conditional_volatility.iloc[-1]
    
    col_g1, col_g2 = st.columns([1, 3])
    
    with col_g1:
        st.markdown("<br>", unsafe_allow_html=True) # Espacement
        st.metric("Nervosité Marché", f"{curr_vol:.2f}%")
        
        if curr_vol > 1.5:
            st.warning("⚠️ Marché Agité")
        else:
            st.success("✅ Marché Calme")
            
    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.set_facecolor('#f8f9fa')
        ax2.plot(res.conditional_volatility.iloc[-120:], color='#f39c12', label='Volatilité Conditionnelle')
        ax2.set_title("Évolution du Risque sur 6 mois")
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig2)

except Exception as e:
    st.warning("Module GARCH non disponible.")
