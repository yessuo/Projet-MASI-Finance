import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

#Configuration de la page (Titre, icône, mise en page)
st.set_page_config(page_title="MASI Prédictions", page_icon="📈", layout="wide")

# ==========================================
# 1. FONCTION DE CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data # Cette ligne permet de garder les données en mémoire (plus rapide)
def get_data():
    file_name = "Moroccan All Shares Historical Data.csv"
    try:
        # Chargement avec format US (Virgule pour milliers)
        df = pd.read_csv(file_name, thousands=',', decimal='.')
        
        # Nettoyage
        df.columns = df.columns.str.strip()
        col_prix = 'Price' if 'Price' in df.columns else 'Dernier'
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        
        # Rendements
        df['Log_Return'] = np.log(df[col_prix] / df[col_prix].shift(1))
        
        # Nettoyage final pour les modèles
        clean_returns = df['Log_Return'].replace([np.inf, -np.inf], np.nan).dropna()
        
        return df, clean_returns, col_prix
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None, None

# ==========================================
# 2. INTERFACE UTILISATEUR (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Configuration")
horizon = st.sidebar.slider("Horizon de prévision (Jours)", min_value=1, max_value=10, value=5)
st.sidebar.markdown("---")
st.sidebar.info("Application développée pour l'analyse du marché marocain (MASI).")

# ==========================================
# 3. CORPS PRINCIPAL
# ==========================================
st.title("📈 Tableau de Bord Prédictif : MASI")
st.markdown("Analyse automatique de tendance (ARIMA) et de risque (GARCH).")

# Chargement des données
df, clean_returns, col_prix = get_data()

if df is not None:
    # --- BLOC 1 : Indicateurs Clés ---
    col1, col2, col3 = st.columns(3)
    
    last_price = df[col_prix].iloc[-1]
    last_return = df['Log_Return'].iloc[-1] * 100
    volatility_hist = clean_returns.std() * np.sqrt(252) * 100
    
    col1.metric("Dernier Prix MASI", f"{last_price:,.2f} MAD", f"{last_return:.2f}%")
    col2.metric("Volatilité Historique (An)", f"{volatility_hist:.2f}%")
    col3.metric("Données analysées", f"{len(df)} Jours")

    st.markdown("---")

    # --- BLOC 2 : PRÉDICTION PRIX (ARIMA) ---
    st.subheader(f"🔮 Prévision de Tendance (Prochains {horizon} jours)")
    
    with st.spinner('Calcul du modèle ARIMA en cours...'):
        # Modèle ARIMA(2,0,2) comme validé précédemment
        model_arima = ARIMA(clean_returns, order=(2, 0, 2))
        fit_arima = model_arima.fit()
        forecast_res = fit_arima.get_forecast(steps=horizon)
        forecast_vals = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)
        
        # Verdict Haussier/Baissier
        tendance = forecast_vals.sum()
        if tendance > 0:
            st.success(f"✅ TENDANCE PRÉVUE : HAUSSIÈRE (+{tendance*100:.4f}%)")
        else:
            st.error(f"🔻 TENDANCE PRÉVUE : BAISSIÈRE ({tendance*100:.4f}%)")
            
        # Graphique ARIMA
        fig_arima, ax = plt.subplots(figsize=(10, 4))
        # Zoom sur les 30 derniers jours
        last_30 = df.iloc[-30:]
        ax.plot(last_30.index, last_30['Log_Return'], label='Historique Récent', color='grey', alpha=0.5)
        
        # Futur
        last_date = last_30.index[-1]
        future_dates = pd.date_range(start=last_date, periods=horizon+1)[1:]
        ax.plot(future_dates, forecast_vals, label='Prévision', color='red', marker='o')
        ax.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.legend()
        ax.set_title("Prédiction des Rendements Futurs")
        st.pyplot(fig_arima)

    st.markdown("---")

    # --- BLOC 3 : PRÉDICTION RISQUE (GARCH) ---
    st.subheader("🛡️ Analyse du Risque (Volatilité)")
    
    with st.spinner('Analyse de la peur (GARCH) en cours...'):
        # Modèle GARCH
        returns_scaled = clean_returns * 100
        garch = arch_model(returns_scaled, vol='Garch', p=1, q=1)
        res_garch = garch.fit(disp='off')
        
        # Prévision
        forecast_garch = res_garch.forecast(horizon=horizon)
        future_vol = np.sqrt(forecast_garch.variance.values[-1, :])
        
        # Verdict Calme/Nerveux
        avg_risk = res_garch.conditional_volatility.mean()
        curr_risk = future_vol[0]
        
        col_a, col_b = st.columns(2)
        
        col_a.metric("Risque Prévu (Demain)", f"{curr_risk:.2f}%")
        col_b.metric("Risque Moyen Historique", f"{avg_risk:.2f}%", delta_color="inverse", delta=f"{curr_risk - avg_risk:.2f}%")
        
        if curr_risk > avg_risk:
            st.warning("⚠️ ALERTE : Le marché est NERVEUX (Risque élevé).")
        else:
            st.success("✅ CALME : Le marché est STABLE (Risque faible).")

        # Graphique GARCH
        fig_garch, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(res_garch.conditional_volatility, color='orange', label='Volatilité (Risque)')
        ax2.set_title("Historique de la Volatilité (GARCH)")
        ax2.legend()
        st.pyplot(fig_garch)

else:
    st.warning("Veuillez vérifier que le fichier 'Moroccan All Shares Historical Data.csv' est bien dans le même dossier.")