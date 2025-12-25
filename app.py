import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. CONFIGURATION
# ==========================================
st.set_page_config(page_title="MASI Prédiction", layout="wide")

# ==========================================
# 1. CHARGEMENT ROBUSTE
# ==========================================
@st.cache_data
def load_data():
    file_name = "Moroccan All Shares Historical Data.csv"
    
    # 1. Vérification basique : Est-ce que le fichier existe ?
    import os
    if not os.path.exists(file_name):
        return None, f"⚠️ Fichier '{file_name}' introuvable. Vérifiez qu'il est bien à côté de app.py."

    try:
        # 2. Tentative de lecture flexible (séparateur automatique)
        # On essaie d'abord avec la virgule pour les milliers
        df = pd.read_csv(file_name, thousands=',', decimal='.')
        
        # Si ça a mal lu (tout dans une colonne), on réessaie avec point-virgule
        if len(df.columns) < 2:
            df = pd.read_csv(file_name, sep=';')

        # Nettoyage des colonnes (enlève les espaces)
        df.columns = df.columns.str.strip()
        
        # 3. Trouver la colonne PRIX et DATE
        cols = df.columns.tolist()
        col_prix = next((c for c in cols if 'Price' in c or 'Dernier' in c or 'Close' in c), None)
        col_date = next((c for c in cols if 'Date' in c), None)

        if not col_prix or not col_date:
            return None, f"⚠️ Colonnes non trouvées. Colonnes détectées : {cols}"

        # 4. Conversion
        df[col_date] = pd.to_datetime(df[col_date])
        df = df.sort_values(col_date).set_index(col_date)
        
        # Conversion du prix en numérique (si c'est encore du texte '12,000')
        if df[col_prix].dtype == object:
            df[col_prix] = df[col_prix].str.replace(',', '').astype(float)

        # Calcul Log Return
        df['Log_Return'] = np.log(df[col_prix] / df[col_prix].shift(1))
        clean_returns = df['Log_Return'].replace([np.inf, -np.inf], np.nan).dropna()
        
        return df, clean_returns, col_prix

    except Exception as e:
        return None, f"⚠️ Erreur de lecture : {str(e)}"

# ==========================================
# 2. INTERFACE
# ==========================================
st.title("📈 Prédiction MASI (Mode Correction)")

df, clean_returns, col_prix = load_data()

# Si le chargement a échoué, on affiche l'erreur
if df is None:
    st.error(clean_returns) # Ici clean_returns contient le message d'erreur
    st.stop() # On arrête tout

# Si on arrive ici, c'est que les données sont chargées !
horizon = st.slider("Horizon (Jours)", 1, 30, 5)

# --- A. PRÉVISION PRIX (ARIMA) ---
try:
    with st.spinner('Calcul ARIMA...'):
        model_arima = ARIMA(clean_returns, order=(5, 1, 0))
        fit_arima = model_arima.fit()
        forecast = fit_arima.get_forecast(steps=horizon)
        
        # Reconstruction Prix
        last_price = df[col_prix].iloc[-1]
        pred_returns = forecast.predicted_mean
        pred_prices = last_price * np.exp(np.cumsum(pred_returns))
        
        # Dates
        dates = pd.date_range(df.index[-1], periods=horizon+1, freq='B')[1:]
        
        # Affichage
        st.subheader("Trajectoire Prévue")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index[-60:], df[col_prix].iloc[-60:], label="Historique")
            ax.plot(dates, pred_prices, color='red', label="Prévision", linestyle='--')
            ax.legend()
            st.pyplot(fig)
            
        with col2:
            cible = pred_prices.iloc[-1]
            st.metric("Cible", f"{cible:,.0f}")
            if cible > last_price:
                st.success("Hausse prévue")
            else:
                st.error("Baisse prévue")

except Exception as e:
    st.error(f"Erreur dans le calcul ARIMA : {e}")

# --- B. RISQUE (GARCH) ---
try:
    st.subheader("Analyse de Volatilité")
    # On multiplie par 100 pour stabiliser GARCH
    garch = arch_model(clean_returns * 100, p=1, q=1)
    res = garch.fit(disp='off')
    st.success("✅ Modèle GARCH calibré avec succès")
    
    # Graphique Volatilité
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(res.conditional_volatility.iloc[-100:], color='orange', label='Volatilité')
    st.pyplot(fig2)

except Exception as e:
    st.warning(f"Le module GARCH n'a pas pu tourner (souvent un problème de version ou de données). Erreur : {e}")
