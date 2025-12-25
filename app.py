import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# ==========================================
# 0. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="MASI Prédictions Pro", 
    page_icon="📈", 
    layout="wide"
)

# Style CSS personnalisé pour cacher les menus par défaut et épurer
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data
def get_data():
    file_name = "Moroccan All Shares Historical Data.csv"
    try:
        # Chargement avec gestion des milliers (12,000.00)
        df = pd.read_csv(file_name, thousands=',', decimal='.')
        
        # Nettoyage des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Identification de la colonne prix
        col_prix = 'Price' if 'Price' in df.columns else 'Dernier'
        
        # Gestion des dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        
        # Calcul des Rendements Logarithmiques (Indispensable pour les modèles)
        df['Log_Return'] = np.log(df[col_prix] / df[col_prix].shift(1))
        
        # Création d'une série propre sans NaN pour l'entraînement
        clean_returns = df['Log_Return'].replace([np.inf, -np.inf], np.nan).dropna()
        
        return df, clean_returns, col_prix
        
    except Exception as e:
        st.error(f"⚠️ Erreur critique : Impossible de charger le fichier '{file_name}'. \nDétail: {e}")
        return None, None, None

# ==========================================
# 2. INTERFACE (SIDEBAR)
# ==========================================
st.sidebar.header("🎛️ Paramètres de Simulation")
horizon = st.sidebar.slider("Horizon de prévision (Jours)", min_value=1, max_value=30, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.info(
    """
    **Modèle Tendance :** ARIMA (5,1,0)
    **Modèle Risque :** GARCH (1,1)
    
    Ce tableau de bord aide à anticiper la trajectoire
    du MASI et à surveiller la volatilité.
    """
)

# ==========================================
# 3. CORPS PRINCIPAL
# ==========================================
st.title("📈 Tableau de Bord Stratégique : MASI")
st.markdown("### Prévision de la trajectoire et Analyse du Risque")

# Chargement
df, clean_returns, col_prix = get_data()

if df is not None:
    # --- A. KPI EN TÊTE ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    last_price = df[col_prix].iloc[-1]
    prev_price = df[col_prix].iloc[-2]
    var_day = ((last_price - prev_price) / prev_price) * 100
    
    volatility_hist = clean_returns.std() * np.sqrt(252) * 100
    
    col1.metric("Dernier Cours (MASI)", f"{last_price:,.2f}", f"{var_day:.2f}%")
    col2.metric("Horizon Prévision", f"{horizon} Jours")
    col3.metric("Volatilité An. (Hist)", f"{volatility_hist:.2f}%")
    col4.metric("Données Disponibles", f"{len(df)} Séances")
    
    st.markdown("---")

    # ==========================================
    # --- B. PRÉVISION DE PRIX (ARIMA RECONSTRUIT) ---
    # ==========================================
    st.subheader(f"🔮 Trajectoire Prévue (Projection des Prix)")
    
    with st.spinner('Calcul des trajectoires en cours...'):
        
        # 1. Entraînement ARIMA
        # On utilise (5,1,0) qui est standard et robuste pour des données journalières
        model_arima = ARIMA(clean_returns, order=(5, 1, 0))
        fit_arima = model_arima.fit()
        
        # 2. Prévision des rendements
        forecast_res = fit_arima.get_forecast(steps=horizon)
        forecast_log_returns = forecast_res.predicted_mean
        conf_int_log = forecast_res.conf_int(alpha=0.05)
        
        # 3. RECONSTRUCTION DU PRIX (La formule magique)
        # Prix_t = Prix_{t-1} * exp(sum(r))
        last_real_date = df.index[-1]
        
        # Cumul des rendements prévus
        cumulative_returns = np.cumsum(forecast_log_returns)
        forecast_prices = last_price * np.exp(cumulative_returns)
        
        # Cumul des bornes (Approximation pour la visualisation)
        cumulative_lower = np.cumsum(conf_int_log.iloc[:, 0])
        cumulative_upper = np.cumsum(conf_int_log.iloc[:, 1])
        lower_conf_price = last_price * np.exp(cumulative_lower)
        upper_conf_price = last_price * np.exp(cumulative_upper)
        
        # Dates futures
        future_dates = pd.date_range(start=last_real_date, periods=horizon + 1, freq='B')[1:]
        
        # 4. AFFICHAGE SPLIT (Graphique à gauche, Chiffres à droite)
        c_graph, c_kpi = st.columns([3, 1])
        
        with c_graph:
            fig_arima, ax = plt.subplots(figsize=(10, 5))
            
            # Historique (90 derniers jours)
            history = df[col_prix].iloc[-90:]
            ax.plot(history.index, history.values, label='Historique Réel', color='#2c3e50', linewidth=1.5)
            
            # Pont (Liaison)
            ax.plot([last_real_date, future_dates[0]], [last_price, forecast_prices[0]], 
                    color='#e74c3c', linestyle='--')
            
            # Prévision
            ax.plot(future_dates, forecast_prices, 
                    label=f'Prévision ARIMA', color='#e74c3c', linestyle='--', marker='o', markersize=4)
            
            # Cône de confiance
            dates_cone = [last_real_date] + list(future_dates)
            lower_cone = [last_price] + list(lower_conf_price)
            upper_cone = [last_price] + list(upper_conf_price)
            
            ax.fill_between(dates_cone, lower_cone, upper_cone, color='#e74c3c', alpha=0.15, label='Zone de Confiance 95%')
            
            ax.set_title("Projection continue du MASI", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            st.pyplot(fig_arima)
            
        with c_kpi:
            st.markdown("### 🎯 Objectif")
            final_price = forecast_prices[-1]
            perf_pct = ((final_price - last_price) / last_price) * 100
            
            st.metric(
                label=f"Cible à {horizon} jours",
                value=f"{final_price:,.0f}",
                delta=f"{perf_pct:.2f} %"
            )
            
            if perf_pct > 0:
                st.success("Signal : ACHAT (Hausse)")
            else:
                st.error("Signal : VENTE (Baisse)")
                
            st.info("L'intervalle rouge indique la zone de probabilité à 95%.")

    st.markdown("---")

    # ==========================================
    # --- C. ANALYSE DU RISQUE (GARCH) ---
    # ==========================================
    st.subheader("🛡️ Analyse de la Volatilité (Risque de Marché)")
    
    with st.spinner('Modélisation de la variance (GARCH) en cours...'):
        # On multiplie par 100 pour que le GARCH converge mieux (échelle %)
        returns_scaled = clean_returns * 100
        
        # Modèle GARCH(1,1) standard
        garch = arch_model(returns_scaled, vol='Garch', p=1, q=1)
        res_garch = garch.fit(disp='off')
        
        # Forecast de la volatilité
        forecast_garch = res_garch.forecast(horizon=horizon)
        future_vol = np.sqrt(forecast_garch.variance.values[-1, :])
        
        # Indicateurs
        curr_vol = future_vol[0]
        avg_vol = res_garch.conditional_volatility.mean()
        
        c_vol1, c_vol2 = st.columns(2)
        
        # Jauge de nervosité
        etat_marche = "CALME" if curr_vol < avg_vol else "NERVEUX"
        couleur_etat = "green" if curr_vol < avg_vol else "red"
        
        c_vol1.markdown(f"#### État du Marché : :{couleur_etat}[{etat_marche}]")
        c_vol1.metric("Volatilité Prévue (Demain)", f"{curr_vol:.2f}%", delta=f"{curr_vol - avg_vol:.2f}%", delta_color="inverse")
        
        # Graphique GARCH
        fig_garch, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(res_garch.conditional_volatility.iloc[-180:], color='#f39c12', label='Volatilité Conditionnelle')
        ax2.axhline(avg_vol, color='grey', linestyle='--', label='Risque Moyen')
        ax2.set_title("Évolution de la nervosité du marché (6 derniers mois)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        c_vol2.pyplot(fig_garch)

else:
    st.warning("⚠️ Fichier de données introuvable. Veuillez vérifier l'emplacement du fichier CSV.")
