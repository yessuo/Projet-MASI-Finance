# 📈 Prédiction et Analyse de l'Indice MASI (Maroc)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Institution](https://img.shields.io/badge/INSEA-Data%20Science-red)

Ce projet a été réalisé dans le cadre de la formation d'Ingénieur à l'**INSEA** (Institut National de Statistique et d'Économie Appliquée), filière **Économie Appliquée, Statistique et Big Data**.

L'objectif est d'analyser l'historique de l'indice boursier marocain (MASI) et de prédire sa tendance à court terme en utilisant des modèles de séries temporelles (ARIMA).

## 📋 Description du Projet

L'analyse financière nécessite des outils robustes pour anticiper les mouvements de marché. Ce projet déploie un pipeline complet de Data Science :
1.  **Collecte & Nettoyage** : Traitement des données boursières brutes (formatage des devises, gestion des dates).
2.  **Analyse Exploratoire** : Calcul des rendements logarithmiques (*Log Returns*) pour stationnariser la série.
3.  **Modélisation** : Utilisation du modèle **ARIMA** (AutoRegressive Integrated Moving Average) pour capturer la dynamique temporelle.
4.  **Visualisation** : Génération de graphiques dynamiques incluant les intervalles de confiance à 95%.

## 🚀 Fonctionnalités Clés

* **Nettoyage Automatisé** : Conversion intelligente des formats numériques (ex: "12,000.00" -> 12000.00).
* **Stationnarisation** : Transformation des prix en rendements pour respecter les hypothèses statistiques.
* **Prévision** : Forecasting sur une fenêtre glissante (ex: 5 jours).
* **Interface / Rapport** : Visualisation claire des tendances haussières ou baissières.

## 🛠️ Stack Technique

* **Langage** : Python
* **Bibliothèques** :
    * `Pandas` & `NumPy` : Manipulation de données.
    * `Statsmodels` : Modélisation ARIMA et analyse statistique.
    * `Matplotlib` / `Seaborn` : Visualisation de données.
    * *(Optionnel : Streamlit si tu as utilisé un framework web)*

## 📂 Structure du Répertoire

```bash
├── data/               # Fichiers de données (CSV/Excel)
├── notebooks/          # Jupyter Notebooks (Exploration & Tests)
├── src/                # Scripts Python nettoyés (Code modulaire)
├── README.md           # Documentation du projet
└── requirements.txt    # Liste des dépendances
