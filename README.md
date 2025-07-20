
# 📈 Dashboard Crypto – Analyse et Prédiction

Ce projet propose un **dashboard dynamique** pour analyser et prédire les prix des cryptomonnaies (Bitcoin, Ethereum) en utilisant :
- Des modèles de Machine Learning (régression quantile, LSTM)
- Des indicateurs financiers techniques
- Une interface développée avec **Streamlit**

## 📁 Structure du projet

Tout le code est regroupé dans le dossier [`notebook/`](./notebook), incluant :

- `app.py` : application Streamlit
- `crypto_analysis.ipynb` : notebook d’analyse et de modélisation
- `*.keras`, `*.pkl`, `*.save` : modèles LSTM, scalers, régressions quantiles
- Dossiers auxiliaires : tuning, tests, etc.

## 📡 Données

Les données de prix sont téléchargées dynamiquement à l’aide de bibliothèques comme `yfinance` ou autres APIs cryptos.  
Aucune donnée locale n'est stockée.

## ▶️ Lancer le dashboard

```bash
cd notebook
streamlit run app.py
