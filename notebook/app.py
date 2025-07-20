import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import joblib
import plotly.graph_objs as go
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
import statsmodels.api as sm
from tensorflow import keras
import os
#executer cette comande dans le terminal streamlit run notebook/app.py
MODEL_DIR = "notebook"  # All model and scaler files are in ./notebook/

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    return df

def compute_indicators(df):
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError(f"La colonne 'Close' est absente du DataFrame : {df.columns}")
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
    df['EMA_21'] = EMAIndicator(close=df['Close'], window=21).ema_indicator()
    bb = BollingerBands(close=df['Close'], window=20)
    df['BOLL_MID_20'] = bb.bollinger_mavg()
    df['BOLL_UP_20'] = bb.bollinger_hband()
    df['BOLL_LOW_20'] = bb.bollinger_lband()
    df['RET_1D'] = df['Close'].pct_change()
    df['LOGRET_1D'] = np.log(df['Close'] / df['Close'].shift(1))
    df['VOLAT_14D'] = df['RET_1D'].rolling(14).std()
    for col in ['RSI_14', 'MACD', 'MACD_signal', 'SMA_50', 'SMA_200', 'EMA_21',
                'BOLL_MID_20', 'BOLL_UP_20', 'BOLL_LOW_20', 'RET_1D', 'LOGRET_1D', 'VOLAT_14D']:
        if col in df:
            df[f"Z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    return df

def patch_ticker_specific_columns(df, ticker, features):
    new_df = df.copy()
    for base in ['RET_1D', 'VOLAT_14D']:
        zbase = f'Z_{base}'
        zpatch = f'Z_{base}_{ticker}'
        if zpatch in features and zpatch not in new_df.columns and zbase in new_df.columns:
            new_df[zpatch] = new_df[zbase]
    return new_df

def ensure_features(df, features, ticker):
    df = patch_ticker_specific_columns(df, ticker, features)
    df['const'] = 1.0
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0.0
    X = df[features]
    return X

@st.cache_resource
def load_model(path_model, path_features):
    with open(path_model, "rb") as f:
        model = pickle.load(f)
    with open(path_features, "rb") as f:
        features = pickle.load(f)
    return model, features

@st.cache_resource
def load_lstm_model_and_scalers(model_path, scaler_y_path, scaler_X_path, features_path):
    model = keras.models.load_model(model_path)
    scaler_y = joblib.load(scaler_y_path)
    scaler_X = joblib.load(scaler_X_path)
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    return model, scaler_y, scaler_X, features

# ---- Streamlit UI Setup ----
st.set_page_config(page_title="Crypto Dashboard Quant", layout="wide")
st.title("Dashboard quantitatif crypto : analyse et prédiction")

cryptos = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
selected_cryptos = st.multiselect(
    "Sélectionne une ou plusieurs cryptos :", options=list(cryptos.keys()), default=["Bitcoin", "Ethereum"]
)
n_days = st.slider("Nombre de jours d'historique", min_value=60, max_value=1800, value=730, step=10)
start_date = pd.Timestamp.today() - pd.Timedelta(days=n_days)
end_date = pd.Timestamp.today()

data_dict = {}
for name in selected_cryptos:
    ticker = cryptos[name]
    df = yf.download(ticker, start=start_date, end=end_date, group_by="ticker")
    df = flatten_columns(df)
    for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
        col1 = f"{base}_{ticker}"
        col2 = f"{ticker}_{base}"
        if col1 in df.columns:
            df[base] = df[col1]
        elif col2 in df.columns:
            df[base] = df[col2]
        elif base in df.columns:
            pass
        else:
            raise ValueError(f"Impossible de trouver la colonne {base} pour {ticker}. Colonnes: {df.columns}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.reset_index().set_index("Date")
    df = compute_indicators(df)
    data_dict[name] = df

tab1, tab2 = st.tabs(["Graphiques financiers", "Prédiction modèle"])

with tab1:
    for name, df in data_dict.items():
        st.subheader(f"Prix et indicateurs : {name}")
        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Bougies"
            ),
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker=dict(opacity=0.2),
                yaxis="y2"
            ),
        ])
        fig.update_layout(
            yaxis2=dict(overlaying='y', side='right', showgrid=False, title="Volume"),
            title=f"{name} : prix (USD) et volumes",
            xaxis_title="Date",
            yaxis_title="Prix (USD)",
            legend=dict(orientation='h')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.line_chart(df[['Close', 'RSI_14', 'MACD', 'SMA_50', 'SMA_200']].dropna())

with tab2:
    st.subheader("Prédiction (BTC & ETH modèles)")

    # Sélection du modèle dans l'onglet seulement
    model_type = st.selectbox(
        "Type de modèle",
        ["Quantile Regression", "LSTM Deep Learning"],
        index=0,
        key="model_type_tab2"
    )

    for name in selected_cryptos:
        ticker = cryptos[name]
        df = data_dict[name].copy()
        if model_type == "Quantile Regression":
            if name == "Bitcoin":
                model_path = os.path.join(MODEL_DIR, "quantile_regression_btc.pkl")
                features_path = os.path.join(MODEL_DIR, "selected_features_btc.pkl")
            else:
                model_path = os.path.join(MODEL_DIR, "quant_reg_eth.pkl")
                features_path = os.path.join(MODEL_DIR, "selected_features_eth.pkl")
            quant_reg, features = load_model(model_path, features_path)
            X_pred = ensure_features(df, features, ticker)
            X_pred = sm.add_constant(X_pred)
            if not X_pred.empty:
                y_pred = quant_reg.predict(X_pred)
                df_pred = df.loc[X_pred.index].copy()
                df_pred["Prédiction"] = y_pred
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Close'], mode='lines', name='Prix réel'))
                fig2.add_trace(go.Scatter(x=df_pred.index, y=df_pred["Prédiction"], mode='lines', name='Prix prédit'))
                fig2.update_layout(
                    title=f"Prévision {name} (modèle quantile regression)",
                    xaxis_title="Date",
                    yaxis_title="Prix (USD)"
                )
                st.plotly_chart(fig2, use_container_width=True)
                # Tableau des 10 dernières prédictions
                st.write("**Tableau des 10 dernières prédictions :**")
                st.dataframe(
                    df_pred[["Close", "Prédiction"]].tail(10).rename(columns={"Close": "Prix réel"}),
                    use_container_width=True
                )
            else:
                st.warning(f"Pas assez de données pour prédire ({name}).")
        else:
            if name == "Bitcoin":
                model_path = os.path.join(MODEL_DIR, "best_lstm_model.keras")
                scaler_y_path = os.path.join(MODEL_DIR, "scaler_y_btc.save")
                scaler_X_path = os.path.join(MODEL_DIR, "scaler_X_btc.save")
                features_path = os.path.join(MODEL_DIR, "selected_features_btc.pkl")
            else:
                model_path = os.path.join(MODEL_DIR, "best_lstm_eth.keras")
                scaler_y_path = os.path.join(MODEL_DIR, "scaler_y_eth.save")
                scaler_X_path = os.path.join(MODEL_DIR, "scaler_X_eth.save")
                features_path = os.path.join(MODEL_DIR, "selected_features_eth.pkl")
            lstm_model, scaler_y, scaler_X, features = load_lstm_model_and_scalers(
                model_path, scaler_y_path, scaler_X_path, features_path
            )
            X_pred = ensure_features(df, features, ticker).values
            X_pred_scaled = scaler_X.transform(X_pred)
            seq_length = 60
            X_seq = []
            for i in range(seq_length, len(X_pred_scaled)):
                X_seq.append(X_pred_scaled[i-seq_length:i])
            X_seq = np.array(X_seq)
            if X_seq.shape[0] > 0:
                y_pred_scaled = lstm_model.predict(X_seq)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                idx_pred = df.index[seq_length:]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=idx_pred, y=df['Close'].iloc[seq_length:], mode='lines', name='Prix réel'))
                fig2.add_trace(go.Scatter(x=idx_pred, y=y_pred.flatten(), mode='lines', name='Prix prédit (LSTM)'))
                fig2.update_layout(
                    title=f"Prévision {name} (LSTM Deep Learning)",
                    xaxis_title="Date",
                    yaxis_title="Prix (USD)"
                )
                st.plotly_chart(fig2, use_container_width=True)
                # Tableau des 10 dernières prédictions
                df_last_pred = pd.DataFrame({
                    "Date": idx_pred,
                    "Prix réel": df["Close"].iloc[seq_length:].values,
                    "Prédiction": y_pred.flatten()
                }).set_index("Date")
                st.write("**Tableau des 10 dernières prédictions :**")
                st.dataframe(df_last_pred.tail(10), use_container_width=True)
            else:
                st.warning(f"Pas assez de données pour prédire ({name}) avec LSTM.")

# Sidebar prédiction rapide = quantile regression par défaut (pour ne pas mélanger UX)
st.sidebar.title("Prédiction en temps réel")
crypto_live = st.sidebar.selectbox("Crypto pour la prévision live", options=list(cryptos.keys()))
if crypto_live == "Bitcoin":
    model_path = os.path.join(MODEL_DIR, "quantile_regression_btc.pkl")
    features_path = os.path.join(MODEL_DIR, "selected_features_btc.pkl")
    ticker = "BTC-USD"
else:
    model_path = os.path.join(MODEL_DIR, "quant_reg_eth.pkl")
    features_path = os.path.join(MODEL_DIR, "selected_features_eth.pkl")
    ticker = "ETH-USD"
quant_reg, features = load_model(model_path, features_path)
df = data_dict[crypto_live].copy()
X_pred = ensure_features(df, features, ticker)
X_pred = sm.add_constant(X_pred)
if not X_pred.empty:
    pred_price = quant_reg.predict(X_pred)[-1]
    st.sidebar.metric(f"Prévision quantile regression ({crypto_live})", f"{pred_price:.2f} USD")
else:
    st.sidebar.warning("Pas assez de données pour prédire (features manquantes).")

st.sidebar.info(
    "Modèle : Quantile Regression\n\nIndicateurs techniques calculés automatiquement.\n"
    "Prédiction = prix de clôture J+1 (daily)."
)
