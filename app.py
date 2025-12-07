
import requests
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from flatlib.chart import Chart
from flatlib.datetime import Datetime
from flatlib.geopos import GeoPos
from flatlib import const
import yfinance as yf

# Define location (required for flatlib)
GEO = GeoPos('28.6139', '77.2090')  # New Delhi

def get_planetary_positions(date_str):
    dt = Datetime(date_str, '12:00', '+05:30')
    chart = Chart(dt, GEO)
    planets = [const.SUN, const.MOON, const.MARS, const.MERCURY, const.JUPITER, const.VENUS, const.SATURN]
    positions = {}
    for planet in planets:
        positions[planet.title()] = chart.get(planet).lon
    return positions

def get_btc_price():
    try:
        res = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr")
        return res.json()['bitcoin']['inr']
    except:
        return "N/A"

def get_sensex_price():
    try:
        url = "https://www.niftyindices.com/Backpage.aspx/GetIndexDetails"
        headers = {"Content-Type": "application/json"}
        payload = "{"indexName":"S&P BSE SENSEX"}"
        res = requests.post(url, data=payload, headers=headers)
        return res.json()['d'][0]['lastPrice']
    except:
        return "N/A"

def get_nifty_price():
    try:
        url = "https://www.nseindia.com/api/quote-equity?symbol=NIFTY"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/"
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        res = session.get(url, headers=headers)
        data = res.json()
        return data['priceInfo']['lastPrice']
    except:
        return "N/A"

def download_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Open', 'Close']]
    df['Date'] = df.index.date.astype(str)
    return df

def create_astro_stock_dataset(symbol, start_date, end_date):
    stock_df = download_stock_data(symbol, start_date, end_date)
    records = []
    for _, row in stock_df.iterrows():
        date = row['Date']
        try:
            planet_pos = get_planetary_positions(date)
            trend = 'Up' if row['Close'] > row['Open'] else 'Down'
            record = {**planet_pos, 'market_trend': trend, 'Date': date}
            records.append(record)
        except:
            continue
    return pd.DataFrame(records)

def train_market_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=['market_trend', 'Date'])
    y = df['market_trend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")
    joblib.dump(model, 'astro_model.pkl')
    return model

def predict_dasha_outlook(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    if year % 7 == 0:
        return "📉 High volatility expected due to Saturn period."
    elif year % 5 == 0:
        return "💰 Potential wealth growth — Jupiter dominant year."
    else:
        return "📊 Neutral dasha period. Watch for Moon and Mars transits."

def launch_streamlit_app():
    st.set_page_config(page_title="Astro Stock Predictor", layout="centered")
    st.title("🪐 Astro-Based Stock Market Predictor")

    st.metric("🔴 Bitcoin Price (INR)", get_btc_price())
    st.metric("📊 Sensex Price (BSE)", get_sensex_price())
    st.metric("📈 Nifty Price (NSE)", get_nifty_price())

    date = st.date_input("Select a Date", datetime.today())
    symbol = st.text_input("Stock Symbol (Yahoo Finance)", value="^NSEI")
    planet = st.selectbox("Visualize Planet vs Market", ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn'])

    if st.button("Predict Trend"):
        pos = get_planetary_positions(date.strftime('%Y-%m-%d'))
        input_df = pd.DataFrame([pos])
        try:
            model = joblib.load('astro_model.pkl')
            prediction = model.predict(input_df)[0]
            st.success(f"📈 Predicted Market Trend on {date.strftime('%Y-%m-%d')}: {prediction}")
            st.info(predict_dasha_outlook(date.strftime('%Y-%m-%d')))
        except:
            st.error("Model file not found. Please train it first.")

    if st.button("Show Planetary Chart"):
        fig, ax = plt.subplots(figsize=(10, 5))
        pos = get_planetary_positions(date.strftime('%Y-%m-%d'))
        sns.barplot(x=list(pos.keys()), y=list(pos.values()), palette='mako', ax=ax)
        ax.set_title(f'Planetary Positions on {date.strftime("%Y-%m-%d")}')
        ax.set_ylabel('Longitude (°)')
        st.pyplot(fig)

    if st.button("Plot Planet vs Market"):
        df = download_stock_data(symbol, date - timedelta(days=30), date)
        df['Date'] = pd.to_datetime(df['Date'])
        df[planet] = df['Date'].dt.strftime('%Y-%m-%d').apply(lambda d: get_planetary_positions(d)[planet])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label='Market Close')
        ax.plot(df['Date'], df[planet], label=f'{planet} Degree', linestyle='--')
        ax.set_title(f'{symbol} Close vs {planet} Position')
        ax.set_xlabel('Date')
        ax.legend()
        st.pyplot(fig)

# === To Launch ===
# launch_streamlit_app()
