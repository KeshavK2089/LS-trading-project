

"""
Life Science Stocks Quant+Sentiment Analyzer

Analyzes top US life science stocks using price data and FDA news sentiment,
exports a table of technical and sentiment indicators to Excel.

Author: Keshav Kotteswaran
License: MIT
"""

import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
import os

# -- CONFIG --

TICKERS = [
    # [ ... as before ... ]
]

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")  # Set your own API key as env variable for security!

def fetch_fda_news_sentiment(ticker):
    """
    Get a sentiment score (1–100) from latest FDA news headlines for the ticker.
    Uses TextBlob polarity on top 5 headlines.
    """
    news_score = 50  # neutral default
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}+FDA&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        scores = []
        for article in articles[:5]:
            headline = article.get("title", "")
            if isinstance(headline, str) and headline:
                sentiment = TextBlob(headline).sentiment.polarity
                scores.append(sentiment)
        if scores:
            avg_sent = sum(scores) / len(scores)
            news_score = int((avg_sent + 1) * 50)
    except Exception:
        pass
    return max(1, min(news_score, 100))

def calculate_buy_score(data, ticker):
    """
    Blend momentum, volatility, and news into a 1–100 buy/sell score.
    """
    close_prices = data["Close"]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]
    if len(close_prices) < 21:
        return None
    short_return = (close_prices.iloc[-1] / close_prices.iloc[-5]) - 1
    medium_return = (close_prices.iloc[-1] / close_prices.iloc[-20]) - 1
    daily_returns = close_prices.pct_change().dropna()
    if len(daily_returns) < 20:
        return None
    volatility_series = daily_returns.rolling(window=20).std()
    if len(volatility_series) == 0:
        return None
    volatility = volatility_series.iloc[-1]
    if pd.isna(volatility) or float(volatility) == 0.0:
        return None
    momentum_score = min(max((short_return + medium_return) * 5000, 1), 100)
    volatility_score = max(1, min(100, 100 - (float(volatility) * 1000)))
    news_score = fetch_fda_news_sentiment(ticker)
    final_score = int(
        (0.4 * momentum_score) + (0.2 * volatility_score) + (0.4 * news_score)
    )
    return final_score

def calculate_quant_indicators(data):
    """
    Compute MA20, RSI14, MACD, Bollinger Band Width, ATR14 for a stock DataFrame.
    Returns a dictionary.
    """
    required_cols = {'Close', 'High', 'Low'}
    if (
        data is None
        or data.empty
        or not required_cols.issubset(data.columns)
        or len(data) < 21
    ):
        return {k: np.nan for k in ['MA_20','RSI_14','MACD','Boll_Band_Width','ATR_14']}
    indicators = {}
    close = data['Close']
    # 1. MA_20
    ma20_series = close.rolling(window=20).mean()
    indicators['MA_20'] = float(ma20_series.iloc[-1]) if len(ma20_series) > 0 else np.nan
    # 2. RSI_14
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    ag = float(avg_gain.iloc[-1]) if len(avg_gain) > 0 else np.nan
    al = float(avg_loss.iloc[-1]) if len(avg_loss) > 0 else np.nan
    if al == 0 or np.isnan(al):
        rsi = 100
    else:
        rs = ag / al
        rsi = 100 - (100 / (1 + rs))
    indicators['RSI_14'] = rsi
    # 3. MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    indicators['MACD'] = float(macd.iloc[-1]) if len(macd) > 0 else np.nan
    # 4. Bollinger Band Width
    std20 = close.rolling(window=20).std()
    ma20_series = close.rolling(window=20).mean()
    upper = ma20_series + (2 * std20)
    lower = ma20_series - (2 * std20)
    upper_last = upper.iloc[-1]
    lower_last = lower.iloc[-1]
    if (
        len(upper) > 0 and len(lower) > 0
        and not pd.isna(upper_last) and not pd.isna(lower_last)
    ):
        indicators['Boll_Band_Width'] = float(upper_last - lower_last)
    else:
        indicators['Boll_Band_Width'] = np.nan
    # 5. ATR_14
    high = data['High']
    low = data['Low']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    atr_series = tr.rolling(window=14).mean()
    indicators['ATR_14'] = float(atr_series.iloc[-1]) if len(atr_series) > 0 else np.nan
    return indicators

def fetch_data():
    """
    Loop through tickers, collect indicators, skip those with missing data, return DataFrame.
    """
    results = []
    skipped_tickers = []
    for ticker in TICKERS:
        print(f"\n==== {ticker} ====")
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        print(f"Downloaded {len(data)} rows for {ticker}")
        if data.empty:
            print(" - Data is empty. Skipping.")
            skipped_tickers.append((ticker, "empty"))
            continue
        if len(data) < 21:
            print(f" - Not enough rows ({len(data)}). Need at least 21. Skipping.")
            skipped_tickers.append((ticker, f"rows={len(data)}"))
            continue
        score = calculate_buy_score(data, ticker)
        if score is None:
            print(" - Buy score is None. Skipping.")
            skipped_tickers.append((ticker, "buy_score_None"))
            continue
        quant_indicators = calculate_quant_indicators(data)
        latest_price = float(data["Close"].iloc[-1])
        results.append({
            "Ticker": ticker,
            "Buy/Sell Score": float(score),
            "Price": float(latest_price),
            "MA_20": quant_indicators['MA_20'],
            "RSI_14": quant_indicators['RSI_14'],
            "MACD": quant_indicators['MACD'],
            "Boll_Band_Width": quant_indicators['Boll_Band_Width'],
            "ATR_14": quant_indicators['ATR_14'],
        })
    if skipped_tickers:
        print("\nSummary of skipped tickers and reasons:")
        for ticker, reason in skipped_tickers:
            print(f" - {ticker}: {reason}")
    else:
        print("\nAll tickers processed!")
    return pd.DataFrame(results)

def export_to_excel(df):
    # If DataFrame is empty, create one with a message
    if df.empty:
        df = pd.DataFrame([{"Message": "No data available for any ticker."}])
    df.to_excel("life_science_trading_analysis.xlsx", index=False)
    print("Excel file saved as life_science_trading_analysis.xlsx")


def main():
    df = fetch_data()
    export_to_excel(df)  # Always create file, even if empty
    if df.empty:
        print("⚠️ No data available.")

if __name__ == "__main__":
    main()
