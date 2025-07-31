import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
import os

# Top 50 life science trading tickers
TICKERS = [
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "BMRN", "INCY", "EXEL", "ALNY",
    "MRNA", "SGEN", "NBIX", "ACAD", "TECH", "CRSP", "BLUE", "BEAM", "NTLA", "EDIT",
    "XLRN", "VIR", "HALO", "PRTA", "FATE", "ARWR", "SRPT", "MDGL", "TGTX", "IMCR",
    "KNSA", "RNA", "VKTX", "APLS", "ACIU", "CABA", "DCPH", "EQRX", "MOR", "PTCT",
    "QURE", "SANA", "TNYA", "VERV", "XENE", "ZYME", "GLYC", "CNTB", "ASND", "RVNC"
]

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "db26f5af4bcb4870aaccc9026dd51f27")

def fetch_fda_news_sentiment(ticker):
    news_score = 50  # neutral default
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}+FDA&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        scores = []
        for article in articles[:5]:
            headline = article.get("title", "")
            if isinstance(headline, str) and len(headline) > 0:
                sentiment = TextBlob(headline).sentiment.polarity
                scores.append(sentiment)

        if len(scores) > 0:
            avg_sent = sum(scores) / len(scores)
            news_score = int((avg_sent + 1) * 50)
    except Exception:
        pass

    return max(1, min(news_score, 100))

def calculate_buy_score(data, ticker):
    close_prices = data["Close"]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    if len(close_prices) < 21:  # Need at least 21 rows for 20-period rolling
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

def fetch_data():
    results = []
    skipped_tickers = []
    for ticker in TICKERS:
        print(f"\n==== {ticker} ====")
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        print(f"Downloaded {len(data)} rows for {ticker}")
        print("Columns:", list(data.columns))
        print(data.head())

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
        print("Quant indicators:", quant_indicators)

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

def calculate_quant_indicators(data):
    # Defensive: If data is empty or missing needed columns, return nan indicators
    required_cols = {'Close', 'High', 'Low'}
    if (
        data is None
        or data.empty
        or not required_cols.issubset(data.columns)
        or len(data) < 21
    ):
        print(" - Data missing required columns or too short. Returning all NaN quant indicators.")
        return {
            'MA_20': np.nan,
            'RSI_14': np.nan,
            'MACD': np.nan,
            'Boll_Band_Width': np.nan,
            'ATR_14': np.nan,
        }

    indicators = {}
    close = data['Close']

    # 1. 20-day Moving Average
    ma20_series = close.rolling(window=20).mean()
    indicators['MA_20'] = float(ma20_series.iloc[-1]) if len(ma20_series) > 0 else np.nan

    # 2. 14-day RSI
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

    # 3. MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    indicators['MACD'] = float(macd.iloc[-1]) if len(macd) > 0 else np.nan

    # 4. Bollinger Band width (20, 2)
    std20 = close.rolling(window=20).std()
    ma20_series = close.rolling(window=20).mean()  # Recreate to be sure
    upper = ma20_series + (2 * std20)
    lower = ma20_series - (2 * std20)

    upper_last = upper.iloc[-1]
    lower_last = lower.iloc[-1]

    if isinstance(upper_last, pd.Series):
        upper_last = upper_last.iloc[0]
    if isinstance(lower_last, pd.Series):
        lower_last = lower_last.iloc[0]

    if (
        len(upper) > 0 and len(lower) > 0
        and not pd.isna(upper_last) and not pd.isna(lower_last)
    ):
        indicators['Boll_Band_Width'] = float(upper_last - lower_last)
    else:
        indicators['Boll_Band_Width'] = np.nan

    # 5. ATR (14-day)
    high = data['High']
    low = data['Low']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    atr_series = tr.rolling(window=14).mean()
    indicators['ATR_14'] = float(atr_series.iloc[-1]) if len(atr_series) > 0 else np.nan

    return indicators


def export_to_excel(df):
    df.to_excel("life_science_trading_analysis.xlsx", index=False)
    print("Excel file saved as life_science_trading_analysis.xlsx")

def main():
    df = fetch_data()
    if df.empty:
        print("⚠️ No data available.")
    else:
        export_to_excel(df)

if __name__ == "__main__":
    main()
