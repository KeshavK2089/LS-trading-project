import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from textblob import TextBlob
import requests
import os
from pathlib import Path

# Top 50 life science trading tickers
TICKERS = [
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "BMRN", "INCY", "EXEL", "ALNY",
    "MRNA", "SGEN", "NBIX", "ACAD", "TECH", "CRSP", "BLUE", "BEAM", "NTLA", "EDIT",
    "XLRN", "VIR", "HALO", "PRTA", "FATE", "ARWR", "SRPT", "MDGL", "TGTX", "IMCR",
    "KNSA", "RNA", "VKTX", "APLS", "ACIU", "CABA", "DCPH", "EQRX", "MOR", "PTCT",
    "QURE", "SANA", "TNYA", "VERV", "XENE", "ZYME", "GLYC", "CNTB", "ASND", "RVNC"
]

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")


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
            if headline:
                sentiment = TextBlob(headline).sentiment.polarity
                scores.append(sentiment)

        if scores:
            avg_sent = sum(scores) / len(scores)
            news_score = int((avg_sent + 1) * 50)  # scale -1→1 to 0→100
    except Exception:
        pass

    return max(1, min(news_score, 100))


def calculate_buy_score(data, ticker):
    close_prices = data["Close"]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    if len(close_prices) < 20:
        return None

    short_return = (close_prices.iloc[-1] / close_prices.iloc[-5]) - 1
    medium_return = (close_prices.iloc[-1] / close_prices.iloc[-20]) - 1

    daily_returns = close_prices.pct_change().dropna()
    volatility = daily_returns.rolling(window=20).std().iloc[-1]

    if pd.isna(volatility) or volatility == 0:
        return None

    momentum_score = min(max((short_return + medium_return) * 5000, 1), 100)
    volatility_score = max(1, min(100, 100 - (volatility * 1000)))
    news_score = fetch_fda_news_sentiment(ticker)

    final_score = int(
        (0.4 * momentum_score) + (0.2 * volatility_score) + (0.4 * news_score)
    )
    return final_score


def fetch_data():
    results = []
    for ticker in TICKERS:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if data.empty:
            continue

        score = calculate_buy_score(data, ticker)
        if score is None:
            continue

        latest_price = data["Close"].iloc[-1]
        results.append({
            "Ticker": ticker,
            "Buy/Sell Score": float(score),
            "Price": float(latest_price)
        })

    return pd.DataFrame(results)


def generate_pdf(df):
    required_cols = {"Ticker", "Buy/Sell Score", "Price"}
    if not required_cols.issubset(df.columns):
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Life Science Trading Analysis", ln=True, align="C")
    pdf.ln(10)

    for _, row in df.iterrows():
        pdf.cell(
            200, 10,
            txt=f"{row['Ticker']}: Score {row['Buy/Sell Score']:.2f} Price ${row['Price']:.2f}",
            ln=True
        )

    pdf.output("analysis_report.pdf")


def plot_scores(df):
    if df.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.bar(df["Ticker"], df["Buy/Sell Score"])
    plt.xticks(rotation=90)
    plt.xlabel("Ticker")
    plt.ylabel("Buy/Sell Score (0-100)")
    plt.title("Life Science Buy/Sell Analysis")
    plt.tight_layout()
    plt.savefig("trading_chart.png")


def main():
    df = fetch_data()
    if df.empty:
        print("⚠️ No data available.")
    else:
        plot_scores(df)
        generate_pdf(df)

    # Move any existing outputs into the 'site' folder (NEW FIX)
    output_dir = Path("site")
    output_dir.mkdir(exist_ok=True)

    for file_name in ("analysis_report.pdf", "trading_chart.png"):
        src = Path(file_name)
        if src.exists():  # <--- prevents FileNotFoundError
            src.rename(output_dir / src.name)

    # Create a basic index.html if any files exist
    index_html = output_dir / "index.html"
    index_html.write_text(f"""
    <html>
      <head><title>Life Science Report</title></head>
      <body style="font-family: Arial; padding: 2rem;">
        <h1>Life Science Trading Analysis</h1>
        <p>Last updated: {pd.Timestamp.now().date()}</p>
        <ul>
          {"<li><a href='analysis_report.pdf'>PDF Report</a></li>" if (output_dir / "analysis_report.pdf").exists() else ""}
          {"<li><a href='trading_chart.png'>Chart</a></li>" if (output_dir / "trading_chart.png").exists() else ""}
        </ul>
      </body>
    </html>
    """)


if __name__ == "__main__":
    main()
