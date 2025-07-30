import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from textblob import TextBlob
import requests

# Top 50 life science trading tickers
TICKERS = [
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "BMRN", "INCY", "EXEL", "ALNY",
    "MRNA", "SGEN", "NBIX", "ACAD", "TECH", "CRSP", "BLUE", "BEAM", "NTLA", "EDIT",
    "XLRN", "VIR", "HALO", "PRTA", "FATE", "ARWR", "SRPT", "MDGL", "TGTX", "IMCR",
    "KNSA", "RNA", "VKTX", "APLS", "ACIU", "CABA", "DCPH", "EQRX", "MOR", "PTCT",
    "QURE", "SANA", "TNYA", "VERV", "XENE", "ZYME", "GLYC", "CNTB", "ASND", "RVNC"
]

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")  # Load from env if available

def fetch_fda_news_sentiment(ticker):
    news_score = 50  # Neutral default score
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}+FDA&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        sentiment_scores = []
        for article in articles[:5]:  # Only analyze top 5
            headline = article.get("title", "")
            if headline:
                sentiment = TextBlob(headline).sentiment.polarity
                sentiment_scores.append(sentiment)

        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            news_score = int((avg_sentiment + 1) * 50)  # -1→1 to 0→100 scale
    except Exception as e:
        print(f"⚠️ News sentiment fetch failed for {ticker}: {e}")

    return max(1, min(news_score, 100))

def calculate_buy_score(data, ticker):
    close_prices = data["Close"]

    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    if len(close_prices) < 20:
        print(f"⚠️ Not enough data to calculate score for {ticker}")
        return None

    # Price momentum
    short_return = (close_prices.iloc[-1] / close_prices.iloc[-5]) - 1
    medium_return = (close_prices.iloc[-1] / close_prices.iloc[-20]) - 1

    # Volatility
    daily_returns = close_prices.pct_change().dropna()
    volatility = daily_returns.rolling(window=20).std().iloc[-1]

    if pd.isna(volatility) or volatility == 0:
        return None

    # Normalize scores
    momentum_score = min(max((short_return + medium_return) * 5000, 1), 100)
    volatility_score = max(1, min(100, 100 - (volatility * 1000)))
    news_score = fetch_fda_news_sentiment(ticker)

    # Weighted score
    final_score = int((0.4 * momentum_score) + (0.2 * volatility_score) + (0.4 * news_score))
    return final_score

def fetch_data():
    results = []
    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)

        if data.empty:
            print(f"⚠️ No data for {ticker}")
            continue

        score = calculate_buy_score(data, ticker)
        if score is None:
            print(f"⚠️ Skipping {ticker} (not enough data)")
            continue

        latest_price = data["Close"].iloc[-1]
        results.append({
            "Ticker": ticker,
            "Buy/Sell Score": float(score),
            "Price": float(latest_price)
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠️ No data available for any ticker.")
    return df

def generate_pdf(df):
    required_columns = {"Ticker", "Buy/Sell Score", "Price"}
    if not required_columns.issubset(df.columns):
        print(f"⚠️ Missing required columns: {required_columns - set(df.columns)}")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Life Science Trading Analysis", ln=True, align="C")
    pdf.ln(10)

    for _, row in df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Ticker']}: Score {row['Buy/Sell Score']:.2f} Price ${row['Price']:.2f}", ln=True)

    pdf.output("analysis_report.pdf")
    print("✅ PDF generated successfully.")

def plot_scores(df):
    if df.empty:
        print("⚠️ No data to plot.")
        return
    plt.figure(figsize=(12, 6))
    plt.bar(df["Ticker"], df["Buy/Sell Score"])
    plt.xticks(rotation=90)
    plt.xlabel("Ticker")
    plt.ylabel("Buy/Sell Score (0-100)")
    plt.title("Life Science Buy/Sell Analysis")
    plt.tight_layout()
    plt.savefig("trading_chart.png")
    print("✅ Chart generated successfully.")

def send_email_report():
    email_user = os.getenv("EMAIL_USER", "animalcafe98398@gmail.com")
    email_password = os.getenv("EMAIL_PASS", "hfxc ozcr pojb qwzu")
    email_to = "keshavkotteswaran@gmail.com"

    msg = MIMEMultipart()
    msg["From"] = email_user
    msg["To"] = email_to
    msg["Subject"] = "Daily Life Science Trading Analysis"

    for file in ["analysis_report.pdf", "trading_chart.png"]:
        if os.path.exists(file):
            with open(file, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={file}")
                msg.attach(part)
        else:
            print(f"⚠️ File not found: {file}")

import ssl

context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login("animalcafe98398@gmail.com", "hfxc ozcr pojb qwzu")   # ← your 16-char App Password
    server.sendmail("animalcafe98398@gmail.com", "keshavkotteswaran@gmail.com", msg.as_string())


def main():
    df = fetch_data()
    if df.empty:
        print("No data to plot.")
        return
    plot_scores(df)
    generate_pdf(df)
    send_email_report()
    print("✅ Analysis complete.")

if __name__ == "__main__":
    main()
