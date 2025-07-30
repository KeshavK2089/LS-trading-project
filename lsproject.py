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

# Top 50 life science trading tickers
TICKERS = [
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "BMRN", "INCY", "EXEL", "ALNY",
    "MRNA", "SGEN", "NBIX", "ACAD", "TECH", "CRSP", "BLUE", "BEAM", "NTLA", "EDIT",
    "XLRN", "VIR", "HALO", "PRTA", "FATE", "ARWR", "SRPT", "MDGL", "TGTX", "IMCR",
    "KNSA", "RNA", "VKTX", "APLS", "ACIU", "CABA", "DCPH", "EQRX", "MOR", "PTCT",
    "QURE", "SANA", "TNYA", "VERV", "XENE", "ZYME", "GLYC", "CNTB", "ASND", "RVNC"
]

# Calculate buy/sell score
def calculate_buy_score(data):
    if data is None or data.empty:
        return None

    close_prices = data['Close']
    momentum = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
    volatility = close_prices.pct_change().std() * np.sqrt(252)
    moving_avg = close_prices.rolling(window=20).mean().iloc[-1]  # ensure single value

    score = (momentum * 50) + (1 / (volatility + 1e-6) * 30)

    # FIXED: Compare single value to single value
    if close_prices.iloc[-1] > float(moving_avg):  
        score += 20

    return min(max(score, 0), 100)


# Fetch data for tickers
def fetch_data():
    results = []
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        data = yf.download(ticker, period="6mo", interval="1d")
        if data.empty:
            print(f"⚠️ Skipping {ticker}: No price data found.")
            continue

        score = calculate_buy_score(data)
        if score is None:
            continue

        results.append({
            "Ticker": ticker,
            "Buy/Sell Score": score,
            "Price": data['Close'].iloc[-1]
        })
    return pd.DataFrame(results)

# Generate a PDF report
def generate_pdf(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Life Science Trading Analysis", ln=True, align="C")

    for _, row in df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Ticker']}: Score {row['Buy/Sell Score']:.2f} Price ${row['Price']:.2f}", ln=True)

    pdf.output("trading_analysis.pdf")

# Generate graph
def plot_scores(df):
    plt.figure(figsize=(12, 6))
    plt.bar(df['Ticker'], df['Buy/Sell Score'])
    plt.xticks(rotation=90)
    plt.xlabel("Ticker")
    plt.ylabel("Buy/Sell Score (0-100)")
    plt.title("Life Science Buy/Sell Analysis")
    plt.tight_layout()
    plt.savefig("trading_chart.png")

# Email the report
def send_email_report():
    email_user = os.getenv("EMAIL_USER")
    email_password = os.getenv("EMAIL_PASSWORD")
    email_to = os.getenv("EMAIL_TO")

    if not email_user or not email_password or not email_to:
        print("⚠️ Email credentials are not set. Skipping email.")
        return

    msg = MIMEMultipart()
    msg["From"] = email_user
    msg["To"] = email_to
    msg["Subject"] = "Daily Life Science Trading Analysis"

    for file in ["trading_analysis.pdf", "trading_chart.png"]:
        with open(file, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={file}")
            msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(email_user, email_password)
        server.sendmail(email_user, email_to, msg.as_string())
        print("✅ Email sent successfully!")

# Main function
def main():
    df = fetch_data()

    if df.empty:
        print("⚠️ No valid data for any tickers.")
        return

    plot_scores(df)
    generate_pdf(df)
    send_email_report()
    print("✅ Analysis complete.")

if __name__ == "__main__":
    main()
