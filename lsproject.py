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
    close_prices = data['Close']

    # Ensure we are working with a Series
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    moving_avg = close_prices.rolling(window=20).mean()
    moving_avg_value = moving_avg.iloc[-1]

    # Make sure moving_avg_value is a scalar
    if isinstance(moving_avg_value, pd.Series):
        moving_avg_value = moving_avg_value.iloc[0]

    if pd.isna(moving_avg_value):
        print("⚠️ Not enough data to calculate moving average.")
        return None

    last_close = close_prices.iloc[-1]

    score = 0
    if last_close > moving_avg_value:
        score += 1
    if last_close > close_prices.iloc[-5:].mean():
        score += 1
    if last_close > close_prices.iloc[-10:].mean():
        score += 1

    return score





# Fetch data for tickers
def fetch_data():
    tickers = ["AAPL", "MSFT", "GOOGL"]  # Example tickers
    results = []

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, period="6mo", interval="1d")

        if data.empty:
            print(f"⚠️ No data for {ticker}")
            continue

        score = calculate_buy_score(data)

        if score is None:
            print(f"⚠️ Skipping {ticker} (no score)")
            continue

        latest_price = data["Close"].iloc[-1]  # Get most recent closing price
        results.append({
            "Ticker": ticker,
            "Buy/Sell Score": float(score),
            "Price": float(latest_price)
        })

    df = pd.DataFrame(results)

    if df.empty:
        print("⚠️ No data available for any ticker.")
    
    return df



# Generate a PDF report
def generate_pdf(df):
    required_columns = {"Ticker", "Buy/Sell Score", "Price"}
    if not required_columns.issubset(df.columns):
        print(f"⚠️ Missing required columns: {required_columns - set(df.columns)}")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for _, row in df.iterrows():
        ticker = str(row["Ticker"])
        score = float(row["Buy/Sell Score"])
        price = float(row["Price"])
        pdf.cell(200, 10, txt=f"{ticker}: Score {score:.2f} Price ${price:.2f}", ln=True)

    pdf.output("analysis_report.pdf")
    print("✅ PDF generated successfully.")



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
        print("No data to plot.")
        return


    plot_scores(df)
    generate_pdf(df)
    send_email_report()
    print("✅ Analysis complete.")

if __name__ == "__main__":
    main()
