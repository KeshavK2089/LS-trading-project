
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime
import numpy as np

# List of top 50 life-science tickers
TICKERS = [
    'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'NBIX', 'INCY', 'ALNY', 'SAGE',
    'PTCT', 'SRPT', 'EXEL', 'IONS', 'BMRN', 'HALO', 'ACAD', 'ARGX', 'BNTX', 'NVO',
    'PFE', 'LLY', 'AZN', 'RHHBY', 'JNJ', 'MRK', 'SNY', 'BAYRY', 'NVS', 'GSK',
    'ABBV', 'AMRX', 'TEVA', 'HZNP', 'PRGO', 'SUPN', 'XENE', 'VKTX', 'CRSP', 'EDIT',
    'BLUE', 'BEAM', 'NTLA', 'ARWR', 'RNA', 'KRTX', 'RVMD', 'MDGL', 'AMYT', 'AGIO'
]

# ClinicalTrials.gov API endpoint
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/query/study_fields"

def fetch_clinical_data(ticker):
    params = {
        'expr': ticker,
        'fields': 'Phase,OverallStatus',
        'min_rnk': 1,
        'max_rnk': 100,
        'fmt': 'json'
    }
    response = requests.get(CLINICAL_TRIALS_API, params=params)
    data = response.json()
    trials = data.get('StudyFieldsResponse', {}).get('StudyFields', [])
    
    phase_counts = {'Phase 1': 0, 'Phase 2': 0, 'Phase 3': 0}
    for trial in trials:
        phases = trial.get('Phase', [])
        for phase in phases:
            if 'Phase 3' in phase:
                phase_counts['Phase 3'] += 1
            elif 'Phase 2' in phase:
                phase_counts['Phase 2'] += 1
            elif 'Phase 1' in phase:
                phase_counts['Phase 1'] += 1
    
    return phase_counts

def calculate_buy_sell_score(momentum, volatility, pipeline_sentiment, fda_impact, sector_weight=1.0):
    return (
        0.4 * momentum +
        0.3 * (1 - volatility) +
        0.2 * pipeline_sentiment * sector_weight +
        0.1 * fda_impact
    ) * 100

def generate_report(df):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_file = f'quant_trading_report_{timestamp}.xlsx'
    pdf_file = f'quant_trading_report_{timestamp}.pdf'
    
    df.to_excel(excel_file, index=False)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Life Science Quant Trading Report", ln=True, align="C")
    
    for i, row in df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Ticker']} - Score: {row['Buy/Sell Score']:.2f}", ln=True)
    
    plt.figure()
    plt.bar(df['Ticker'], df['Buy/Sell Score'])
    plt.xticks(rotation=90)
    plt.ylabel('Buy/Sell Score')
    plt.title('Quantitative Trading Scores')
    plt.tight_layout()
    chart_path = f"score_chart_{timestamp}.png"
    plt.savefig(chart_path)
    
    pdf.add_page()
    pdf.image(chart_path, x=10, y=20, w=180)
    
    pdf.output(pdf_file)
    
    return excel_file, pdf_file

def send_email(files, recipients):
    sender = "keshavkotteswaran@gmail.com"
    password = "vjco uaki hszw aefn"
    server = smtplib.SMTP('smtp.gmail.com', 587)

    for recipient in recipients:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = "Life Science Quant Trading Report"
        
        msg.attach(MIMEText("Attached is the latest quant trading report with FDA clinical data.", 'plain'))
        
        for file in files:
            attachment = open(file, 'rb')
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {file}")
            msg.attach(part)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()

def main():
    results = []
    
    for ticker in TICKERS:
        data = yf.download(ticker, period="6mo", interval="1d")
        momentum = (data['Close'][-1] - data['Close'][0]) / data['Close'][0]
        volatility = data['Close'].pct_change().std()
        
        phase_counts = fetch_clinical_data(ticker)
        pipeline_sentiment = (phase_counts['Phase 1']*0.25 + phase_counts['Phase 2']*0.5 + phase_counts['Phase 3']*1.0) / 10
        
        fda_impact = 0.05 if phase_counts['Phase 3'] > 0 else 0
        
        sector_weight = 1.2 if ticker in ['BIIB', 'VRTX', 'MRNA', 'BMRN', 'REGN'] else 1.0
        
        score = calculate_buy_sell_score(momentum, volatility, pipeline_sentiment, fda_impact, sector_weight)
        
        results.append({
            'Ticker': ticker,
            'Momentum': momentum,
            'Volatility': volatility,
            'Active Trials': sum(phase_counts.values()),
            'Phase 3 Trials': phase_counts['Phase 3'],
            'Pipeline Sentiment': pipeline_sentiment,
            'FDA Impact': fda_impact,
            'Buy/Sell Score': score
        })
    
    df = pd.DataFrame(results)
    excel_file, pdf_file = generate_report(df)
    
    recipients = ["kotteswaran.k@northeastern.edu"]
    send_email([excel_file, pdf_file], recipients)

if __name__ == '__main__':
    main()
