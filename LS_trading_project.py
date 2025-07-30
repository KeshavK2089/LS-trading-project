"""
LS_trading_project.py
========================

This module provides a simple framework for running a momentum‑style
quantitative trading analysis on pharmaceutical stocks.  It fetches daily
price data from the Stooq website, calculates a set of technical
indicators and generates buy/sell signals based on those indicators.

Key features:

* **Data source:** The script downloads historic price data from
  ``https://stooq.pl`` for a given list of ticker symbols.  This API
  returns comma‑separated values for U.S. shares when suffixed with
  ``.us``.

* **Technical indicators:** For each ticker the script calculates:
  - A short and long simple moving average (SMA) to detect trend
    direction.
  - The relative strength index (RSI) to measure momentum and
    overbought/oversold conditions.
  - The moving average convergence divergence (MACD) and its signal
    line to gauge trend momentum.

* **Signal generation:** A combined signal is produced when the short
  SMA is above the long SMA **and** the RSI is above 50 **and** the
  MACD is above its signal line.  A sell signal is generated when the
  opposite is true.  Positions are shifted forward by one day to
  simulate execution the day after a signal appears.

* **Performance measurement:** The script tracks the cumulative return
  of a buy‑and‑hold strategy versus the cumulative return of the
  indicator‑driven strategy for each ticker.  It provides a simple
  ranking by final strategy return percentage.

.. warning::

    This code is provided for **educational purposes only**.  Day
    trading carries significant financial risk and most individual day
    traders lose money【29948981333204†L654-L694】.  Quantitative
    strategies can stop working when market conditions change or once
    they become widely used【440896089455146†L268-L279】.  You should
    consult a qualified financial advisor before making real trades.

Usage example::

    from quant_trading_project import run_analysis, create_summary

    tickers = ['PFE', 'JNJ', 'MRNA', 'LLY', 'AZN', 'BMY']
    results = run_analysis(tickers, '2024-12-01', '2025-07-30')
    summary = create_summary(results)
    print(summary)

You can extend this module by plugging the ``run_analysis`` and
``create_summary`` functions into a web framework (e.g. Flask) or a
scheduled job (e.g. cron or GitHub Actions) to run the analysis
automatically.
"""

from __future__ import annotations

import datetime as _dt
import io as _io
import typing as _t

import pandas as _pd
import requests as _requests


def fetch_stooq(ticker: str) -> _pd.DataFrame:
    """Download daily historical price data for a U.S. stock from Stooq.

    Stooq returns data for U.S. equities when the symbol is suffixed
    with ``.us``.  Column names are in Polish, so they are converted
    to English for convenience.

    Parameters
    ----------
    ticker: str
        The stock ticker symbol (e.g. ``'PFE'``).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``Date``, ``Open``, ``High``, ``Low``,
        ``Close`` and ``Volume``.
    """
    url = f"https://stooq.pl/q/d/l/?s={ticker.lower()}.us&i=d"
    response = _requests.get(url)
    response.raise_for_status()
    csv_data = response.text
    df = _pd.read_csv(_io.StringIO(csv_data))
    # Rename columns from Polish to English.
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = _pd.to_datetime(df['Date'])
    return df


def calculate_rsi(series: _pd.Series, period: int = 14) -> _pd.Series:
    """Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    series : pandas.Series
        Time series of prices.
    period : int, optional
        Look‑back period for the RSI, by default 14.

    Returns
    -------
    pandas.Series
        The RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series: _pd.Series,
                   short: int = 12,
                   long: int = 26,
                   signal: int = 9) -> tuple[_pd.Series, _pd.Series]:
    """Calculate the Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    series : pandas.Series
        Time series of prices.
    short : int, optional
        Span for the short EMA, by default 12.
    long : int, optional
        Span for the long EMA, by default 26.
    signal : int, optional
        Span for the signal line, by default 9.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        A tuple containing the MACD and the MACD signal line.
    """
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def generate_signals(df: _pd.DataFrame,
                     short_window: int = 20,
                     long_window: int = 50) -> _pd.DataFrame:
    """Add technical indicators and trading signals to a DataFrame.

    This function computes a short and long SMA, the RSI, and the MACD
    and uses them to generate trading signals.  A long (buy) signal is
    generated when all of the following are true:

    * ``SMA_short > SMA_long``
    * ``RSI > 50``
    * ``MACD > MACD_signal``

    A short (sell) signal is generated when the opposite conditions
    are true.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least a ``Close`` column.
    short_window : int, optional
        Look‑back period for the short SMA, by default 20.
    long_window : int, optional
        Look‑back period for the long SMA, by default 50.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with added indicator and signal columns.
    """
    df = df.copy()
    # Calculate moving averages.
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    # Calculate RSI.
    df['RSI'] = calculate_rsi(df['Close'])
    # Calculate MACD and its signal line.
    macd, macd_signal = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal

    # Generate signals.
    df['Signal'] = 0
    buy_condition = (df['SMA_short'] > df['SMA_long']) & (df['RSI'] > 50) & (df['MACD'] > df['MACD_signal'])
    sell_condition = (df['SMA_short'] < df['SMA_long']) & (df['RSI'] < 50) & (df['MACD'] < df['MACD_signal'])
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1

    # Shift positions forward to simulate acting on next day’s open.
    df['Position'] = df['Signal'].shift(1).fillna(0)
    # Calculate daily and strategy returns.
    df['Daily Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Position'] * df['Daily Return']
    # Cumulative returns (starting from 1).
    df['Cumulative Market Return'] = (1 + df['Daily Return']).cumprod()
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
    return df


def run_analysis(tickers: list[str], start_date: str, end_date: str,
                 short_window: int = 20, long_window: int = 50) -> dict[str, _pd.DataFrame]:
    """Fetch data, calculate indicators and run backtest for multiple tickers.

    Parameters
    ----------
    tickers : list[str]
        A list of stock ticker symbols.
    start_date : str
        Start date in ``YYYY-MM-DD`` format.
    end_date : str
        End date in ``YYYY-MM-DD`` format.
    short_window : int, optional
        Look‑back period for the short SMA, by default 20.
    long_window : int, optional
        Look‑back period for the long SMA, by default 50.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary mapping each ticker to its enriched DataFrame.
    """
    start_dt = _pd.to_datetime(start_date)
    end_dt = _pd.to_datetime(end_date)
    results: dict[str, _pd.DataFrame] = {}
    for ticker in tickers:
        data = fetch_stooq(ticker)
        data = data[(data['Date'] >= start_dt) & (data['Date'] <= end_dt)]
        if data.empty:
            continue
        enriched = generate_signals(data, short_window=short_window, long_window=long_window)
        results[ticker] = enriched
    return results


def create_summary(results: dict[str, _pd.DataFrame]) -> _pd.DataFrame:
    """Summarize strategy performance across multiple tickers.

    Parameters
    ----------
    results : dict[str, pandas.DataFrame]
        Output from ``run_analysis``.

    Returns
    -------
    pandas.DataFrame
        A summary table containing each ticker and its final market
        return, final strategy return and strategy return percentage.
    """
    summary_rows: list[dict[str, _t.Any]] = []
    for ticker, df in results.items():
        if df.empty:
            continue
        final_market = df['Cumulative Market Return'].iloc[-1]
        final_strategy = df['Cumulative Strategy Return'].iloc[-1]
        summary_rows.append({
            'Ticker': ticker,
            'FinalMarketReturn': final_market,
            'FinalStrategyReturn': final_strategy,
            'StrategyReturnPct': (final_strategy - 1) * 100
        })
    summary = _pd.DataFrame(summary_rows)
    if summary.empty:
        return summary
    summary['StrategyReturnPct'] = summary['StrategyReturnPct'].round(2)
    return summary.sort_values(by='StrategyReturnPct', ascending=False).reset_index(drop=True)


if __name__ == '__main__':  # pragma: no cover
    # Example usage when running this script directly.  Fetch the
    # specified tickers and print a performance summary.
    tickers = ['PFE', 'JNJ', 'MRNA', 'LLY', 'AZN', 'BMY']
    start_date = '2024-12-01'
    end_date = '2025-07-30'
    print(f"Running analysis for {tickers} from {start_date} to {end_date}...")
    result_data = run_analysis(tickers, start_date, end_date)
    summary_table = create_summary(result_data)
    print(summary_table)
