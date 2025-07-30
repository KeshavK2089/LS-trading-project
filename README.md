LS Stock Tracker returning purchasing power

# ABOUT THE RESULTS
#
# Ticker:
#   The stock symbol being analyzed (e.g., PFE, JNJ, etc.).
#
# FinalMarketReturn:
#   The cumulative return from simply buying the stock at the start date
#   and holding it until the end date.  Values > 1 mean the stock gained
#   overall; values < 1 mean it lost value during the period.
#
# FinalStrategyReturn:
#   The cumulative return from the trading strategy.  It starts at 1 and
#   multiplies each day’s signal-driven return.  Values above 1 indicate
#   the strategy made money; values below 1 indicate a loss.
#
# StrategyReturnPct:
#   The percentage change from the starting capital, calculated as
#   (FinalStrategyReturn – 1) × 100.  Positive percentages mean the
#   strategy was profitable; negative percentages mean it lost money.
#
# Interpretation:
#   - Compare FinalStrategyReturn to FinalMarketReturn: if the strategy
#     return is higher, the model outperformed buy-and-hold for that stock.
#     If it’s lower, the strategy underperformed.
#   - Look for positive StrategyReturnPct values as potential winners.
#   - Remember these results are historical; past performance does not
#     guarantee future results.  Day trading is risky and this project is
#     for educational use only.


