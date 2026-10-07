# FitCheck

FitCheck is a Streamlit portfolio analytics platform for a student investment fund. It provides the first three portfolio workflow blocks: current holdings, candidate funding, and historical optimization.

## What It Does

- Starts with a single `Add portfolio` row and collects the name in a compact dialog.
- Creates the portfolio row and its first empty holding row after the name is submitted.
- Keeps the portfolio name editable directly in its table row.
- Keeps new holdings as draft rows until company, weight, and purchase date are complete and `Add to Portfolio` is selected.
- Lets users edit committed holdings and refresh their company, automatic ticker and sector classification, weight, purchase date, and purchase-date close.
- Hides the add-security control at 100% and restores it after an existing weight is lowered.
- Applies sector grouping and alphabetical sorting only after the holding is added.
- Provides a plus/minus control to expand or collapse the individual security rows.
- Starts with an empty portfolio and adds holdings from a single plus control below the table.
- Searches Yahoo Finance for matching company names and fills a read-only ticker automatically.
- Captures company name, ticker, portfolio weight, and purchase date.
- Classifies holdings from Yahoo Finance sector metadata and automatically groups them as Materials, FIGs, TMTH, Consumers, Infrastructure, or Industrials.
- Sorts companies alphabetically within each sector group.
- Keeps each security weight between 1% and 6%.
- Includes permanent CAD Cash and USD Cash rows in the current portfolio total.
- Flags duplicate tickers.
- Pulls the closing price for each selected purchase date automatically from Yahoo Finance with `yfinance`.
- Uses exchange calendars to disable weekends, market holidays, and future dates.
- Disables dates before the security's earliest available Yahoo Finance price, which serves as the listing-date cutoff when an explicit IPO date is unavailable.
- Supports both arrow-based month navigation and direct month/year selection for older dates.
- If today is selected before the market closes, shows the most recent completed close and its actual date.
- Supports Yahoo Finance ticker formats, including Canadian tickers such as `RY.TO`, `SHOP.TO`, and `CASH.TO`.

## Run It

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Then open <http://127.0.0.1:8501> in your browser.

The original standalone interface remains available locally with `python3 app.py`.

## Deploy On Streamlit Community Cloud

1. Fork or import this repository into your GitHub account.
2. In Streamlit Community Cloud, choose **Create app** and select the repository.
3. Set the main file path to `app.py`.
4. Deploy. Streamlit Cloud installs the Python packages from `requirements.txt` automatically.

No application secrets are required. Yahoo Finance data is retrieved at runtime through `yfinance`.

## Input Columns

- `Company Name`: searchable security name supplied by Yahoo Finance.
- `Ticker`: automatically filled Yahoo Finance symbol; Canadian listings retain suffixes such as `.TO`.
- `Portfolio Weight (%)`: numeric allocation with a fixed, non-editable percent suffix.
- `Purchase Date`: selected from valid trading sessions for the security's exchange.
- `Price`: dollar-formatted Yahoo Finance close for the selected purchase date, or the dated previous close when today's session is still open.

## Output Modes

- Expanded: the portfolio row and each editable holding row are visible.
- Condensed: only the portfolio row is visible; its purchase date remains blank.

## Block 2: Candidate Stock

- Keeps a potential new security separate from current portfolio holdings.
- Uses Yahoo Finance company search and a read-only ticker.
- Determines the candidate weight automatically from the selected objective, historical window, and funding constraints.
- Supports funding from available cash or automatic reductions to current holdings.
- Lets the user choose CAD Cash or USD Cash as the primary cash source and uses the other currency as a fallback when needed.
- Lets the user reduce all eligible portfolio positions or only holdings in the candidate's sector group.
- Provides checkboxes for manually limiting which eligible holdings may be reduced.
- Keeps every holding visible in the funding section and marks unavailable rows as `Optimizer Ineligible`.
- Includes `Optimize Reduction Automatically`, which chooses reduction amounts for the active return, volatility, or Sharpe objective.
- Automatically allocates reductions proportionally based on each eligible holding's reducible capacity, then draws any remainder from cash.
- Keeps reductions from taking an existing security below a 1% weight.
- Shows the portion funded by each reduced stock, CAD Cash, USD Cash, and any unfunded shortfall.

## Block 3: Portfolio Metrics

- Requires current holdings plus Cash to equal exactly 100.00% before calculating.
- Calculates annualized return, volatility, and Sharpe ratio for 6M, 1Y, 3Y, and 5Y windows using adjusted Yahoo Finance closes.
- Lets the user maximize return, minimize volatility, or maximize Sharpe ratio.
- Searches candidate weights from 1.00% through 6.00% in 0.05% increments.
- Respects the 1% minimum for reduced holdings and the available Cash balance.
- Presents Return, Volatility, and Sharpe Ratio as before/after/difference metric blocks for the selected historical window.
- Provides a 6M, 1Y, 3Y, and 5Y historical-window selector for the main metric blocks.
- Includes an objective-aware sensitivity table comparing optimized candidate weight, return, volatility, and Sharpe ratio across all four windows.
- Labels the recommendation `Very Sensitive` when optimized weights span at least 2.00 percentage points across available windows; otherwise it labels it `Not Sensitive`.
- Automatically clears and recalculates metrics, optimizer results, funding details, and sensitivity output whenever portfolio, candidate, cash, or funding inputs change.
- Discards stale in-flight analysis responses and blocks results whenever the current portfolio does not total exactly 100.00%.
- Reports the optimized candidate weight, named stock reductions, and CAD/USD funding split.
- Uses a 0% risk-free rate for the displayed Sharpe ratio.
