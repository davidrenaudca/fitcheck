from __future__ import annotations

import csv
import io
import json
import math
import numpy as np
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import yfinance as yf
except ImportError:  # Keeps the interface usable before dependencies are installed.
    yf = None

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None


HOST = "127.0.0.1"
PORT = 8501

SECTOR_GROUPS = {
    "basic materials": "Materials",
    "materials": "Materials",
    "financial services": "FIGs",
    "financials": "FIGs",
    "technology": "TMTH",
    "information technology": "TMTH",
    "communication services": "TMTH",
    "consumer cyclical": "Consumers",
    "consumer defensive": "Consumers",
    "consumer discretionary": "Consumers",
    "consumer staples": "Consumers",
    "real estate": "Infrastructure",
    "utilities": "Infrastructure",
    "industrials": "Industrials",
}


def parse_weight(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0


def clean_holding(row: dict) -> dict:
    return {
        "company_name": str(row.get("company_name", "")).strip(),
        "ticker": str(row.get("ticker", "")).strip().upper(),
        "gics_sector": str(row.get("gics_sector", "")).strip(),
        "sector_group": str(row.get("sector_group", "")).strip(),
        "weight": parse_weight(row.get("weight")),
        "purchase_date": str(row.get("purchase_date", "")).strip(),
    }


@lru_cache(maxsize=256)
def classify_security(ticker: str) -> dict:
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    info = yf.Ticker(ticker.strip().upper()).get_info()
    sector = str(info.get("sector") or info.get("sectorDisp") or "").strip()
    industry = str(info.get("industry") or info.get("industryDisp") or "").strip()
    normalized_sector = sector.casefold()

    if "reit" in industry.casefold():
        group = "Infrastructure"
    else:
        group = SECTOR_GROUPS.get(normalized_sector, "Unclassified")

    return {
        "ticker": ticker.strip().upper(),
        "gics_sector": sector or "Unknown",
        "industry": industry,
        "sector_group": group,
    }


def clean_holdings(rows: list[dict]) -> list[dict]:
    return [holding for holding in map(clean_holding, rows) if holding["ticker"]]


def validate_holdings(holdings: list[dict]) -> list[str]:
    errors = []

    if not holdings:
        errors.append("Enter at least one holding before preparing the portfolio.")

    if any(holding["weight"] <= 0 for holding in holdings):
        errors.append("Every holding must have a portfolio weight greater than 0%.")

    tickers = [holding["ticker"] for holding in holdings]
    duplicate_tickers = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
    if duplicate_tickers:
        errors.append(f"Duplicate tickers found: {', '.join(duplicate_tickers)}.")

    total_weight = sum(holding["weight"] for holding in holdings)
    if holdings and abs(total_weight - 100) > 0.5:
        errors.append(
            f"Portfolio weights currently total {total_weight:.2f}%. They should total 100%."
        )

    if any(not holding["purchase_date"] for holding in holdings):
        errors.append("Every holding needs a purchase date.")

    return errors


def calendar_name_for_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker.endswith(".V"):
        return "TSXV"
    if ticker.endswith((".TO", ".NE", ".CN")):
        return "TSX"
    return "NYSE"


def market_schedule(ticker: str, start: date, end: date):
    if mcal is None:
        raise RuntimeError("pandas-market-calendars is not installed")
    calendar = mcal.get_calendar(calendar_name_for_ticker(ticker))
    return calendar.schedule(start_date=start.isoformat(), end_date=end.isoformat())


@lru_cache(maxsize=256)
def earliest_available_price_date(ticker: str) -> date:
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    history = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    closes = history.get("Close")
    if closes is None or closes.dropna().empty:
        raise ValueError("Yahoo Finance did not return any historical prices.")
    return closes.dropna().index[0].date()


@lru_cache(maxsize=256)
def trading_days_for_month(ticker: str, year: int, month: int) -> list[str]:
    month_start = date(year, month, 1)
    if month == 12:
        month_end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year, month + 1, 1) - timedelta(days=1)

    cutoff = earliest_available_price_date(ticker)
    start = max(month_start, cutoff)
    end = min(month_end, date.today())
    if end < start:
        return []

    schedule = market_schedule(ticker, start, end)
    return [session.date().isoformat() for session in schedule.index]


def purchase_date_close(ticker: str, selected_date: str) -> dict:
    ticker = ticker.strip().upper()
    target = date.fromisoformat(selected_date)
    if target > date.today():
        raise ValueError("Purchase date cannot be in the future.")
    cutoff = earliest_available_price_date(ticker)
    if target < cutoff:
        raise ValueError(
            f"Purchase date cannot be before the earliest available price date ({cutoff.isoformat()})."
        )

    schedule = market_schedule(ticker, target - timedelta(days=14), target)
    sessions = [(session.date(), row) for session, row in schedule.iterrows()]
    matching = [(session, row) for session, row in sessions if session == target]
    if not matching:
        raise ValueError("The selected date is not a trading day for this security.")

    effective_date = target
    used_previous_close = False
    if target == date.today():
        market_close = matching[0][1]["market_close"].to_pydatetime()
        if datetime.now(timezone.utc) < market_close.astimezone(timezone.utc):
            prior_sessions = [session for session, _ in sessions if session < target]
            if not prior_sessions:
                raise ValueError("No prior market close is available.")
            effective_date = prior_sessions[-1]
            used_previous_close = True

    history = yf.Ticker(ticker).history(
        start=(effective_date - timedelta(days=7)).isoformat(),
        end=(effective_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    closes = history.get("Close")
    if closes is None:
        raise ValueError("Yahoo Finance did not return closing prices.")
    closes = closes.dropna()
    matching_closes = [
        float(value)
        for timestamp, value in closes.items()
        if timestamp.date() == effective_date
    ]

    if not matching_closes and target == date.today():
        prior = [(timestamp.date(), float(value)) for timestamp, value in closes.items()]
        if prior:
            effective_date, close = prior[-1]
            used_previous_close = True
        else:
            raise ValueError("No completed market close is available yet.")
    elif not matching_closes:
        raise ValueError("Yahoo Finance did not return a close for the selected date.")
    else:
        close = matching_closes[-1]

    return {
        "ticker": ticker,
        "selected_date": selected_date,
        "close": round(close, 2),
        "price_date": effective_date.isoformat(),
        "used_previous_close": used_previous_close,
    }


def fetch_price_data(holdings: list[dict]) -> tuple[dict[str, dict], list[str]]:
    prices = {}
    warnings = []

    if yf is None:
        return prices, [
            "yfinance is not installed yet. Run `pip install -r requirements.txt` to pull Yahoo Finance prices."
        ]

    for holding in holdings:
        ticker = holding["ticker"]
        try:
            result = purchase_date_close(ticker, holding["purchase_date"])
        except Exception as error:
            warnings.append(f"{ticker}: closing price lookup failed ({error}).")
            continue
        prices[ticker] = {
            "purchase_close": result["close"],
            "purchase_close_date": result["price_date"],
            "used_previous_close": result["used_previous_close"],
        }

    return prices, warnings


@lru_cache(maxsize=128)
def search_securities(query: str) -> list[dict]:
    query = query.strip()
    if len(query) < 2 or yf is None:
        return []

    try:
        quotes = yf.Search(
            query,
            max_results=8,
            news_count=0,
            lists_count=0,
            include_cb=False,
            recommended=0,
            timeout=8,
        ).quotes
    except Exception:
        return []

    results = []
    for quote in quotes:
        symbol = str(quote.get("symbol", "")).strip().upper()
        name = str(quote.get("longname") or quote.get("shortname") or "").strip()
        if not symbol or not name:
            continue
        results.append(
            {
                "company_name": name,
                "ticker": symbol,
                "exchange": str(quote.get("exchDisp") or quote.get("exchange") or "").strip(),
                "type": str(quote.get("quoteType") or quote.get("typeDisp") or "").strip(),
            }
        )
    return results


def build_output(
    holdings: list[dict], portfolio_name: str, condense: bool, prices: dict[str, dict]
) -> list[dict]:
    portfolio_name = portfolio_name.strip() or "Student Investment Fund"

    if condense:
        holdings_detail = ", ".join(
            f"{holding['ticker']} ({holding['weight']:.2f}%)" for holding in holdings
        )
        price_detail = ", ".join(
            f"{ticker}: {price['purchase_close']:.2f} on {price['purchase_close_date']}"
            for ticker, price in prices.items()
        )
        return [
            {
                "Portfolio Name": portfolio_name,
                "Company Name": "Combined portfolio",
                "Ticker": "COMBINED",
                "Portfolio Weight (%)": "100.00",
                "Purchase Date": "",
                "Underlying Holdings": holdings_detail,
                "Purchase Date Detail": ", ".join(
                    f"{holding['ticker']}: {holding['purchase_date']}" for holding in holdings
                ),
                "Purchase Close Detail": price_detail or "Not available",
            }
        ]

    rows = []
    for holding in holdings:
        price = prices.get(holding["ticker"], {})
        purchase_close = price.get("purchase_close")
        rows.append(
            {
                "Portfolio Name": portfolio_name,
                "Company Name": holding["company_name"],
                "Ticker": holding["ticker"],
                "GICS Sector": holding["gics_sector"],
                "Sector Group": holding["sector_group"],
                "Portfolio Weight (%)": f"{holding['weight']:.2f}",
                "Purchase Date": holding["purchase_date"],
                "Purchase Close": f"{purchase_close:.2f}" if purchase_close else "",
                "Purchase Close Date": price.get("purchase_close_date", ""),
                "Used Previous Close": price.get("used_previous_close", False),
            }
        )

    return rows


def rows_to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def prepare_portfolio(payload: dict) -> dict:
    holdings = clean_holdings(payload.get("holdings", []))
    errors = validate_holdings(holdings)
    prepared_rows = []
    price_warnings = []

    if not errors:
        prices, price_warnings = fetch_price_data(holdings)
        prepared_rows = build_output(
            holdings,
            payload.get("portfolio_name", ""),
            bool(payload.get("condense")),
            prices,
        )

    return {
        "errors": errors,
        "warnings": price_warnings,
        "holdings_count": len(holdings),
        "total_weight": round(sum(holding["weight"] for holding in holdings), 2),
        "prepared_rows": prepared_rows,
        "csv": rows_to_csv(prepared_rows),
    }


ANALYSIS_WINDOWS = {
    "6M": 183,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
}


def annualized_metrics(daily_returns) -> dict:
    observations = len(daily_returns)
    if observations < 20:
        raise ValueError("Not enough overlapping price history is available.")

    compounded = float((1 + daily_returns).prod())
    annual_return = compounded ** (252 / observations) - 1 if compounded > 0 else -1
    annual_volatility = float(daily_returns.std(ddof=1)) * math.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
    return {
        "return": round(annual_return * 100, 2),
        "volatility": round(annual_volatility * 100, 2),
        "sharpe": round(sharpe, 3),
    }


def candidate_funding_plan(
    holdings: list[dict],
    cash_weights: dict[str, float],
    candidate_weight: float,
    funding_method: str,
    reduction_scope: str,
    candidate_group: str,
    preferred_cash: str,
    allowed_tickers: set[str] | None = None,
    reduction_allocations: dict[str, float] | None = None,
) -> dict | None:
    post_weights = {holding["ticker"]: holding["weight"] for holding in holdings}
    applied_reductions = {}

    if funding_method == "reduce":
        reduction_caps = {
            holding["ticker"]: max(0, holding["weight"] - 1)
            for holding in holdings
            if reduction_scope == "all" or holding["sector_group"] == candidate_group
            if allowed_tickers is None or holding["ticker"] in allowed_tickers
        }

        if reduction_allocations is None:
            available_reductions = sum(reduction_caps.values())
            reduction_total = min(candidate_weight, available_reductions)
            if available_reductions > 0 and reduction_total > 0:
                for ticker, cap in reduction_caps.items():
                    reduction = reduction_total * (cap / available_reductions)
                    if reduction > 0:
                        applied_reductions[ticker] = reduction
                        post_weights[ticker] -= reduction
        else:
            for ticker, amount in reduction_allocations.items():
                reduction = min(max(0, amount), reduction_caps.get(ticker, 0))
                if reduction > 0:
                    applied_reductions[ticker] = reduction
                    post_weights[ticker] -= reduction
            reduction_total = min(candidate_weight, sum(applied_reductions.values()))
    else:
        reduction_total = 0

    remaining = candidate_weight - reduction_total
    cash_used = {"cad": 0.0, "usd": 0.0}
    cash_order = [preferred_cash, "usd" if preferred_cash == "cad" else "cad"]
    for currency in cash_order:
        amount = min(remaining, cash_weights[currency])
        cash_used[currency] = amount
        remaining -= amount

    if remaining > 1e-9:
        return None
    if any(weight < 1 - 1e-9 or weight > 6 + 1e-9 for weight in post_weights.values()):
        return None

    return {
        "security_weights": post_weights,
        "cash_weights": {
            currency: cash_weights[currency] - cash_used[currency]
            for currency in ("cad", "usd")
        },
        "cash_used": cash_used,
        "reductions": applied_reductions,
    }


def optimize_reduction_allocations(
    holdings: list[dict],
    candidate_ticker: str,
    candidate_weight: float,
    base_plan: dict,
    daily_returns,
    optimize_metric: str,
    eligible_tickers: set[str],
) -> dict[str, float]:
    target = sum(base_plan["reductions"].values())
    if target <= 1e-9:
        return {}

    capacities = {
        holding["ticker"]: max(0, holding["weight"] - 1)
        for holding in holdings
        if holding["ticker"] in eligible_tickers
    }
    allocations = {ticker: 0.0 for ticker in capacities}
    tickers = [holding["ticker"] for holding in holdings] + [candidate_ticker]
    ticker_indexes = {ticker: index for index, ticker in enumerate(tickers)}
    portfolio_weights = np.array(
        [holding["weight"] for holding in holdings] + [candidate_weight],
        dtype=float,
    )
    annual_means = daily_returns[tickers].mean().to_numpy(dtype=float) * 252
    annual_covariance = daily_returns[tickers].cov().to_numpy(dtype=float) * 252

    def score(weights) -> float:
        fractions = weights / 100
        expected_return = float(fractions @ annual_means)
        variance = float(fractions @ annual_covariance @ fractions)
        volatility = math.sqrt(max(variance, 0))
        if optimize_metric == "return":
            return expected_return
        if optimize_metric == "volatility":
            return volatility
        return expected_return / volatility if volatility > 0 else -math.inf

    remaining = target
    while remaining > 1e-9:
        step = min(0.05, remaining)
        choices = []
        for ticker, capacity in capacities.items():
            available = capacity - allocations[ticker]
            amount = min(step, available)
            if amount <= 1e-9:
                continue
            trial_weights = portfolio_weights.copy()
            trial_weights[ticker_indexes[ticker]] -= amount
            choices.append((score(trial_weights), ticker, amount))
        if not choices:
            break
        _, selected, amount = (
            min(choices, key=lambda item: (item[0], item[1]))
            if optimize_metric == "volatility"
            else max(choices, key=lambda item: (item[0], item[1]))
        )
        allocations[selected] += amount
        portfolio_weights[ticker_indexes[selected]] -= amount
        remaining -= amount

    return {
        ticker: amount
        for ticker, amount in allocations.items()
        if amount > 1e-9
    }


def load_return_history(tickers: list[str]):
    start = date.today() - timedelta(days=ANALYSIS_WINDOWS["5Y"] + 14)
    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(date.today() + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise ValueError("Yahoo Finance did not return historical prices.")

    closes = data["Close"] if "Close" in data else data
    if len(tickers) == 1:
        closes = closes.to_frame(name=tickers[0]) if getattr(closes, "ndim", 1) == 1 else closes
    closes.columns = [str(column).upper() for column in closes.columns]
    missing = [ticker for ticker in tickers if ticker not in closes.columns or closes[ticker].dropna().empty]
    if missing:
        raise ValueError(f"No historical prices found for: {', '.join(missing)}.")
    return closes[tickers].pct_change(fill_method=None)


def analyze_portfolio(payload: dict) -> dict:
    holdings = clean_holdings(payload.get("holdings", []))
    cash_weights = {
        "cad": parse_weight(payload.get("cad_cash_weight", payload.get("cash_weight"))),
        "usd": parse_weight(payload.get("usd_cash_weight")),
    }
    candidate = payload.get("candidate") or {}
    candidate_ticker = str(candidate.get("ticker", "")).strip().upper()
    funding_method = str(candidate.get("funding_method", "cash"))
    reduction_scope = str(candidate.get("reduction_scope", "all")).lower()
    optimize_reductions = bool(candidate.get("optimize_reductions", True))
    selected_reductions = {
        str(ticker).strip().upper()
        for ticker in candidate.get("selected_reductions", [])
        if str(ticker).strip()
    }
    preferred_cash = str(candidate.get("cash_source", "cad")).lower()
    candidate_group = str(candidate.get("sector_group", ""))
    optimize_metric = str(payload.get("optimize_metric", "sharpe"))
    errors = []
    warnings = []

    total_weight = sum(holding["weight"] for holding in holdings) + sum(cash_weights.values())
    invalid_cash = [currency.upper() for currency, weight in cash_weights.items() if weight < 0 or weight > 100]
    if invalid_cash:
        errors.append(f"{' and '.join(invalid_cash)} Cash weights must be from 0.00% to 100.00%.")
    if abs(total_weight - 100) > 0.005:
        errors.append(
            f"Current portfolio weights, including CAD Cash and USD Cash, must equal exactly 100.00%. Current total: {total_weight:.2f}%."
        )
    if not holdings:
        errors.append("Add at least one security to the current portfolio.")
    invalid_holdings = [
        holding["ticker"] for holding in holdings if holding["weight"] < 1 or holding["weight"] > 6
    ]
    if invalid_holdings:
        errors.append(
            f"Every security must have a weight from 1.00% to 6.00%. Check: {', '.join(invalid_holdings)}."
        )
    if not candidate_ticker:
        errors.append("Select a candidate stock.")
    if candidate_ticker in {holding["ticker"] for holding in holdings}:
        errors.append("The candidate stock is already held in the current portfolio.")
    if funding_method not in {"cash", "reduce"}:
        errors.append("Choose a valid funding source.")
    if reduction_scope not in {"all", "sector"}:
        errors.append("Choose all eligible positions or the candidate sector for reductions.")
    if funding_method == "reduce" and reduction_scope == "sector" and not candidate_group:
        errors.append("Wait for the candidate sector classification before calculating reductions.")
    if preferred_cash not in {"cad", "usd"}:
        errors.append("Choose CAD Cash or USD Cash as the primary cash source.")
    if optimize_metric not in {"return", "volatility", "sharpe"}:
        errors.append("Choose return, volatility, or Sharpe ratio for optimization.")

    scope_eligible = {
        holding["ticker"]
        for holding in holdings
        if reduction_scope == "all" or holding["sector_group"] == candidate_group
    }
    reducible_eligible = {
        holding["ticker"]
        for holding in holdings
        if holding["ticker"] in scope_eligible and holding["weight"] > 1
    }
    invalid_selected = selected_reductions - reducible_eligible
    if funding_method == "reduce" and invalid_selected:
        errors.append("Selected reductions must be above 1.00% and eligible under the chosen reduction scope.")
    eligible_tickers = reducible_eligible if optimize_reductions else selected_reductions

    if errors:
        return {"errors": errors, "warnings": warnings, "results": []}

    feasible_plans = []
    for basis_points in range(100, 601, 5):
        weight = basis_points / 100
        plan = candidate_funding_plan(
            holdings, cash_weights, weight, funding_method,
            reduction_scope, candidate_group, preferred_cash, eligible_tickers
        )
        if plan is not None:
            feasible_plans.append((weight, plan))
    if not feasible_plans:
        return {
            "errors": ["No candidate weight from 1.00% to 6.00% can be funded under the current constraints."],
            "warnings": warnings,
            "results": [],
        }

    tickers = [holding["ticker"] for holding in holdings] + [candidate_ticker]
    returns = load_return_history(tickers)
    results = []

    for label, days in ANALYSIS_WINDOWS.items():
        window_start = date.today() - timedelta(days=days)
        window_returns = returns.loc[returns.index.date >= window_start].dropna()
        if len(window_returns) < 20:
            warnings.append(f"{label}: not enough overlapping history to calculate metrics.")
            continue

        current_series = sum(
            window_returns[holding["ticker"]] * (holding["weight"] / 100)
            for holding in holdings
        )
        current_metrics = annualized_metrics(current_series)
        candidates = []

        for candidate_weight, base_plan in feasible_plans:
            plan = base_plan
            if funding_method == "reduce" and optimize_reductions:
                allocations = optimize_reduction_allocations(
                    holdings,
                    candidate_ticker,
                    candidate_weight,
                    base_plan,
                    window_returns,
                    optimize_metric,
                    eligible_tickers,
                )
                plan = candidate_funding_plan(
                    holdings,
                    cash_weights,
                    candidate_weight,
                    funding_method,
                    reduction_scope,
                    candidate_group,
                    preferred_cash,
                    eligible_tickers,
                    allocations,
                )
                if plan is None:
                    continue
            optimized_series = sum(
                window_returns[ticker] * (weight / 100)
                for ticker, weight in plan["security_weights"].items()
            )
            optimized_series += window_returns[candidate_ticker] * (candidate_weight / 100)
            metrics = annualized_metrics(optimized_series)
            candidates.append((candidate_weight, plan, metrics))

        if optimize_metric == "volatility":
            best = min(candidates, key=lambda item: item[2]["volatility"])
        else:
            best = max(candidates, key=lambda item: item[2][optimize_metric])

        best_weight, best_plan, best_metrics = best
        holding_names = {holding["ticker"]: holding["company_name"] for holding in holdings}
        results.append(
            {
                "window": label,
                "observations": len(window_returns),
                "current": current_metrics,
                "optimized": best_metrics,
                "candidate_weight": round(best_weight, 2),
                "reduction_scope": reduction_scope,
                "optimize_reductions": optimize_reductions,
                "cash_used": {
                    currency: round(weight, 2)
                    for currency, weight in best_plan["cash_used"].items()
                },
                "reductions": [
                    {
                        "ticker": ticker,
                        "company_name": holding_names.get(ticker, ticker),
                        "weight": round(weight, 2),
                    }
                    for ticker, weight in best_plan["reductions"].items()
                ],
            }
        )

    return {
        "errors": [],
        "warnings": warnings,
        "optimize_metric": optimize_metric,
        "results": results,
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FitCheck</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #171a1f;
      --muted: #717784;
      --line: #dfe3e8;
      --soft: #f5f7f8;
      --portfolio: #eef2f4;
      --accent: #176b61;
      --danger: #a43a3a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #fff;
    }

    .app-header {
      min-height: 56px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }

    .app-header-inner {
      display: flex;
      align-items: center;
      max-width: 1200px;
      min-height: 56px;
      margin: 0 auto;
      padding: 0 20px;
    }

    .app-brand {
      color: #7b263d;
      font-size: 20px;
      font-weight: 750;
      letter-spacing: 0;
    }

    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 30px 20px 56px;
    }

    .table-shell {
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: visible;
      box-shadow: 0 1px 3px rgba(23, 26, 31, .05);
    }

    table {
      width: 100%;
      min-width: 850px;
      border-collapse: collapse;
      table-layout: fixed;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: middle;
    }

    th {
      height: 42px;
      background: var(--soft);
      color: #4e5560;
      font-size: 12px;
      font-weight: 650;
    }

    th:nth-child(1) { width: 29%; }
    th:nth-child(2) { width: 14%; }
    th:nth-child(3) { width: 15%; }
    th:nth-child(4) { width: 18%; }
    th:nth-child(5) { width: 14%; }
    th:nth-child(6) { width: 10%; }

    tbody tr:last-child td,
    tfoot tr:last-child td { border-bottom: 0; }

    .portfolio-row td {
      height: 58px;
      background: var(--portfolio);
      font-weight: 650;
    }

    .sector-header td {
      height: 34px;
      padding: 7px 12px 7px 52px;
      background: #f7f8f9;
      color: #59616b;
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
    }

    .holding-row input[readonly],
    .holding-row select:disabled {
      border-color: transparent;
      background: transparent;
      box-shadow: none;
      cursor: default;
      color: var(--ink);
      opacity: 1;
    }

    .draft-actions-row td {
      padding: 9px 12px;
      background: #fbfcfc;
    }

    .draft-actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 12px;
    }

    .draft-validation {
      color: var(--muted);
      font-size: 12px;
    }

    .commit-holding-button {
      min-height: 34px;
      border: 1px solid var(--accent);
      border-radius: 5px;
      padding: 7px 12px;
      background: var(--accent);
      color: white;
      font-size: 12px;
      font-weight: 700;
    }

    .commit-holding-button:disabled {
      border-color: #d6dade;
      background: #e9ecee;
      color: #9ba1a8;
      cursor: default;
    }

    .empty-portfolio-row td {
      height: 64px;
      padding: 0;
      text-align: center;
      background: #fbfcfc;
    }

    .add-portfolio-trigger {
      width: 100%;
      height: 63px;
      border: 0;
      background: transparent;
      color: #525a64;
      font-weight: 650;
    }

    .add-portfolio-trigger:hover { background: #f5f7f8; color: var(--ink); }
    .add-portfolio-trigger:focus-visible {
      outline: 2px solid rgba(123, 38, 61, .28);
      outline-offset: -2px;
    }

    .portfolio-inline-form {
      width: min(420px, calc(100% - 32px));
      margin: 0 auto;
      padding: 2px 0 18px;
      text-align: left;
    }

    .portfolio-inline-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 10px;
    }

    td:last-child { text-align: center; }

    .portfolio-name-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    input[type="text"], input[type="number"], input[type="date"], select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 5px;
      min-height: 36px;
      padding: 7px 9px;
      font: inherit;
      background: white;
      color: var(--ink);
    }

    select { appearance: auto; }

    input:focus {
      border-color: #7a9e99;
      box-shadow: 0 0 0 3px rgba(23, 107, 97, 0.1);
      outline: none;
    }

    .portfolio-row .portfolio-name {
      width: 100%;
      min-height: 34px;
      border: 0;
      padding: 4px 2px;
      background: transparent;
      font-weight: 700;
    }

    .portfolio-row .portfolio-name::placeholder { color: #8c929b; opacity: 1; }
    .portfolio-row .portfolio-name:focus { box-shadow: none; border-bottom: 1px solid #7a9e99; border-radius: 0; }

    button {
      font: inherit;
      cursor: pointer;
    }

    .icon-button {
      width: 30px;
      height: 30px;
      flex: 0 0 30px;
      padding: 0;
      border-radius: 50%;
      border: 1px solid #c8ced5;
      background: #fff;
      color: #30363e;
      font-size: 18px;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .icon-button:hover { border-color: #929aa4; background: #fafbfb; }

    .remove-button {
      position: relative;
      width: 26px;
      height: 26px;
      padding: 0;
      border: 1px solid #b94b52;
      border-radius: 50%;
      background: #b94b52;
      color: white;
      font-family: Arial, sans-serif;
      font-size: 0;
      font-weight: 400;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .remove-button::before,
    .remove-button::after,
    .add-button::before,
    .add-button::after {
      content: "";
      position: absolute;
      top: 50%;
      left: 50%;
      width: 10px;
      height: 1px;
      border-radius: 1px;
      background: white;
    }

    .remove-button::before { transform: translate(-50%, -50%) rotate(45deg); }
    .remove-button::after { transform: translate(-50%, -50%) rotate(-45deg); }
    .remove-button:hover { background: #a63d45; border-color: #a63d45; }

    .row-actions { display: flex; align-items: center; justify-content: center; gap: 6px; }
    .edit-button {
      min-height: 28px;
      border: 1px solid #c8ced5;
      border-radius: 5px;
      padding: 4px 7px;
      background: white;
      color: #4c545e;
      font-size: 11px;
      font-weight: 700;
    }
    .editing-row .edit-button { border-color: var(--accent); background: var(--accent); color: white; }

    .add-limit-message {
      margin: 8px 2px 0;
      color: var(--muted);
      font-size: 11px;
    }

    .weight-field { position: relative; }
    .weight-field .weight-input { padding-right: 25px; appearance: textfield; }
    .weight-field .weight-input::-webkit-inner-spin-button,
    .weight-field .weight-input::-webkit-outer-spin-button { appearance: none; margin: 0; }
    .weight-suffix {
      position: absolute;
      top: 50%;
      right: 9px;
      transform: translateY(-50%);
      color: #626a74;
      pointer-events: none;
    }

    .company-cell { position: relative; padding-left: 52px; }
    .date-cell { position: relative; }

    .ticker-input[readonly] {
      border-color: transparent;
      background: #f7f8f9;
      color: #4f5660;
      cursor: default;
    }

    .suggestions {
      position: absolute;
      z-index: 10;
      top: calc(100% - 7px);
      left: 52px;
      right: 12px;
      overflow: hidden;
      border: 1px solid #cbd1d8;
      border-radius: 6px;
      background: white;
      box-shadow: 0 10px 26px rgba(25, 35, 45, 0.14);
    }

    .suggestion {
      width: 100%;
      border: 0;
      border-bottom: 1px solid #edf0f2;
      padding: 9px 11px;
      background: white;
      text-align: left;
    }

    .suggestion:last-child { border-bottom: 0; }
    .suggestion:hover, .suggestion:focus { background: #f2f7f6; outline: none; }
    .suggestion-name { display: block; color: var(--ink); font-weight: 650; }
    .suggestion-meta { display: block; margin-top: 2px; color: var(--muted); font-size: 12px; }

    .purchase-close { color: #424951; font-variant-numeric: tabular-nums; }
    .price-note { display: block; margin-top: 2px; color: var(--muted); font-size: 11px; }
    .portfolio-meta { color: #555d67; font-size: 13px; font-weight: 500; white-space: nowrap; }

    .calendar-popover {
      position: absolute;
      z-index: 20;
      top: calc(100% - 7px);
      left: 12px;
      width: 276px;
      padding: 12px;
      border: 1px solid #cbd1d8;
      border-radius: 7px;
      background: white;
      box-shadow: 0 10px 26px rgba(25, 35, 45, 0.14);
    }

    .calendar-header {
      display: grid;
      grid-template-columns: 30px 1fr 30px;
      align-items: center;
      margin-bottom: 8px;
    }

    .calendar-title {
      border: 0;
      border-radius: 4px;
      padding: 5px 7px;
      background: transparent;
      color: var(--ink);
      text-align: center;
      font-size: 13px;
      font-weight: 700;
    }
    .calendar-title:hover { background: #f0f3f4; }
    .calendar-nav { border: 0; background: transparent; color: #505760; font-size: 18px; }
    .calendar-nav:disabled { color: #c4c8cd; cursor: default; }

    .calendar-grid {
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 3px;
    }

    .calendar-weekday { padding: 3px 0; color: var(--muted); font-size: 10px; text-align: center; }
    .calendar-day {
      aspect-ratio: 1;
      border: 0;
      border-radius: 50%;
      background: transparent;
      color: var(--ink);
      font-size: 12px;
    }

    .calendar-day:hover:not(:disabled) { background: #e8f1ef; }
    .calendar-day:disabled { color: #c4c7cc; cursor: default; }
    .calendar-day.selected { background: var(--accent); color: white; }
    .calendar-empty { aspect-ratio: 1; }

    .month-selector-header {
      display: grid;
      grid-template-columns: 30px 1fr 30px;
      align-items: center;
      margin-bottom: 12px;
    }

    .selector-range { text-align: center; font-size: 13px; font-weight: 700; }

    .selector-section-label {
      margin: 11px 0 6px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .year-selector-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 5px;
    }

    .year-option,
    .month-option {
      min-height: 34px;
      border: 1px solid transparent;
      border-radius: 5px;
      background: #f5f7f8;
      color: #414850;
      font-size: 12px;
    }

    .year-option:hover:not(:disabled),
    .month-option:hover:not(:disabled) { border-color: #a8c4c0; background: #edf5f3; }

    .year-option.selected,
    .month-option.selected { border-color: var(--accent); color: var(--accent); font-weight: 700; }

    .year-option:disabled,
    .month-option:disabled { color: #b9bec4; background: #fafafa; cursor: default; }

    .month-selector-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }

    tfoot td {
      height: 52px;
      background: #fbfcfc;
      text-align: center;
    }

    .add-button {
      position: relative;
      width: 26px;
      height: 26px;
      flex-basis: 26px;
      color: white;
      border-color: #2f7d68;
      background: #2f7d68;
      font-family: Arial, sans-serif;
      font-size: 0;
      font-weight: 400;
      line-height: 1;
    }
    .add-button::before { transform: translate(-50%, -50%); }
    .add-button::after { transform: translate(-50%, -50%) rotate(90deg); }
    .add-button:hover { border-color: #286a59; background: #286a59; }

    .dialog-label {
      display: block;
      margin-bottom: 6px;
      color: #4e5560;
      font-size: 12px;
      font-weight: 650;
    }

    .dialog-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 18px;
    }

    .dialog-button {
      min-height: 36px;
      padding: 7px 12px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: white;
      color: #414850;
      font-weight: 650;
    }

    .dialog-button.primary { border-color: var(--accent); background: var(--accent); color: white; }

    .section-title {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }

    .current-section-title { margin-top: 0; }

    .candidate-section,
    .analysis-section {
      margin-top: 34px;
      border-top: 1px solid var(--line);
      padding-top: 24px;
    }

    .candidate-shell {
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: visible;
    }

    .candidate-fields {
      display: grid;
      grid-template-columns: minmax(260px, 2fr) minmax(150px, 1fr);
      gap: 14px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }

    .field-label {
      display: block;
      margin-bottom: 6px;
      color: #4e5560;
      font-size: 12px;
      font-weight: 650;
    }

    .candidate-company { position: relative; }
    .candidate-suggestions { left: 0; right: 0; }
    .candidate-sector {
      min-height: 16px;
      margin-top: 5px;
      color: var(--muted);
      font-size: 11px;
    }
    .funding-area { padding: 16px; }

    .funding-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .funding-title { color: #4e5560; font-size: 12px; font-weight: 650; }

    .funding-segmented {
      display: inline-flex;
      padding: 2px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f4f6f7;
    }

    .funding-option {
      min-height: 32px;
      border: 0;
      border-radius: 4px;
      padding: 6px 11px;
      background: transparent;
      color: #59616b;
      font-size: 12px;
      font-weight: 650;
    }

    .funding-option[aria-pressed="true"] {
      background: white;
      color: var(--ink);
      box-shadow: 0 1px 3px rgba(25, 35, 45, 0.12);
    }
    .funding-option:disabled { opacity: 0.45; cursor: default; }

    .cash-funding {
      margin-top: 14px;
      color: #4e6f67;
      font-size: 13px;
      font-weight: 650;
    }

    .cash-priority {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 14px;
    }

    .reduction-scope {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }

    .automatic-reduction-option {
      display: flex;
      align-items: center;
      gap: 9px;
      margin-top: 14px;
      color: #343b43;
      font-size: 12px;
      font-weight: 650;
    }
    .automatic-reduction-option input,
    .reduction-checkbox { width: 16px; height: 16px; accent-color: var(--accent); }

    .reduction-list {
      margin-top: 14px;
      border-top: 1px solid var(--line);
    }

    .reduction-row {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 10px;
      align-items: center;
      min-height: 50px;
      border-bottom: 1px solid var(--line);
      color: #343b43;
      font-size: 13px;
    }
    .reduction-row.manual { grid-template-columns: 22px minmax(180px, 1fr) auto; }
    .reduction-row.ineligible { color: #7b8289; }

    .reduction-row:last-child { border-bottom: 0; }
    .reduction-security strong { display: block; font-size: 13px; }
    .reduction-security span { color: var(--muted); font-size: 11px; }
    .reduction-amount { color: var(--accent); font-weight: 700; }
    .reduction-amount.ineligible { color: var(--muted); }
    .reduction-checkbox:disabled { cursor: not-allowed; opacity: .45; }
    .reduction-empty { padding: 16px 0; color: var(--muted); font-size: 13px; }

    .funding-summary {
      display: flex;
      justify-content: flex-end;
      gap: 6px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .funding-summary.complete { color: var(--accent); font-weight: 700; }
    .funding-summary.warning,
    .cash-funding.warning { color: #9a5b13; font-weight: 700; }

    .cash-row td { background: #fbfcfc; }
    .cash-name {
      padding-left: 52px;
      color: #343b43;
      font-weight: 650;
    }

    .analysis-controls {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }

    .analysis-control-label {
      display: block;
      margin-bottom: 6px;
      color: #4e5560;
      font-size: 12px;
      font-weight: 650;
    }

    .analysis-action {
      min-height: 36px;
      border: 1px solid var(--accent);
      border-radius: 5px;
      padding: 7px 13px;
      background: var(--accent);
      color: white;
      font-size: 12px;
      font-weight: 700;
    }

    .analysis-window-control { margin-bottom: 14px; }
    .analysis-window-selector { display: inline-flex; }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .metric-block {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 16px;
      background: #fff;
    }

    .metric-block h3 { margin: 0 0 14px; font-size: 14px; letter-spacing: 0; }
    .metric-line {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      min-height: 29px;
      color: var(--muted);
      font-size: 12px;
    }
    .metric-line strong { color: var(--ink); font-size: 16px; }
    .metric-line.difference { border-top: 1px solid var(--line); padding-top: 9px; margin-top: 5px; }
    .metric-line.difference strong { color: var(--accent); font-size: 13px; }

    .sensitivity-analysis { margin-top: 24px; }
    .sensitivity-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 9px;
    }
    .sensitivity-header h3 { margin: 0; font-size: 14px; letter-spacing: 0; }
    .sensitivity-objective { color: var(--muted); font-size: 11px; }
    .sensitivity-shell { overflow-x: auto; border: 1px solid var(--line); border-radius: 7px; }
    .sensitivity-table { min-width: 650px; table-layout: auto; }
    .sensitivity-table th,
    .sensitivity-table td { width: auto; padding: 9px 12px; text-align: left; font-size: 12px; white-space: nowrap; }
    .sensitivity-table tbody tr.selected td { background: #eef6f3; }
    .sensitivity-table td:first-child,
    .sensitivity-table td:nth-child(2) { font-weight: 700; }
    .sensitivity-label {
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-top: 9px;
      color: var(--muted);
      font-size: 11px;
    }
    .sensitivity-label strong { color: var(--accent); font-size: 12px; }
    .sensitivity-label.very-sensitive strong { color: #a34f36; }

    .analysis-funding {
      margin-top: 14px;
      padding: 14px 0;
      border-top: 1px solid var(--line);
      color: #414850;
      font-size: 13px;
    }
    .analysis-funding strong { display: block; margin-bottom: 6px; }
    .analysis-funding p { margin: 4px 0; }
    .analysis-note { margin: 9px 2px 0; color: var(--muted); font-size: 11px; }
    .analysis-message { margin: 0 0 12px; color: var(--muted); font-size: 13px; }
    .analysis-message.warning { color: #9a5b13; font-weight: 650; }

    .status {
      min-height: 22px;
      margin: 10px 2px 0;
      color: var(--muted);
      font-size: 13px;
    }

    .status.warning { color: #8a5a16; }

    @media (max-width: 760px) {
      main { padding: 20px 12px 40px; }
      .table-shell { overflow-x: auto; }
      .candidate-fields { grid-template-columns: 1fr; }
      .funding-header { align-items: flex-start; flex-direction: column; }
      .cash-priority { align-items: flex-start; flex-direction: column; }
      .reduction-scope { align-items: flex-start; flex-direction: column; }
      .reduction-scope .funding-segmented { width: 100%; flex-direction: column; }
      .reduction-scope .funding-option { width: 100%; text-align: left; }
      .reduction-row { grid-template-columns: minmax(130px, 1fr) auto; }
      .reduction-row.manual { grid-template-columns: 22px minmax(130px, 1fr) auto; }
      .analysis-controls { align-items: flex-start; flex-direction: column; }
      .metric-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header class="app-header">
    <div class="app-header-inner">
      <div class="app-brand" aria-label="FitCheck">FitCheck</div>
    </div>
  </header>
  <main>
    <h2 class="section-title current-section-title">Current Portfolio</h2>
    <div class="table-shell">
      <table id="holdings-table">
        <thead>
          <tr>
            <th>Company Name</th>
            <th>Ticker</th>
            <th>Portfolio Weight (%)</th>
            <th>Purchase Date</th>
            <th>Price</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr id="empty-portfolio-row" class="empty-portfolio-row">
            <td colspan="6">
              <button id="add-portfolio-trigger" class="add-portfolio-trigger" type="button" onclick="openPortfolioDialog()" aria-expanded="false" aria-controls="portfolio-inline-form">Add portfolio</button>
              <div id="portfolio-inline-form" class="portfolio-inline-form" hidden>
                <form onsubmit="createPortfolio(event)">
                  <label class="dialog-label" for="new-portfolio-name">Portfolio name</label>
                  <input id="new-portfolio-name" type="text" autocomplete="off" placeholder="Enter portfolio name" required>
                  <div class="portfolio-inline-actions">
                    <button class="dialog-button" type="button" onclick="closePortfolioDialog()">Cancel</button>
                    <button class="dialog-button primary" type="submit">Create</button>
                  </div>
                </form>
              </div>
            </td>
          </tr>
          <tr id="portfolio-row" class="portfolio-row" hidden>
            <td>
              <div class="portfolio-name-wrap">
                <button id="expand-button" class="icon-button" type="button" onclick="toggleExpanded()" aria-label="Collapse holdings" aria-expanded="true">-</button>
                <input id="portfolio-name" class="portfolio-name" type="text" placeholder="Add portfolio name" aria-label="Portfolio name">
              </div>
            </td>
            <td></td>
            <td id="portfolio-row-weight">0.00%</td>
            <td></td>
            <td></td>
            <td id="portfolio-row-count" class="portfolio-meta">0 holdings</td>
          </tr>
        </tbody>
        <tfoot hidden>
          <tr>
            <td colspan="6">
              <button class="icon-button add-button" type="button" onclick="addRow()" aria-label="Add holding" title="Add holding">+</button>
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
    <div id="add-limit-message" class="add-limit-message" hidden>Portfolio is at 100.00%. Lower an existing weight to add another security.</div>
    <div id="status" class="status" aria-live="polite"></div>

    <section id="candidate-section" class="candidate-section" hidden>
      <h2 class="section-title">Candidate Portfolio</h2>
      <div class="candidate-shell">
        <div class="candidate-fields">
          <div class="candidate-company">
            <label class="field-label" for="candidate-company">Company Name</label>
            <input id="candidate-company" type="text" placeholder="Search company" autocomplete="off">
            <div id="candidate-suggestions" class="suggestions candidate-suggestions" hidden></div>
          </div>
          <div>
            <label class="field-label" for="candidate-ticker">Ticker</label>
            <input id="candidate-ticker" class="ticker-input" type="text" placeholder="Ticker" readonly>
            <div id="candidate-sector" class="candidate-sector"></div>
          </div>
        </div>
        <div class="funding-area">
          <div class="funding-header">
            <span class="funding-title">Funding source</span>
            <div class="funding-segmented" role="group" aria-label="Funding source">
              <button id="funding-cash" class="funding-option" type="button" aria-pressed="true" onclick="setFundingMode('cash')">Available cash</button>
              <button id="funding-reduce" class="funding-option" type="button" aria-pressed="false" onclick="setFundingMode('reduce')">Reduce Positions</button>
            </div>
          </div>
          <div class="cash-priority">
            <span class="funding-title">Primary cash source</span>
            <div class="funding-segmented" role="group" aria-label="Primary cash source">
              <button id="cash-source-cad" class="funding-option" type="button" aria-pressed="true" onclick="setCashSource('cad')">CAD Cash</button>
              <button id="cash-source-usd" class="funding-option" type="button" aria-pressed="false" onclick="setCashSource('usd')">USD Cash</button>
            </div>
          </div>
          <div id="cash-funding" class="cash-funding" aria-live="polite"></div>
          <div id="reduce-funding" hidden>
            <label id="automatic-reduction-option" class="automatic-reduction-option" hidden>
              <input id="optimize-reductions" type="checkbox" checked onchange="setOptimizeReductions(this.checked)">
              <span>Optimize Reduction Automatically</span>
            </label>
            <div id="reduction-scope-control" class="reduction-scope" hidden>
              <span class="funding-title">Reduction scope</span>
              <div class="funding-segmented" role="group" aria-label="Reduction scope">
                <button id="reduction-scope-all" class="funding-option" type="button" aria-pressed="true" onclick="setReductionScope('all')">All eligible positions</button>
                <button id="reduction-scope-sector" class="funding-option" type="button" aria-pressed="false" onclick="setReductionScope('sector')">Candidate sector only</button>
              </div>
            </div>
            <div id="reduction-list" class="reduction-list"></div>
            <div id="funding-summary" class="funding-summary" aria-live="polite"></div>
          </div>
        </div>
      </div>
    </section>

    <section id="analysis-section" class="analysis-section" hidden>
      <h2 class="section-title">Portfolio Metrics</h2>
      <div class="analysis-controls">
        <div>
          <span class="analysis-control-label">Optimize for</span>
          <div class="funding-segmented" role="group" aria-label="Metric to optimize">
            <button id="optimize-return" class="funding-option" type="button" aria-pressed="false" onclick="setOptimizeMetric('return')">Return</button>
            <button id="optimize-volatility" class="funding-option" type="button" aria-pressed="false" onclick="setOptimizeMetric('volatility')">Volatility</button>
            <button id="optimize-sharpe" class="funding-option" type="button" aria-pressed="true" onclick="setOptimizeMetric('sharpe')">Sharpe ratio</button>
          </div>
        </div>
        <button id="calculate-metrics" class="analysis-action" type="button" onclick="runAnalysis()">Calculate metrics</button>
      </div>
      <div id="analysis-message" class="analysis-message" aria-live="polite"></div>
      <div id="analysis-results" hidden>
        <div class="analysis-window-control">
          <span class="analysis-control-label">Historical window</span>
          <div id="analysis-window-selector" class="analysis-window-selector funding-segmented" role="group" aria-label="Historical window"></div>
        </div>
        <div id="metric-grid" class="metric-grid"></div>
        <section class="sensitivity-analysis" aria-labelledby="sensitivity-title">
          <div class="sensitivity-header">
            <h3 id="sensitivity-title">Sensitivity Analysis</h3>
            <span id="sensitivity-objective" class="sensitivity-objective"></span>
          </div>
          <div class="sensitivity-shell">
            <table class="sensitivity-table">
              <thead>
                <tr><th>Window</th><th>Candidate Weight</th><th>Return</th><th>Volatility</th><th>Sharpe Ratio</th></tr>
              </thead>
              <tbody id="sensitivity-body"></tbody>
            </table>
          </div>
          <div id="sensitivity-label" class="sensitivity-label" aria-live="polite"></div>
        </section>
        <div id="analysis-funding" class="analysis-funding"></div>
        <p class="analysis-note">Annualized metrics use adjusted daily closes and a 0% risk-free rate for Sharpe ratio.</p>
      </div>
    </section>

  </main>

  <script>
    const today = "__TODAY__";
    let isExpanded = true;
    let portfolioCreated = false;
    let portfolioDialogOpen = false;
    let fundingMode = "cash";
    let cashSource = "cad";
    let reductionScope = "all";
    let optimizeReductions = true;
    let selectedReductionTickers = new Set();
    let reductionEligibilityKey = "";
    let optimizeMetric = "sharpe";
    let analysisRows = [];
    let analysisWindow = "1Y";
    let analysisRevision = 0;
    let analysisRefreshTimer;
    let candidateSearchTimer;
    const sectorGroupOrder = ["Materials", "FIGs", "TMTH", "Consumers", "Infrastructure", "Industrials", "Unclassified"];

    function portfolioName() {
      return document.getElementById("portfolio-name").value.trim();
    }

    function openPortfolioDialog() {
      const panel = document.getElementById("portfolio-inline-form");
      const trigger = document.getElementById("add-portfolio-trigger");
      const input = document.getElementById("new-portfolio-name");
      if (!portfolioDialogOpen) input.value = "";
      input.setCustomValidity("");
      panel.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      portfolioDialogOpen = true;
      if (window.fitcheckUiState) window.fitcheckUiState.portfolioModalOpen = true;
      input.focus();
    }

    function closePortfolioDialog() {
      document.getElementById("portfolio-inline-form").hidden = true;
      document.getElementById("add-portfolio-trigger").setAttribute("aria-expanded", "false");
      portfolioDialogOpen = false;
      if (window.fitcheckUiState) window.fitcheckUiState.portfolioModalOpen = false;
    }

    function createPortfolio(event) {
      event.preventDefault();
      const input = document.getElementById("new-portfolio-name");
      const name = input.value.trim();
      if (!name) {
        input.setCustomValidity("Enter a portfolio name.");
        input.reportValidity();
        return;
      }

      input.setCustomValidity("");
      portfolioCreated = true;
      document.getElementById("portfolio-name").value = name;
      document.getElementById("empty-portfolio-row").hidden = true;
      document.getElementById("portfolio-row").hidden = false;
      document.getElementById("candidate-section").hidden = false;
      document.getElementById("analysis-section").hidden = false;
      createCashRows();
      closePortfolioDialog();
      setExpanded(true);
      addRow();
    }

    function createCashRows() {
      if (document.getElementById("cad-cash-row")) return;
      let previous = document.getElementById("portfolio-row");
      [["cad", "CAD Cash"], ["usd", "USD Cash"]].forEach(([currency, label]) => {
        const row = document.createElement("tr");
        row.id = `${currency}-cash-row`;
        row.className = "cash-row";
        row.innerHTML = `
          <td class="cash-name">${label}</td>
          <td></td>
          <td><div class="weight-field"><input id="${currency}-cash-weight" class="weight-input" type="number" min="0" max="100" step="0.01" placeholder="0.00" aria-label="${label} portfolio weight percent"><span class="weight-suffix" aria-hidden="true">%</span></div></td>
          <td></td>
          <td></td>
          <td></td>
        `;
        previous.after(row);
        previous = row;
        const input = row.querySelector(".weight-input");
        input.addEventListener("input", () => {
          updatePortfolioSummary();
        });
        input.addEventListener("change", () => formatWeight(input));
        input.addEventListener("blur", () => formatWeight(input));
        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter") input.blur();
        });
      });
      updatePortfolioSummary();
    }

    function setExpanded(expanded) {
      isExpanded = expanded;
      const button = document.getElementById("expand-button");
      button.textContent = expanded ? "-" : "+";
      button.setAttribute("aria-label", expanded ? "Collapse holdings" : "Expand holdings");
      button.setAttribute("aria-expanded", String(expanded));
      document.querySelectorAll(".holding-row").forEach((row) => {
        row.hidden = !expanded;
      });
      document.querySelectorAll(".draft-row, .draft-actions-row").forEach((row) => {
        row.hidden = !expanded;
      });
      document.querySelectorAll(".cash-row").forEach((row) => {
        row.hidden = !expanded;
      });
      document.querySelectorAll(".sector-header").forEach((row) => {
        row.hidden = !expanded;
      });
      document.querySelector("#holdings-table tfoot").hidden =
        !portfolioCreated || !expanded || Boolean(document.querySelector(".draft-row"))
        || Boolean(document.querySelector(".editing-row"))
        || currentPortfolioWeight() >= 100 - 0.005;
      closeSuggestions();
      closeCalendars();
    }

    function toggleExpanded() {
      setExpanded(!isExpanded);
    }

    function addRow() {
      const editingRow = document.querySelector(".editing-row");
      if (editingRow) {
        editingRow.querySelector(".company-input").focus();
        return;
      }
      if (currentPortfolioWeight() >= 100 - 0.005) {
        updatePortfolioSummary();
        return;
      }
      const existingDraft = document.querySelector(".draft-row");
      if (existingDraft) {
        existingDraft.querySelector(".company-input").focus();
        return;
      }

      const tbody = document.querySelector("#holdings-table tbody");
      const tr = document.createElement("tr");
      tr.className = "draft-row";
      tr.innerHTML = `
        <td class="company-cell">
          <input class="company-input" type="text" placeholder="Search company" autocomplete="off" aria-label="Company name">
          <div class="suggestions" hidden></div>
          <select class="sector-input" hidden disabled aria-label="Sector">
            <option value="">Sector</option>
            <option>Materials</option><option>FIGs</option><option>TMTH</option>
            <option>Consumers</option><option>Infrastructure</option><option>Industrials</option>
            <option>Unclassified</option>
          </select>
        </td>
        <td><input class="ticker-input" type="text" placeholder="Ticker" readonly aria-label="Ticker"></td>
        <td><div class="weight-field"><input class="weight-input" type="number" min="1" max="6" step="0.01" placeholder="0.00" aria-label="Portfolio weight percent"><span class="weight-suffix" aria-hidden="true">%</span></div></td>
        <td class="date-cell">
          <input class="date-input" type="text" placeholder="Select date" readonly aria-label="Purchase date">
          <div class="calendar-popover" hidden></div>
        </td>
        <td class="purchase-close"><span class="price-value">-</span><span class="price-note"></span></td>
        <td><div class="row-actions"><button class="edit-button" type="button" onclick="toggleHoldingEdit(this)" hidden>Edit</button><button class="remove-button" type="button" onclick="removeRow(this)" aria-label="Remove holding" title="Remove holding">x</button></div></td>
      `;
      tbody.appendChild(tr);

      const actions = document.createElement("tr");
      actions.className = "draft-actions-row";
      actions.innerHTML = `
        <td colspan="6">
          <div class="draft-actions">
            <span class="draft-validation">Complete company, weight, and purchase date.</span>
            <button class="commit-holding-button" type="button" onclick="commitHolding(this)" disabled>Add to Portfolio</button>
          </div>
        </td>
      `;
      tbody.appendChild(actions);

      bindRow(tr);
      tr.hidden = !isExpanded;
      actions.hidden = !isExpanded;
      document.querySelector("#holdings-table tfoot").hidden = true;
      updateDraftValidation(tr);
      tr.querySelector(".company-input").focus();
    }

    function bindRow(row) {
      const companyInput = row.querySelector(".company-input");
      let searchTimer;

      companyInput.addEventListener("input", () => {
        row.dataset.selectedCompany = "";
        delete row.dataset.gicsSector;
        delete row.dataset.sectorGroup;
        row.querySelector(".ticker-input").value = "";
        row.querySelector(".sector-input").value = "";
        row.querySelector(".date-input").value = "";
        resetPrice(row);
        clearTimeout(searchTimer);
        const query = companyInput.value.trim();
        if (query.length < 2) {
          hideSuggestions(row);
        } else {
          searchTimer = setTimeout(() => searchCompanies(row, query), 250);
        }
        updateDraftValidation(row);
        if (row.classList.contains("holding-row")) updatePortfolioSummary();
      });

      companyInput.addEventListener("keydown", (event) => {
        const first = row.querySelector(".suggestion");
        if (event.key === "Enter" && first) {
          event.preventDefault();
          first.click();
        }
        if (event.key === "Escape") hideSuggestions(row);
      });

      row.querySelectorAll(".weight-input").forEach((input) => {
        input.addEventListener("input", () => {
          updateDraftValidation(row);
          if (row.classList.contains("holding-row")) updatePortfolioSummary();
        });
        input.addEventListener("change", () => formatWeight(input));
        input.addEventListener("blur", () => formatWeight(input));
        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter") input.blur();
        });
      });

      const dateInput = row.querySelector(".date-input");
      const calendarPopover = row.querySelector(".calendar-popover");
      calendarPopover.addEventListener("click", (event) => event.stopPropagation());
      dateInput.addEventListener("click", (event) => {
        event.stopPropagation();
        openCalendar(row);
      });
      dateInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openCalendar(row);
        }
        if (event.key === "Escape") closeCalendars();
      });

      row.querySelector(".sector-input").addEventListener("change", (event) => {
        row.dataset.sectorGroup = event.target.value;
        row.dataset.gicsSector = event.target.value;
        updateDraftValidation(row);
        updatePortfolioSummary();
      });
    }

    async function searchCompanies(row, query) {
      const list = row.querySelector(".suggestions");
      try {
        const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
        const results = await response.json();
        if (row.querySelector(".company-input").value.trim() !== query) return;
        renderSuggestions(row, results);
      } catch (error) {
        list.hidden = true;
      }
    }

    function renderSuggestions(row, results) {
      const list = row.querySelector(".suggestions");
      list.replaceChildren();
      results.forEach((result) => {
        const option = document.createElement("button");
        option.type = "button";
        option.className = "suggestion";

        const name = document.createElement("span");
        name.className = "suggestion-name";
        name.textContent = result.company_name;

        const meta = document.createElement("span");
        meta.className = "suggestion-meta";
        meta.textContent = [result.ticker, result.exchange].filter(Boolean).join(" · ");

        option.append(name, meta);
        option.addEventListener("click", () => selectCompany(row, result));
        list.appendChild(option);
      });
      list.hidden = results.length === 0;
    }

    function selectCompany(row, result) {
      row.querySelector(".company-input").value = result.company_name;
      row.querySelector(".ticker-input").value = result.ticker;
      row.querySelector(".sector-input").value = "";
      row.querySelector(".date-input").value = "";
      row.dataset.selectedCompany = result.company_name;
      hideSuggestions(row);
      resetPrice(row);
      updateDraftValidation(row);
      classifyHoldingRow(row, result.ticker);
      row.querySelector(".weight-input").focus();
    }

    async function classifyHoldingRow(row, ticker) {
      row.dataset.sectorGroup = "Classifying";
      updateDraftValidation(row);
      try {
        const response = await fetch(`/classification?ticker=${encodeURIComponent(ticker)}`);
        const result = await response.json();
        if (row.querySelector(".ticker-input").value.trim().toUpperCase() !== ticker.toUpperCase()) return;
        row.dataset.gicsSector = result.gics_sector || "Unknown";
        row.dataset.sectorGroup = response.ok ? result.sector_group : "Unclassified";
      } catch (error) {
        row.dataset.gicsSector = "Unknown";
        row.dataset.sectorGroup = "Unclassified";
      }
      row.querySelector(".sector-input").value = row.dataset.sectorGroup;
      updateDraftValidation(row);
      if (row.classList.contains("holding-row")) updatePortfolioSummary();
    }

    function updateDraftValidation(row) {
      if (!row?.classList.contains("draft-row")) return;
      const actions = row.nextElementSibling;
      if (!actions?.classList.contains("draft-actions-row")) return;

      const companyReady = Boolean(
        row.dataset.selectedCompany && row.querySelector(".ticker-input").value.trim()
      );
      const weight = Number(row.querySelector(".weight-input").value || 0);
      const weightReady = weight >= 1 && weight <= 6;
      const dateReady = Boolean(row.querySelector(".date-input").value);
      const sectorReady = Boolean(
        row.dataset.sectorGroup && row.dataset.sectorGroup !== "Classifying"
      );
      const fitsPortfolio = currentPortfolioWeight() + weight <= 100 + 0.005;
      const ready = companyReady && weightReady && dateReady && sectorReady && fitsPortfolio;
      const button = actions.querySelector(".commit-holding-button");
      const message = actions.querySelector(".draft-validation");
      button.disabled = !ready;

      if (!companyReady) message.textContent = "Select a company from the suggestions.";
      else if (!weightReady) message.textContent = "Enter a weight between 1.00% and 6.00%.";
      else if (!dateReady) message.textContent = "Select a purchase date.";
      else if (!sectorReady) message.textContent = "Identifying sector...";
      else if (!fitsPortfolio) message.textContent = "Lower another weight before adding this security.";
      else message.textContent = "Ready to add.";
    }

    function commitHolding(button) {
      const actions = button.closest(".draft-actions-row");
      const row = actions?.previousElementSibling;
      if (!row?.classList.contains("draft-row")) return;
      updateDraftValidation(row);
      if (button.disabled) return;

      row.classList.remove("draft-row");
      row.classList.add("holding-row");
      row.dataset.committed = "true";
      row.querySelector(".company-input").readOnly = true;
      row.querySelector(".weight-input").readOnly = true;
      row.querySelector(".sector-input").disabled = true;
      row.querySelector(".edit-button").hidden = false;
      actions.remove();

      organizeHoldingRows();
      updatePortfolioSummary();
    }

    function toggleHoldingEdit(button) {
      const row = button.closest("tr");
      const editing = row.classList.contains("editing-row");
      const company = row.querySelector(".company-input");
      const weight = row.querySelector(".weight-input");
      const sector = row.querySelector(".sector-input");

      if (!editing) {
        row.classList.add("editing-row");
        company.readOnly = false;
        weight.readOnly = false;
        sector.disabled = false;
        button.textContent = "Save";
        updatePortfolioSummary();
        company.focus();
        return;
      }

      const valid = row.dataset.selectedCompany
        && row.querySelector(".ticker-input").value.trim()
        && Number(weight.value) >= 1
        && Number(weight.value) <= 6
        && row.querySelector(".date-input").value
        && row.dataset.sectorGroup
        && row.dataset.sectorGroup !== "Classifying";
      const status = document.getElementById("status");
      if (!valid) {
        status.textContent = "Complete company, sector, weight, and purchase date before saving.";
        status.className = "status warning";
        return;
      }

      row.classList.remove("editing-row");
      company.readOnly = true;
      weight.readOnly = true;
      sector.disabled = true;
      button.textContent = "Edit";
      status.textContent = "";
      status.className = "status";
      organizeHoldingRows();
      updatePortfolioSummary();
    }

    function organizeHoldingRows() {
      const tbody = document.querySelector("#holdings-table tbody");
      tbody.querySelectorAll(".sector-header").forEach((row) => row.remove());
      const rows = [...tbody.querySelectorAll(".holding-row")];
      const grouped = new Map();
      const incomplete = [];

      rows.forEach((row) => {
        const group = row.dataset.sectorGroup;
        if (!group || group === "Classifying") {
          incomplete.push(row);
          return;
        }
        if (!grouped.has(group)) grouped.set(group, []);
        grouped.get(group).push(row);
      });

      sectorGroupOrder.forEach((group) => {
        const groupRows = grouped.get(group);
        if (!groupRows?.length) return;
        groupRows.sort((first, second) =>
          first.querySelector(".company-input").value.localeCompare(
            second.querySelector(".company-input").value,
            undefined,
            { sensitivity: "base" }
          )
        );

        const header = document.createElement("tr");
        header.className = "sector-header";
        header.hidden = !isExpanded;
        const cell = document.createElement("td");
        cell.colSpan = 6;
        cell.textContent = group;
        header.appendChild(cell);
        tbody.appendChild(header);
        groupRows.forEach((row) => tbody.appendChild(row));
      });

      incomplete.forEach((row) => tbody.appendChild(row));
    }

    function bindCandidateInput() {
      const companyInput = document.getElementById("candidate-company");

      companyInput.addEventListener("input", () => {
        document.getElementById("candidate-ticker").value = "";
        document.getElementById("candidate-section").dataset.sectorGroup = "";
        document.getElementById("candidate-sector").textContent = "";
        updateFundingSummary();
        clearTimeout(candidateSearchTimer);
        const query = companyInput.value.trim();
        if (query.length < 2) {
          hideCandidateSuggestions();
        } else {
          candidateSearchTimer = setTimeout(() => searchCandidateCompanies(query), 250);
        }
      });

      companyInput.addEventListener("keydown", (event) => {
        const first = document.querySelector("#candidate-suggestions .suggestion");
        if (event.key === "Enter" && first) {
          event.preventDefault();
          first.click();
        }
        if (event.key === "Escape") hideCandidateSuggestions();
      });

    }

    async function searchCandidateCompanies(query) {
      const list = document.getElementById("candidate-suggestions");
      try {
        const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
        const results = await response.json();
        if (document.getElementById("candidate-company").value.trim() !== query) return;
        renderCandidateSuggestions(results);
      } catch (error) {
        list.hidden = true;
      }
    }

    function renderCandidateSuggestions(results) {
      const list = document.getElementById("candidate-suggestions");
      list.replaceChildren();
      results.forEach((result) => {
        const option = document.createElement("button");
        option.type = "button";
        option.className = "suggestion";

        const name = document.createElement("span");
        name.className = "suggestion-name";
        name.textContent = result.company_name;

        const meta = document.createElement("span");
        meta.className = "suggestion-meta";
        meta.textContent = [result.ticker, result.exchange].filter(Boolean).join(" · ");

        option.append(name, meta);
        option.addEventListener("click", () => selectCandidateCompany(result));
        list.appendChild(option);
      });
      list.hidden = results.length === 0;
    }

    function selectCandidateCompany(result) {
      document.getElementById("candidate-company").value = result.company_name;
      document.getElementById("candidate-ticker").value = result.ticker;
      hideCandidateSuggestions();
      classifyCandidate(result.ticker);
    }

    async function classifyCandidate(ticker) {
      const section = document.getElementById("candidate-section");
      const sector = document.getElementById("candidate-sector");
      section.dataset.sectorGroup = "";
      sector.textContent = "Identifying sector...";
      if (fundingMode === "reduce") updateFundingSummary();
      else scheduleAnalysisRefresh();

      try {
        const response = await fetch(`/classification?ticker=${encodeURIComponent(ticker)}`);
        const result = await response.json();
        if (document.getElementById("candidate-ticker").value.trim().toUpperCase() !== ticker.toUpperCase()) return;
        const group = response.ok ? result.sector_group : "Unclassified";
        section.dataset.sectorGroup = group;
        section.dataset.gicsSector = result.gics_sector || "Unknown";
        sector.textContent = group === result.gics_sector ? group : `${group} · ${result.gics_sector || "Unknown"}`;
      } catch (error) {
        section.dataset.sectorGroup = "Unclassified";
        section.dataset.gicsSector = "Unknown";
        sector.textContent = "Unclassified";
      }
      if (fundingMode === "reduce") updateFundingSummary();
      else scheduleAnalysisRefresh();
    }

    function hideCandidateSuggestions() {
      document.getElementById("candidate-suggestions").hidden = true;
    }

    function setFundingMode(mode) {
      fundingMode = mode;
      document.getElementById("funding-cash").setAttribute("aria-pressed", String(mode === "cash"));
      document.getElementById("funding-reduce").setAttribute("aria-pressed", String(mode === "reduce"));
      document.getElementById("reduce-funding").hidden = mode !== "reduce";
      document.getElementById("automatic-reduction-option").hidden = mode !== "reduce";
      document.getElementById("reduction-scope-control").hidden = mode !== "reduce";
      updateFundingSummary();
    }

    function setCashSource(currency) {
      cashSource = currency;
      ["cad", "usd"].forEach((name) => {
        document.getElementById(`cash-source-${name}`).setAttribute(
          "aria-pressed", String(name === currency)
        );
      });
      updateFundingSummary();
    }

    function setReductionScope(scope) {
      reductionScope = scope;
      ["all", "sector"].forEach((name) => {
        document.getElementById(`reduction-scope-${name}`).setAttribute(
          "aria-pressed", String(name === scope)
        );
      });
      updateFundingSummary();
    }

    function setOptimizeReductions(enabled) {
      optimizeReductions = enabled;
      updateFundingSummary();
    }

    function reductionHoldingStatuses() {
      const candidateGroup = document.getElementById("candidate-section").dataset.sectorGroup || "";
      const holdings = activeRows().map((holding) => {
        const capacity = Math.max(0, Number(holding.weight || 0) - 1);
        let eligible = true;
        let reason = "";
        if (fundingMode !== "reduce") {
          eligible = false;
          reason = "Positions are not selected as the funding source.";
        } else if (capacity <= 0) {
          eligible = false;
          reason = "Already at the 1.00% minimum weight.";
        } else if (reductionScope === "sector" && !candidateGroup) {
          eligible = false;
          reason = "Waiting for the candidate sector.";
        } else if (reductionScope === "sector" && holding.sector_group !== candidateGroup) {
          eligible = false;
          reason = `Outside the ${candidateGroup} reduction scope.`;
        }
        return { ...holding, capacity, eligible, reason };
      });
      const eligible = holdings.filter((holding) => holding.eligible);
      const key = `${fundingMode}|${reductionScope}|${candidateGroup}|${eligible.map((holding) => holding.ticker).sort().join(",")}`;
      if (key !== reductionEligibilityKey) {
        selectedReductionTickers = new Set(eligible.map((holding) => holding.ticker));
        reductionEligibilityKey = key;
      }
      return holdings;
    }

    function updateFundingSummary() {
      const summary = document.getElementById("funding-summary");
      const cashMessage = document.getElementById("cash-funding");
      const cash = readCashWeights();
      cashMessage.textContent = `Available: CAD Cash ${cash.cad.toFixed(2)}% · USD Cash ${cash.usd.toFixed(2)}%. ` +
        "The optimizer will determine the candidate weight.";
      cashMessage.classList.remove("warning");

      const list = document.getElementById("reduction-list");
      list.replaceChildren();
      if (fundingMode !== "reduce") {
        summary.textContent = "";
        summary.classList.remove("complete", "warning");
        scheduleAnalysisRefresh();
        return;
      }

      const holdings = reductionHoldingStatuses();
      holdings.forEach((holding) => {
        const row = document.createElement("div");
        row.className = `reduction-row${optimizeReductions || fundingMode !== "reduce" ? "" : " manual"}${holding.eligible ? "" : " ineligible"}`;
        if (!optimizeReductions && fundingMode === "reduce") {
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.className = "reduction-checkbox";
          checkbox.disabled = !holding.eligible;
          checkbox.checked = holding.eligible && selectedReductionTickers.has(holding.ticker);
          checkbox.setAttribute("aria-label", `Allow reduction of ${holding.company_name || holding.ticker}`);
          checkbox.addEventListener("change", () => {
            if (checkbox.checked) selectedReductionTickers.add(holding.ticker);
            else selectedReductionTickers.delete(holding.ticker);
            updateFundingSummary();
          });
          row.appendChild(checkbox);
        }
        const security = document.createElement("div");
        security.className = "reduction-security";
        const name = document.createElement("strong");
        name.textContent = holding.company_name || holding.ticker;
        const meta = document.createElement("span");
        meta.textContent = holding.eligible
          ? `${holding.ticker} · up to ${holding.capacity.toFixed(2)}% reducible`
          : `${holding.ticker} · ${holding.reason}`;
        security.append(name, meta);
        const status = document.createElement("span");
        status.className = `reduction-amount${holding.eligible ? "" : " ineligible"}`;
        status.textContent = holding.eligible
          ? (optimizeReductions ? "Optimizer Eligible" : (selectedReductionTickers.has(holding.ticker) ? "Selected" : "Not selected"))
          : "Optimizer Ineligible";
        row.append(security, status);
        list.appendChild(row);
      });

      [["cad", "CAD Cash"], ["usd", "USD Cash"]].forEach(([currency, label]) => {
        const available = cash[currency];
        const row = document.createElement("div");
        row.className = `reduction-row${available > 0 ? "" : " ineligible"}`;
        const security = document.createElement("div");
        security.className = "reduction-security";
        const name = document.createElement("strong");
        name.textContent = label;
        const meta = document.createElement("span");
        meta.textContent = `${available.toFixed(2)}% available`;
        security.append(name, meta);
        const status = document.createElement("span");
        status.className = `reduction-amount${available > 0 ? "" : " ineligible"}`;
        status.textContent = available <= 0
          ? "Optimizer Ineligible"
          : (currency === cashSource ? "Primary Cash" : "Fallback Cash");
        row.append(security, status);
        list.appendChild(row);
      });

      const eligibleCount = holdings.filter((holding) => holding.eligible).length;
      const selectedCount = holdings.filter(
        (holding) => holding.eligible && selectedReductionTickers.has(holding.ticker)
      ).length;
      if (optimizeReductions) {
        summary.textContent = `${eligibleCount} holding${eligibleCount === 1 ? "" : "s"} available to the optimizer; cash will fund any remainder.`;
      } else {
        summary.textContent = `${selectedCount} eligible holding${selectedCount === 1 ? "" : "s"} selected; cash will fund any remainder.`;
      }
      summary.classList.remove("complete", "warning");
      scheduleAnalysisRefresh();
    }

    function readCandidate() {
      return {
        company_name: document.getElementById("candidate-company").value.trim(),
        ticker: document.getElementById("candidate-ticker").value.trim(),
        gics_sector: document.getElementById("candidate-section").dataset.gicsSector || "",
        sector_group: document.getElementById("candidate-section").dataset.sectorGroup || "",
        funding_method: fundingMode,
        reduction_scope: reductionScope,
        optimize_reductions: optimizeReductions,
        selected_reductions: [...selectedReductionTickers],
        cash_source: cashSource
      };
    }

    function readCashWeights() {
      return {
        cad: Number(document.getElementById("cad-cash-weight")?.value || 0),
        usd: Number(document.getElementById("usd-cash-weight")?.value || 0)
      };
    }

    function setOptimizeMetric(metric) {
      const changed = optimizeMetric !== metric;
      optimizeMetric = metric;
      ["return", "volatility", "sharpe"].forEach((name) => {
        document.getElementById(`optimize-${name}`).setAttribute(
          "aria-pressed", String(name === metric)
        );
      });
      if (changed) scheduleAnalysisRefresh();
    }

    function formatMetric(value, kind) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
      return kind === "sharpe" ? Number(value).toFixed(3) : `${Number(value).toFixed(2)}%`;
    }

    function formatDifference(before, after, kind) {
      const difference = Number(after) - Number(before);
      const sign = difference > 0 ? "+" : "";
      return kind === "sharpe"
        ? `${sign}${difference.toFixed(3)}`
        : `${sign}${difference.toFixed(2)} pp`;
    }

    function optimizationObjectiveLabel() {
      if (optimizeMetric === "return") return "Maximize return";
      if (optimizeMetric === "volatility") return "Minimize volatility";
      return "Maximize Sharpe ratio";
    }

    function renderSensitivityTable() {
      const body = document.getElementById("sensitivity-body");
      body.replaceChildren();
      document.getElementById("sensitivity-objective").textContent = optimizationObjectiveLabel();
      ["6M", "1Y", "3Y", "5Y"].forEach((windowName) => {
        const result = analysisRows.find((item) => item.window === windowName);
        const row = document.createElement("tr");
        row.dataset.window = windowName;
        const values = result
          ? [
              windowName,
              `${Number(result.candidate_weight).toFixed(2)}%`,
              formatMetric(result.optimized.return, "return"),
              formatMetric(result.optimized.volatility, "volatility"),
              formatMetric(result.optimized.sharpe, "sharpe")
            ]
          : [windowName, "Unavailable", "-", "-", "-"];
        values.forEach((value) => {
          const cell = document.createElement("td");
          cell.textContent = value;
          row.appendChild(cell);
        });
        body.appendChild(row);
      });

      const weights = analysisRows
        .map((row) => Number(row.candidate_weight))
        .filter((weight) => Number.isFinite(weight));
      const spread = weights.length ? Math.max(...weights) - Math.min(...weights) : 0;
      const verySensitive = weights.length > 1 && spread >= 2;
      const label = document.getElementById("sensitivity-label");
      label.classList.toggle("very-sensitive", verySensitive);
      label.replaceChildren();
      const result = document.createElement("strong");
      result.textContent = verySensitive ? "Very Sensitive" : "Not Sensitive";
      const detail = document.createElement("span");
      detail.textContent = `${spread.toFixed(2)} percentage-point range across available windows.`;
      label.append(result, detail);
    }

    function setAnalysisWindow(windowName) {
      analysisWindow = windowName;
      document.querySelectorAll("#analysis-window-selector .funding-option").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.window === windowName));
      });
      const row = analysisRows.find((item) => item.window === windowName);
      if (!row) return;
      document.querySelectorAll("#sensitivity-body tr").forEach((tableRow) => {
        tableRow.classList.toggle("selected", tableRow.dataset.window === windowName);
      });

      const metrics = [
        ["Return", "return"],
        ["Volatility", "volatility"],
        ["Sharpe Ratio", "sharpe"]
      ];
      const grid = document.getElementById("metric-grid");
      grid.replaceChildren();
      metrics.forEach(([label, key]) => {
        const block = document.createElement("section");
        block.className = "metric-block";
        const heading = document.createElement("h3");
        heading.textContent = label;
        block.appendChild(heading);
        [["Before", formatMetric(row.current[key], key)],
         ["After", formatMetric(row.optimized[key], key)],
         ["Difference", formatDifference(row.current[key], row.optimized[key], key)]
        ].forEach(([name, value], index) => {
          const line = document.createElement("div");
          line.className = `metric-line${index === 2 ? " difference" : ""}`;
          const caption = document.createElement("span");
          caption.textContent = name;
          const amount = document.createElement("strong");
          amount.textContent = value;
          line.append(caption, amount);
          block.appendChild(line);
        });
        grid.appendChild(block);
      });

      const funding = document.getElementById("analysis-funding");
      funding.replaceChildren();
      const title = document.createElement("strong");
      title.textContent = `${row.candidate_weight.toFixed(2)}% optimized candidate weight`;
      funding.appendChild(title);
      if ((row.reductions || []).length) {
        const scope = document.createElement("p");
        const scopeLabel = row.reduction_scope === "sector"
          ? "the candidate sector"
          : "all eligible portfolio positions";
        scope.textContent = row.optimize_reductions
          ? `The optimizer selected reductions from ${scopeLabel}.`
          : `Reductions use only the checked holdings within ${scopeLabel}.`;
        funding.appendChild(scope);
        row.reductions.forEach((reduction) => {
          const line = document.createElement("p");
          line.textContent = `${reduction.company_name} (${reduction.ticker}) reduced by ${Number(reduction.weight).toFixed(2)}%.`;
          funding.appendChild(line);
        });
      } else {
        const line = document.createElement("p");
        line.textContent = "No existing stock positions are reduced.";
        funding.appendChild(line);
      }
      const cash = document.createElement("p");
      cash.textContent = `CAD Cash ${Number(row.cash_used.cad).toFixed(2)}% · USD Cash ${Number(row.cash_used.usd).toFixed(2)}%`;
      funding.appendChild(cash);
      if (fundingMode === "cash") {
        document.getElementById("cash-funding").textContent =
          `Cash used: CAD Cash ${Number(row.cash_used.cad).toFixed(2)}% · USD Cash ${Number(row.cash_used.usd).toFixed(2)}%`;
      }
    }

    function renderAnalysis(result) {
      const container = document.getElementById("analysis-results");
      analysisRows = result.results || [];
      if (!analysisRows.length) {
        container.hidden = true;
        return;
      }

      const selector = document.getElementById("analysis-window-selector");
      selector.replaceChildren();
      ["6M", "1Y", "3Y", "5Y"].forEach((windowName) => {
        const row = analysisRows.find((item) => item.window === windowName);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "funding-option";
        button.dataset.window = windowName;
        button.textContent = windowName;
        button.disabled = !row;
        button.addEventListener("click", () => setAnalysisWindow(windowName));
        selector.appendChild(button);
      });
      if (!analysisRows.some((row) => row.window === analysisWindow)) {
        analysisWindow = analysisRows[0].window;
      }
      container.hidden = false;
      renderSensitivityTable();
      setAnalysisWindow(analysisWindow);
    }

    function clearAnalysisResults() {
      analysisRows = [];
      document.getElementById("analysis-results").hidden = true;
    }

    function scheduleAnalysisRefresh() {
      if (!portfolioCreated) return;
      clearTimeout(analysisRefreshTimer);
      const revision = ++analysisRevision;
      const message = document.getElementById("analysis-message");
      const button = document.getElementById("calculate-metrics");
      clearAnalysisResults();
      button.disabled = false;

      const total = currentPortfolioWeight();
      message.className = "analysis-message";
      if (Math.abs(total - 100) > 0.005) {
        message.textContent = `Portfolio must total 100.00% before metrics and optimizer results can be calculated. Current total: ${total.toFixed(2)}%.`;
        message.classList.add("warning");
        return;
      }

      const holdingsReady = activeRows().length > 0 && activeRows().every((holding) =>
        holding.company_name.trim()
        && holding.ticker.trim()
        && holding.sector_group
        && Number(holding.weight) >= 1
        && Number(holding.weight) <= 6
        && holding.purchase_date
      );
      const candidate = readCandidate();
      const candidateReady = candidate.ticker && candidate.sector_group;
      if (!holdingsReady || !candidateReady) {
        message.textContent = !holdingsReady
          ? "Complete all current portfolio security fields before calculating metrics."
          : "Complete the candidate security before calculating metrics.";
        message.classList.add("warning");
        return;
      }

      message.textContent = "Updating metrics...";
      analysisRefreshTimer = setTimeout(() => runAnalysis(revision), 650);
    }

    async function runAnalysis(requestRevision = null) {
      if (requestRevision === null) {
        clearTimeout(analysisRefreshTimer);
        requestRevision = ++analysisRevision;
      } else if (requestRevision !== analysisRevision) {
        return;
      }
      const message = document.getElementById("analysis-message");
      const results = document.getElementById("analysis-results");
      const button = document.getElementById("calculate-metrics");
      const holdings = activeRows();
      const cashWeights = readCashWeights();
      const total = holdings.reduce((sum, holding) => sum + Number(holding.weight || 0), 0) + cashWeights.cad + cashWeights.usd;

      message.className = "analysis-message";
      if (Math.abs(total - 100) > 0.005) {
        clearAnalysisResults();
        message.textContent = `Portfolio must total 100.00% before metrics and optimizer results can be calculated. Current total: ${total.toFixed(2)}%.`;
        message.classList.add("warning");
        return;
      }

      button.disabled = true;
      message.textContent = "Calculating historical metrics...";
      clearAnalysisResults();
      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            holdings,
            cad_cash_weight: cashWeights.cad,
            usd_cash_weight: cashWeights.usd,
            candidate: readCandidate(),
            optimize_metric: optimizeMetric
          })
        });
        const result = await response.json();
        if (requestRevision !== analysisRevision) return;
        const messages = [...(result.errors || []), ...(result.warnings || [])];
        message.textContent = messages.join(" ");
        message.classList.toggle("warning", messages.length > 0);
        renderAnalysis(result);
      } catch (error) {
        if (requestRevision !== analysisRevision) return;
        message.textContent = "Metrics could not be calculated. Check the connection and try again.";
        message.classList.add("warning");
      } finally {
        if (requestRevision === analysisRevision) button.disabled = false;
      }
    }

    function hideSuggestions(row) {
      row.querySelector(".suggestions").hidden = true;
    }

    function closeSuggestions() {
      document.querySelectorAll(".suggestions").forEach((list) => { list.hidden = true; });
    }

    function closeCalendars(exceptRow = null) {
      document.querySelectorAll(".holding-row, .draft-row").forEach((row) => {
        if (row !== exceptRow) row.querySelector(".calendar-popover").hidden = true;
      });
    }

    function resetPrice(row) {
      row.querySelector(".price-value").textContent = "-";
      row.querySelector(".price-note").textContent = "";
    }

    function formatWeight(input) {
      if (input.value === "") return;
      const value = Number(input.value);
      if (Number.isFinite(value)) input.value = value.toFixed(2);
      const row = input.closest("tr");
      if (row?.classList.contains("draft-row")) updateDraftValidation(row);
      else updatePortfolioSummary();
    }

    async function openCalendar(row) {
      if (row.classList.contains("holding-row") && !row.classList.contains("editing-row")) return;
      const ticker = row.querySelector(".ticker-input").value.trim();
      if (!ticker) return;

      closeSuggestions();
      closeCalendars(row);
      const selected = row.querySelector(".date-input").value;
      const initial = selected ? new Date(`${selected}T12:00:00`) : new Date(`${today}T12:00:00`);
      row.dataset.calendarYear = String(initial.getFullYear());
      row.dataset.calendarMonth = String(initial.getMonth() + 1);
      await renderCalendar(row);
    }

    async function renderCalendar(row) {
      const ticker = row.querySelector(".ticker-input").value.trim();
      const year = Number(row.dataset.calendarYear);
      const month = Number(row.dataset.calendarMonth);
      const popover = row.querySelector(".calendar-popover");
      popover.hidden = false;
      popover.textContent = "Loading...";

      try {
        const response = await fetch(`/trading-days?ticker=${encodeURIComponent(ticker)}&year=${year}&month=${month}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Calendar unavailable");
        if (Number(row.dataset.calendarYear) !== year || Number(row.dataset.calendarMonth) !== month) return;
        row.dataset.listingCutoff = result.cutoff;

        const validDays = new Set(result.days);
        const selected = row.querySelector(".date-input").value;
        const monthName = new Intl.DateTimeFormat("en-CA", { month: "long", year: "numeric" })
          .format(new Date(year, month - 1, 1));
        const current = new Date(`${today}T12:00:00`);
        const atCurrentMonth = year === current.getFullYear() && month === current.getMonth() + 1;
        const cutoff = new Date(`${result.cutoff}T12:00:00`);
        const atCutoffMonth = year === cutoff.getFullYear() && month === cutoff.getMonth() + 1;

        popover.replaceChildren();
        const header = document.createElement("div");
        header.className = "calendar-header";
        const previous = calendarNav("‹", "Previous month", () => changeCalendarMonth(row, -1));
        previous.disabled = atCutoffMonth;
        const title = document.createElement("button");
        title.type = "button";
        title.className = "calendar-title";
        title.textContent = monthName;
        title.setAttribute("aria-label", `Choose month and year, currently ${monthName}`);
        title.addEventListener("click", (event) => {
          event.stopPropagation();
          row.dataset.calendarDecade = String(Math.floor(year / 10) * 10);
          renderMonthYearSelector(row);
        });
        const next = calendarNav("›", "Next month", () => changeCalendarMonth(row, 1));
        next.disabled = atCurrentMonth;
        header.append(previous, title, next);

        const grid = document.createElement("div");
        grid.className = "calendar-grid";
        ["S", "M", "T", "W", "T", "F", "S"].forEach((label) => {
          const weekday = document.createElement("div");
          weekday.className = "calendar-weekday";
          weekday.textContent = label;
          grid.appendChild(weekday);
        });

        const firstWeekday = new Date(year, month - 1, 1).getDay();
        const daysInMonth = new Date(year, month, 0).getDate();
        for (let blank = 0; blank < firstWeekday; blank += 1) {
          const spacer = document.createElement("div");
          spacer.className = "calendar-empty";
          grid.appendChild(spacer);
        }

        for (let day = 1; day <= daysInMonth; day += 1) {
          const isoDate = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
          const button = document.createElement("button");
          button.type = "button";
          button.className = "calendar-day";
          button.textContent = String(day);
          button.disabled = !validDays.has(isoDate);
          if (isoDate === selected) button.classList.add("selected");
          button.addEventListener("click", () => selectPurchaseDate(row, isoDate));
          grid.appendChild(button);
        }

        popover.append(header, grid);
      } catch (error) {
        popover.textContent = "Trading calendar unavailable.";
      }
    }

    function renderMonthYearSelector(row) {
      const popover = row.querySelector(".calendar-popover");
      const selectedYear = Number(row.dataset.calendarYear);
      const selectedMonth = Number(row.dataset.calendarMonth);
      const current = new Date(`${today}T12:00:00`);
      const currentYear = current.getFullYear();
      const currentMonth = current.getMonth() + 1;
      const cutoff = new Date(`${row.dataset.listingCutoff}T12:00:00`);
      const cutoffYear = cutoff.getFullYear();
      const cutoffMonth = cutoff.getMonth() + 1;
      const currentDecade = Math.floor(currentYear / 10) * 10;
      const cutoffDecade = Math.floor(cutoffYear / 10) * 10;
      const decadeStart = Number(row.dataset.calendarDecade || Math.floor(selectedYear / 10) * 10);
      row.dataset.calendarDecade = String(decadeStart);

      popover.replaceChildren();

      const header = document.createElement("div");
      header.className = "month-selector-header";
      const previousDecade = calendarNav("‹", "Previous decade", () => changeCalendarDecade(row, -10));
      previousDecade.disabled = decadeStart <= cutoffDecade;
      const range = document.createElement("div");
      range.className = "selector-range";
      range.textContent = `${decadeStart}–${decadeStart + 9}`;
      const nextDecade = calendarNav("›", "Next decade", () => changeCalendarDecade(row, 10));
      nextDecade.disabled = decadeStart >= currentDecade;
      header.append(previousDecade, range, nextDecade);

      const yearsLabel = document.createElement("div");
      yearsLabel.className = "selector-section-label";
      yearsLabel.textContent = "Year";
      const years = document.createElement("div");
      years.className = "year-selector-grid";
      for (let year = decadeStart; year <= decadeStart + 9; year += 1) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "year-option";
        button.textContent = String(year);
        button.disabled = year > currentYear || year < cutoffYear;
        if (year === selectedYear) button.classList.add("selected");
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          row.dataset.calendarYear = String(year);
          renderMonthYearSelector(row);
        });
        years.appendChild(button);
      }

      const monthsLabel = document.createElement("div");
      monthsLabel.className = "selector-section-label";
      monthsLabel.textContent = "Month";
      const grid = document.createElement("div");
      grid.className = "month-selector-grid";
      const monthLabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      monthLabels.forEach((label, index) => {
        const month = index + 1;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "month-option";
        button.textContent = label;
        const beforeListing = selectedYear < cutoffYear || (selectedYear === cutoffYear && month < cutoffMonth);
        const afterCurrent = selectedYear > currentYear || (selectedYear === currentYear && month > currentMonth);
        button.disabled = beforeListing || afterCurrent;
        if (selectedYear === Number(row.dataset.calendarYear) && month === selectedMonth) {
          button.classList.add("selected");
        }
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          row.dataset.calendarMonth = String(month);
          renderCalendar(row);
        });
        grid.appendChild(button);
      });

      popover.append(header, yearsLabel, years, monthsLabel, grid);
    }

    function calendarNav(label, ariaLabel, action) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "calendar-nav";
      button.textContent = label;
      button.setAttribute("aria-label", ariaLabel);
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        action();
      });
      return button;
    }

    function changeCalendarDecade(row, amount) {
      row.dataset.calendarDecade = String(Number(row.dataset.calendarDecade) + amount);
      renderMonthYearSelector(row);
    }

    function changeCalendarMonth(row, amount) {
      const current = new Date(Number(row.dataset.calendarYear), Number(row.dataset.calendarMonth) - 1 + amount, 1);
      row.dataset.calendarYear = String(current.getFullYear());
      row.dataset.calendarMonth = String(current.getMonth() + 1);
      renderCalendar(row);
    }

    function selectPurchaseDate(row, isoDate) {
      row.querySelector(".date-input").value = isoDate;
      row.querySelector(".calendar-popover").hidden = true;
      updateDraftValidation(row);
      if (row.classList.contains("holding-row")) scheduleAnalysisRefresh();
      fetchPurchaseClose(row);
    }

    async function fetchPurchaseClose(row) {
      const ticker = row.querySelector(".ticker-input").value.trim();
      const selectedDate = row.querySelector(".date-input").value;
      const value = row.querySelector(".price-value");
      const note = row.querySelector(".price-note");
      value.textContent = "...";
      note.textContent = "";

      try {
        const response = await fetch(`/price?ticker=${encodeURIComponent(ticker)}&date=${encodeURIComponent(selectedDate)}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Price unavailable");
        value.textContent = `$${Number(result.close).toFixed(2)}`;
        note.textContent = result.used_previous_close ? `Previous close: ${result.price_date}` : "";
      } catch (error) {
        value.textContent = "-";
        note.textContent = "Unavailable";
      } finally {
        if (row.classList.contains("holding-row")) scheduleAnalysisRefresh();
      }
    }

    function removeRow(button) {
      const row = button.closest("tr");
      if (row.classList.contains("draft-row") && row.nextElementSibling?.classList.contains("draft-actions-row")) {
        row.nextElementSibling.remove();
      }
      row.remove();
      organizeHoldingRows();
      updatePortfolioSummary();
    }

    function readRows() {
      return [...document.querySelectorAll(".holding-row")].map((tr) => ({
        company_name: tr.querySelector(".company-input").value,
        ticker: tr.querySelector(".ticker-input").value,
        gics_sector: tr.dataset.gicsSector || "",
        sector_group: tr.dataset.sectorGroup || "",
        weight: tr.querySelector(".weight-input").value,
        purchase_date: tr.querySelector(".date-input").value
      }));
    }

    function activeRows() {
      return readRows().filter((row) => row.company_name.trim() || row.ticker.trim());
    }

    function currentPortfolioWeight() {
      const cash = readCashWeights();
      return activeRows().reduce(
        (total, row) => total + Number(row.weight || 0),
        cash.cad + cash.usd
      );
    }

    function updatePortfolioSummary() {
      const rows = activeRows();
      const cash = readCashWeights();
      const totalWeight = rows.reduce((total, row) => total + Number(row.weight || 0), 0) + cash.cad + cash.usd;
      const countLabel = rows.length === 1 ? "1 holding" : `${rows.length} holdings`;
      document.getElementById("portfolio-row-weight").textContent = `${totalWeight.toFixed(2)}%`;
      document.getElementById("portfolio-row-count").textContent = countLabel;
      const atLimit = totalWeight >= 100 - 0.005;
      document.querySelector("#holdings-table tfoot").hidden =
        !portfolioCreated || !isExpanded || atLimit || Boolean(document.querySelector(".draft-row"))
        || Boolean(document.querySelector(".editing-row"));
      document.getElementById("add-limit-message").hidden = !portfolioCreated || !atLimit;
      updateFundingSummary();
    }

    document.addEventListener("click", (event) => {
      if (!event.target.closest(".company-cell")) closeSuggestions();
      if (!event.target.closest(".candidate-company")) hideCandidateSuggestions();
      if (!event.target.closest(".date-cell")) closeCalendars();
    });
    document.getElementById("portfolio-name").addEventListener("input", scheduleAnalysisRefresh);
    bindCandidateInput();
    setExpanded(true);
  </script>
</body>
</html>
""".replace("__TODAY__", date.today().isoformat())


class PortfolioInputHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/classification":
            ticker = parse_qs(parsed_path.query).get("ticker", [""])[0].strip().upper()
            try:
                if not ticker:
                    raise ValueError("Ticker is required.")
                result = classify_security(ticker)
                status = 200
            except (TypeError, ValueError, RuntimeError) as error:
                result = {"error": str(error)}
                status = 400
            except Exception as error:
                result = {"error": f"Sector lookup failed: {error}"}
                status = 502

            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "private, max-age=86400")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed_path.path == "/trading-days":
            params = parse_qs(parsed_path.query)
            try:
                ticker = params.get("ticker", [""])[0].strip().upper()
                year = int(params.get("year", [""])[0])
                month = int(params.get("month", [""])[0])
                if not ticker or month < 1 or month > 12:
                    raise ValueError("Ticker, year, and month are required.")
                result = {
                    "ticker": ticker,
                    "calendar": calendar_name_for_ticker(ticker),
                    "cutoff": earliest_available_price_date(ticker).isoformat(),
                    "days": trading_days_for_month(ticker, year, month),
                }
                status = 200
            except (TypeError, ValueError, RuntimeError) as error:
                result = {"error": str(error)}
                status = 400

            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed_path.path == "/price":
            params = parse_qs(parsed_path.query)
            try:
                ticker = params.get("ticker", [""])[0].strip().upper()
                selected_date = params.get("date", [""])[0].strip()
                if not ticker or not selected_date or yf is None:
                    raise ValueError("Ticker and purchase date are required.")
                result = purchase_date_close(ticker, selected_date)
                status = 200
            except (TypeError, ValueError, RuntimeError) as error:
                result = {"error": str(error)}
                status = 400
            except Exception as error:
                result = {"error": f"Yahoo Finance lookup failed: {error}"}
                status = 502

            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            cache_control = (
                "no-store"
                if selected_date == date.today().isoformat()
                else "private, max-age=300"
            )
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed_path.path == "/search":
            query = parse_qs(parsed_path.query).get("q", [""])[0]
            result = search_securities(query)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if parsed_path.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path not in {"/prepare", "/analyze"}:
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body)
            result = (
                analyze_portfolio(payload)
                if parsed_path.path == "/analyze"
                else prepare_portfolio(payload)
            )
            self.send_response(200)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            result = {"errors": [f"Could not process portfolio input: {error}"], "results": []}
            self.send_response(400)
        except Exception as error:
            result = {"errors": [f"Historical analysis failed: {error}"], "results": []}
            self.send_response(502)

        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


def run_server() -> None:
    server = ThreadingHTTPServer((HOST, PORT), PortfolioInputHandler)
    print(f"Portfolio Input is running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


def streamlit_runtime_active() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def run_streamlit_native_app() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="FitCheck", page_icon="FC", layout="wide")
    st.markdown(
        """
        <style>
          .stAppHeader { display: none; }
          .fitcheck-header { border-bottom: 1px solid #dfe3e8; margin: -2.5rem -5rem 1.75rem; padding: 1rem 5rem; }
          .fitcheck-brand { color: #7b263d; font-size: 1.25rem; font-weight: 750; }
          [data-testid="stMetric"] { border: 1px solid #dfe3e8; padding: 1rem; border-radius: 7px; }
          .eligibility-row { border-bottom: 1px solid #e8ebee; padding: .55rem 0; }
          .ineligible { color: #717784; font-weight: 650; }
          @media (max-width: 700px) {
            .fitcheck-header { margin: -2rem -1rem 1.25rem; padding: .9rem 1rem; }
          }
        </style>
        <div class="fitcheck-header"><span class="fitcheck-brand">FitCheck</span></div>
        """,
        unsafe_allow_html=True,
    )

    defaults = {
        "portfolio_created": False,
        "portfolio_name": "",
        "holdings": [],
        "cad_cash": 0.0,
        "usd_cash": 0.0,
        "candidate": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.subheader("Current Portfolio")
    if not st.session_state.portfolio_created:
        with st.form("create_portfolio"):
            name = st.text_input("Portfolio name", placeholder="Enter portfolio name")
            submitted = st.form_submit_button("Add portfolio", use_container_width=True)
        if submitted:
            if name.strip():
                st.session_state.portfolio_name = name.strip()
                st.session_state.portfolio_created = True
                st.rerun()
            else:
                st.warning("Enter a portfolio name.")
        st.stop()

    st.session_state.portfolio_name = st.text_input(
        "Portfolio name",
        value=st.session_state.portfolio_name,
        placeholder="Add portfolio name",
        label_visibility="collapsed",
    )

    cash_left, cash_right = st.columns(2)
    with cash_left:
        st.session_state.cad_cash = st.number_input(
            "CAD Cash (%)", min_value=0.0, max_value=100.0,
            value=float(st.session_state.cad_cash), step=0.25, format="%.2f",
        )
    with cash_right:
        st.session_state.usd_cash = st.number_input(
            "USD Cash (%)", min_value=0.0, max_value=100.0,
            value=float(st.session_state.usd_cash), step=0.25, format="%.2f",
        )

    holdings = st.session_state.holdings
    for group in ["Materials", "FIGs", "TMTH", "Consumers", "Infrastructure", "Industrials", "Unclassified"]:
        group_holdings = sorted(
            [holding for holding in holdings if holding["sector_group"] == group],
            key=lambda holding: holding["company_name"].casefold(),
        )
        if not group_holdings:
            continue
        st.caption(group.upper())
        display_rows = [
            {
                "Company Name": holding["company_name"],
                "Ticker": holding["ticker"],
                "Portfolio Weight (%)": f'{holding["weight"]:.2f}%',
                "Purchase Date": holding["purchase_date"],
                "Price": f'${holding["price"]:.2f}',
            }
            for holding in group_holdings
        ]
        st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)

    if holdings:
        with st.expander("Edit current securities"):
            for index, holding in enumerate(list(holdings)):
                cols = st.columns([3, 1.2, 1.6, .7])
                cols[0].write(f'**{holding["company_name"]}**  \n{holding["ticker"]}')
                new_weight = cols[1].number_input(
                    "Weight", min_value=1.0, max_value=6.0, value=float(holding["weight"]),
                    step=0.25, format="%.2f", key=f"edit_weight_{holding['ticker']}_{index}",
                )
                new_date = cols[2].date_input(
                    "Purchase date", value=date.fromisoformat(holding["purchase_date"]),
                    max_value=date.today(), key=f"edit_date_{holding['ticker']}_{index}",
                )
                if cols[3].button("Remove", key=f"remove_{holding['ticker']}_{index}"):
                    st.session_state.holdings.pop(index)
                    st.rerun()
                if new_weight != holding["weight"] or new_date.isoformat() != holding["purchase_date"]:
                    try:
                        price = purchase_date_close(holding["ticker"], new_date.isoformat())
                        holding["weight"] = float(new_weight)
                        holding["purchase_date"] = new_date.isoformat()
                        holding["price"] = price["close"]
                    except Exception as error:
                        st.warning(f'{holding["ticker"]}: {error}')

    total_weight = sum(float(holding["weight"]) for holding in holdings)
    total_weight += st.session_state.cad_cash + st.session_state.usd_cash
    st.caption(f"Current portfolio weight: {total_weight:.2f}%")

    if total_weight < 100 - 0.005:
        with st.expander("Add security", expanded=not holdings):
            query = st.text_input("Company name", placeholder="Search company", key="holding_search")
            results = search_securities(query) if len(query.strip()) >= 2 else []
            options = {f'{item["company_name"]} ({item["ticker"]})': item for item in results}
            selected_label = st.selectbox(
                "Matching company", [""] + list(options),
                format_func=lambda value: value or "Select a company",
            )
            add_left, add_right = st.columns(2)
            weight = add_left.number_input(
                "Weight (%)", min_value=1.0, max_value=6.0, value=1.0,
                step=0.25, format="%.2f", key="new_weight",
            )
            purchase_date = add_right.date_input(
                "Purchase date", value=date.today(), max_value=date.today(), key="new_purchase_date",
            )
            if st.button("Add to Portfolio", disabled=not selected_label):
                selected = options[selected_label]
                ticker = selected["ticker"]
                if ticker in {holding["ticker"] for holding in holdings}:
                    st.warning("That security is already in the portfolio.")
                elif total_weight + weight > 100 + 0.005:
                    st.warning("This holding would take the portfolio above 100.00%.")
                else:
                    try:
                        classification = classify_security(ticker)
                        price = purchase_date_close(ticker, purchase_date.isoformat())
                        holdings.append({
                            "company_name": selected["company_name"],
                            "ticker": ticker,
                            "gics_sector": classification["gics_sector"],
                            "sector_group": classification["sector_group"],
                            "weight": float(weight),
                            "purchase_date": purchase_date.isoformat(),
                            "price": price["close"],
                        })
                        st.rerun()
                    except Exception as error:
                        st.warning(str(error))
    else:
        st.caption("Portfolio is at 100.00%. Lower an existing weight to add another security.")

    st.divider()
    st.subheader("Candidate Portfolio")
    candidate_query = st.text_input("Candidate company", placeholder="Search company", key="candidate_search")
    candidate_results = search_securities(candidate_query) if len(candidate_query.strip()) >= 2 else []
    candidate_options = {
        f'{item["company_name"]} ({item["ticker"]})': item for item in candidate_results
        if item["ticker"] not in {holding["ticker"] for holding in holdings}
    }
    candidate_label = st.selectbox(
        "Matching candidate", [""] + list(candidate_options),
        format_func=lambda value: value or "Select a candidate",
    )
    if candidate_label:
        selected_candidate = candidate_options[candidate_label]
        previous_ticker = (st.session_state.candidate or {}).get("ticker")
        if previous_ticker != selected_candidate["ticker"]:
            try:
                classification = classify_security(selected_candidate["ticker"])
                st.session_state.candidate = {**selected_candidate, **classification}
            except Exception as error:
                st.warning(str(error))
                st.session_state.candidate = None

    candidate = st.session_state.candidate
    if candidate:
        st.caption(f'{candidate["ticker"]} · {candidate["sector_group"]} · {candidate["gics_sector"]}')
        funding_method = st.radio("Funding source", ["Available Cash", "Reduce Positions"], horizontal=True)
        cash_source = st.radio("Primary cash source", ["CAD Cash", "USD Cash"], horizontal=True)
        reduction_scope = "all"
        optimize_reductions = True
        selected_reductions = []
        if funding_method == "Reduce Positions":
            reduction_scope_label = st.radio(
                "Reduction scope", ["All Eligible Positions", "Candidate Sector Only"], horizontal=True,
            )
            reduction_scope = "sector" if reduction_scope_label == "Candidate Sector Only" else "all"
            optimize_reductions = st.checkbox("Optimize Reduction Automatically", value=True)

        st.caption("Funding eligibility")
        for holding in holdings:
            capacity = max(0.0, float(holding["weight"]) - 1)
            in_scope = reduction_scope == "all" or holding["sector_group"] == candidate["sector_group"]
            eligible = funding_method == "Reduce Positions" and capacity > 0 and in_scope
            reason = ""
            if funding_method != "Reduce Positions":
                reason = "Positions are not selected as the funding source."
            elif capacity <= 0:
                reason = "Already at the 1.00% minimum weight."
            elif not in_scope:
                reason = f'Outside the {candidate["sector_group"]} reduction scope.'
            if not optimize_reductions and funding_method == "Reduce Positions":
                checked = st.checkbox(
                    f'{holding["company_name"]} ({holding["ticker"]})',
                    value=eligible, disabled=not eligible, key=f"reduce_{holding['ticker']}",
                )
                if checked and eligible:
                    selected_reductions.append(holding["ticker"])
            else:
                status = "Optimizer Eligible" if eligible else "Optimizer Ineligible"
                detail = f"up to {capacity:.2f}% reducible" if eligible else reason
                st.markdown(f'**{holding["company_name"]} ({holding["ticker"]})** · {detail} · **{status}**')
        st.markdown(
            f'**CAD Cash** · {st.session_state.cad_cash:.2f}% available · '
            f'{"Primary Cash" if cash_source == "CAD Cash" else "Fallback Cash"}'
        )
        st.markdown(
            f'**USD Cash** · {st.session_state.usd_cash:.2f}% available · '
            f'{"Primary Cash" if cash_source == "USD Cash" else "Fallback Cash"}'
        )

        st.divider()
        st.subheader("Portfolio Metrics")
        objective_label = st.radio(
            "Optimize for", ["Return", "Volatility", "Sharpe Ratio"], horizontal=True,
        )
        window = st.radio("Historical window", ["6M", "1Y", "3Y", "5Y"], horizontal=True)
        if abs(total_weight - 100) > 0.005:
            st.warning(
                f"Portfolio must total 100.00% before metrics and optimizer results can be calculated. "
                f"Current total: {total_weight:.2f}%."
            )
        elif not holdings:
            st.warning("Add at least one current security before calculating metrics.")
        else:
            objective = {"Return": "return", "Volatility": "volatility", "Sharpe Ratio": "sharpe"}[objective_label]
            payload = {
                "holdings": holdings,
                "cad_cash_weight": st.session_state.cad_cash,
                "usd_cash_weight": st.session_state.usd_cash,
                "candidate": {
                    "company_name": candidate["company_name"],
                    "ticker": candidate["ticker"],
                    "gics_sector": candidate["gics_sector"],
                    "sector_group": candidate["sector_group"],
                    "funding_method": "reduce" if funding_method == "Reduce Positions" else "cash",
                    "reduction_scope": reduction_scope,
                    "optimize_reductions": optimize_reductions,
                    "selected_reductions": selected_reductions,
                    "cash_source": "cad" if cash_source == "CAD Cash" else "usd",
                },
                "optimize_metric": objective,
            }
            with st.spinner("Updating historical metrics..."):
                analysis = analyze_portfolio(payload)
            for error in analysis.get("errors", []):
                st.error(error)
            for warning in analysis.get("warnings", []):
                st.warning(warning)
            rows = analysis.get("results", [])
            selected_row = next((row for row in rows if row["window"] == window), None)
            if selected_row:
                metric_cols = st.columns(3)
                for column, key, label, suffix in zip(
                    metric_cols,
                    ["return", "volatility", "sharpe"],
                    ["Return", "Volatility", "Sharpe Ratio"],
                    ["%", "%", ""],
                ):
                    before = selected_row["current"][key]
                    after = selected_row["optimized"][key]
                    column.metric(label, f"{after:.2f}{suffix}", f"{after - before:+.2f}")
                    column.caption(f"Before: {before:.2f}{suffix}")
                st.caption(f'Optimized candidate weight: {selected_row["candidate_weight"]:.2f}%')
                reductions = selected_row.get("reductions", [])
                if reductions:
                    st.write("Reduced holdings: " + ", ".join(
                        f'{item["company_name"]} {item["weight"]:.2f}%' for item in reductions
                    ))
                cash_used = selected_row["cash_used"]
                st.write(f'CAD Cash {cash_used["cad"]:.2f}% · USD Cash {cash_used["usd"]:.2f}%')
                sensitivity = pd.DataFrame([
                    {
                        "Window": row["window"],
                        "Candidate Weight": f'{row["candidate_weight"]:.2f}%',
                        "Return": f'{row["optimized"]["return"]:.2f}%',
                        "Volatility": f'{row["optimized"]["volatility"]:.2f}%',
                        "Sharpe Ratio": f'{row["optimized"]["sharpe"]:.3f}',
                    }
                    for row in rows
                ])
                st.subheader("Sensitivity Analysis")
                st.dataframe(sensitivity, hide_index=True, use_container_width=True)
                weights = [row["candidate_weight"] for row in rows]
                label = "Very Sensitive" if weights and max(weights) - min(weights) >= 2 else "Not Sensitive"
                st.markdown(f"**{label}**")


STREAMLIT_COMPONENT_BRIDGE = r"""
  <script>
    (() => {
      const pendingRequests = new Map();
      const handledResponses = new Set();
      let requestCounter = 0;
      const browserFetch = window.fetch.bind(window);

      function postMessage(type, payload = {}) {
        window.parent.postMessage({ isStreamlitMessage: true, type, ...payload }, "*");
      }

      function updateFrameHeight() {
        const height = Math.max(
          document.documentElement.scrollHeight,
          document.body ? document.body.scrollHeight : 0
        );
        postMessage("streamlit:setFrameHeight", { height: height + 2 });
      }

      window.addEventListener("message", (event) => {
        if (event.data?.type !== "streamlit:render") return;
        const response = event.data.args?.response;
        if (response?.id && !handledResponses.has(response.id)) {
          handledResponses.add(response.id);
          const pending = pendingRequests.get(response.id);
          if (pending) {
            pendingRequests.delete(response.id);
            const body = JSON.stringify(response.data ?? {});
            pending.resolve({
              ok: response.status >= 200 && response.status < 300,
              status: response.status,
              json: async () => response.data,
              text: async () => body
            });
          }
        }
        window.setTimeout(updateFrameHeight, 0);
      });

      window.fetch = (resource, options = {}) => {
        const url = typeof resource === "string" ? resource : resource?.url;
        if (!url || !url.startsWith("/")) return browserFetch(resource, options);

        const id = `${Date.now()}-${++requestCounter}`;
        const request = {
          id,
          url,
          method: String(options.method || "GET").toUpperCase(),
          body: typeof options.body === "string" ? options.body : ""
        };
        const promise = new Promise((resolve, reject) => {
          pendingRequests.set(id, { resolve, reject });
        });
        postMessage("streamlit:setComponentValue", { value: request });
        return promise;
      };

      window.addEventListener("DOMContentLoaded", () => {
        const panel = document.getElementById("portfolio-inline-form");
        const trigger = document.getElementById("add-portfolio-trigger");
        if (panel) panel.hidden = true;
        trigger?.setAttribute("aria-expanded", "false");
        window.fitcheckUiState = { portfolioModalOpen: false };
        postMessage("streamlit:componentReady", { apiVersion: 1 });
        updateFrameHeight();
        if (document.body) {
          new ResizeObserver(updateFrameHeight).observe(document.body);
        }
      });
    })();
  </script>
"""


def streamlit_component_html() -> str:
    script_marker = "  <script>\n    const today"
    if script_marker not in HTML:
        raise RuntimeError("FitCheck component script marker was not found.")
    return HTML.replace(script_marker, STREAMLIT_COMPONENT_BRIDGE + script_marker, 1)


def handle_streamlit_request(request: dict) -> dict:
    request_id = str(request.get("id", ""))
    method = str(request.get("method", "GET")).upper()
    parsed_path = urlparse(str(request.get("url", "")))
    status = 200

    try:
        if method == "GET" and parsed_path.path == "/search":
            query = parse_qs(parsed_path.query).get("q", [""])[0]
            result = search_securities(query)
        elif method == "GET" and parsed_path.path == "/classification":
            ticker = parse_qs(parsed_path.query).get("ticker", [""])[0].strip().upper()
            if not ticker:
                raise ValueError("Ticker is required.")
            result = classify_security(ticker)
        elif method == "GET" and parsed_path.path == "/trading-days":
            params = parse_qs(parsed_path.query)
            ticker = params.get("ticker", [""])[0].strip().upper()
            year = int(params.get("year", [""])[0])
            month = int(params.get("month", [""])[0])
            if not ticker or month < 1 or month > 12:
                raise ValueError("Ticker, year, and month are required.")
            result = {
                "ticker": ticker,
                "calendar": calendar_name_for_ticker(ticker),
                "cutoff": earliest_available_price_date(ticker).isoformat(),
                "days": trading_days_for_month(ticker, year, month),
            }
        elif method == "GET" and parsed_path.path == "/price":
            params = parse_qs(parsed_path.query)
            ticker = params.get("ticker", [""])[0].strip().upper()
            selected_date = params.get("date", [""])[0].strip()
            if not ticker or not selected_date:
                raise ValueError("Ticker and purchase date are required.")
            result = purchase_date_close(ticker, selected_date)
        elif method == "POST" and parsed_path.path in {"/prepare", "/analyze"}:
            payload = json.loads(str(request.get("body", "{}")))
            result = (
                analyze_portfolio(payload)
                if parsed_path.path == "/analyze"
                else prepare_portfolio(payload)
            )
        else:
            status = 404
            result = {"error": "Unknown FitCheck request."}
    except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
        status = 400
        result = {"error": str(error)}
    except Exception as error:
        status = 502
        result = {"error": f"FitCheck data request failed: {error}"}

    return {"id": request_id, "status": status, "data": result}


def run_streamlit_app() -> None:
    import tempfile
    from pathlib import Path

    import streamlit as st
    from streamlit.components.v1 import declare_component

    st.set_page_config(page_title="FitCheck", layout="wide")
    if "fitcheck_modal_open" not in st.session_state:
        st.session_state.fitcheck_modal_open = False
    st.markdown(
        """
        <style>
          html, body, #root, .stApp,
          [data-testid="stApp"],
          [data-testid="stAppViewContainer"] {
            background: #fff !important;
          }
          header[data-testid="stHeader"], footer { display: none; }
          [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: none;
            padding: 0;
          }
          [data-testid="stCustomComponentV1"] {
            display: block;
            border: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
          }
          iframe[title="app.fitcheck_interface"] {
            border: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    component_directory = Path(tempfile.gettempdir()) / "fitcheck_streamlit_component"
    component_directory.mkdir(parents=True, exist_ok=True)
    component_index = component_directory / "index.html"
    component_source = streamlit_component_html()
    if not component_index.exists() or component_index.read_text(encoding="utf-8") != component_source:
        component_index.write_text(component_source, encoding="utf-8")

    fitcheck_component = declare_component("fitcheck_interface", path=component_directory)
    response = st.session_state.get("fitcheck_response")
    request = fitcheck_component(
        response=response,
        default=None,
        key="fitcheck-interface",
        height=900,
    )

    if isinstance(request, dict) and request.get("id"):
        request_id = str(request["id"])
        if request_id != st.session_state.get("fitcheck_processed_request"):
            st.session_state.fitcheck_processed_request = request_id
            st.session_state.fitcheck_response = handle_streamlit_request(request)
            st.rerun()


if __name__ == "__main__":
    if streamlit_runtime_active():
        run_streamlit_app()
    else:
        run_server()
