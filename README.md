# Leah Freqtrade Strategies

[![Freqtrade Compatibility](https://img.shields.io/badge/Freqtrade-2024.x%2B-blue.svg)](https://www.freqtrade.io/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Bederf/leah-freqtrade-strategies/pulls)

Production-grade, quantitatively audited strategies for [Freqtrade](https://www.freqtrade.io/).

Unlike traditional retail indicators (RSI/MACD crossovers or martingale grid bots) which suffer from severe regime decay and drawdown risk, these strategies are built on **evidence-first quantitative mechanics**:
* **Macro Regime Gating:** Automatic left-tail risk truncation when Bitcoin breaks below its macro trend.
* **Dual Momentum:** Cross-sectional relative strength combined with an absolute positive filter and benchmark fallback.
* **FreqAI Machine Learning:** Robust 15-feature volatility expansion classification free of lookahead bias.

---

## Performance Summary

### Strategy Comparison: Window 2 (143-Week Modern ETF Era: Jan 2024 – Oct 2026)

| Strategy Arm | Cumulative Return | Max Drawdown | Calmar Ratio | Sharpe Ratio | Mechanics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Buy-and-Hold BTC** | +95.91% | -50.83% | 0.54 | 0.75 | Unhedged beta exposure |
| **Leah_E015_Trend** | +54.00% | **-31.68%** | 0.54 | 0.62 | Long BTC when EMA30 > EMA90, else 100% Cash |
| **LeahClock4_A2** | +52.10% | -42.10% | 0.44 | 0.60 | Top-3 Alts + BTC Benchmark Fallback |
| **E014 Macro Standalone** | **+97.31%** | **-16.73%** | **1.68** | **1.56** | SPY / TLT / GLD 25w Momentum ($t = +1.708$) |
| **E015 + E014 Unified** | **+119.16%** | **-32.36%** | **1.02** | **0.96** | Hold BTC in Bull; rotate into E014 in Bear |

### 5.5-Year Full Cycle (298 Weeks: Feb 2021 – Oct 2026)

| Strategy Arm | Cumulative Return | Max Drawdown | Calmar Ratio | Sharpe Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Buy-and-Hold BTC** | +112.04% | -74.23% | 0.19 | 0.50 |
| **Leah_E015_Trend** | +117.52% | **-50.29%** | **0.29** | 0.53 |
| **E015 + E014 Unified** | **+179.61%** | **-51.90%** | **0.38** | **0.62** |

---

## Strategy Catalog

### 1. `Leah_E015_Trend.py` (Bitcoin Macro Trend Control)
* **Timeframe:** `1d`
* **Pair:** `BTC/USDT`
* **Concept:** Quantitative regime filter based on daily `EMA(30) > EMA(90)`.
* **Execution:**
  * When `EMA30 > EMA90`: Holds 100% BTC.
  * When `EMA30 <= EMA90`: Exits to 100% USDT Cash.
* **Why it works:** Prevents catastrophic drawdowns during multi-month crypto winters (slashed the 2022 crash from **-74%** down to **-50%**).

### 2. `LeahClock4_A2.py` (Dual Momentum + BTC Benchmark Fallback)
* **Timeframe:** `1d` (Weekly Friday rebalance)
* **Universe:** Top liquid alts (`ETH/USDT`, `SOL/USDT`, `LINK/USDT`, `BNB/USDT`, `AVAX/USDT`, `ADA/USDT`, `DOGE/USDT`, `XRP/USDT`) + `BTC/USDT`.
* **Concept:** 
  1. **Macro Gate:** If BTC `EMA30 <= EMA90`, 100% USDT cash (no entries).
  2. **Cross-Sectional Rank:** Selects top 3 alts ranked by 30-day return ($T-1$ lag).
  3. **Dual Filter:** A pair only qualifies if:
     $$\text{mom}_{30d} > 0 \quad \text{AND} \quad \text{mom}_{30d} > \text{btc\_mom}_{30d}$$
  4. **Benchmark Fallback:** Any unqualified slot automatically defaults to Bitcoin, eliminating cash drag while maintaining market beta.

### 3. `LeahAI.py` (FreqAI Volatility Expansion Classifier)
* **Timeframe:** `5m`
* **Concept:** Machine learning classifier predicting the probability of volatility expansion ($\text{ATR} > 5\%$) within the next 12 candles.
* **Feature Pipeline:** 15 stable features across momentum, return ratios, and candle morphology, validated across out-of-sample splits.

---

## Quickstart Guide

### 1. Installation

Clone this repository directly into your Freqtrade installation:

```bash
git clone https://github.com/Bederf/leah-freqtrade-strategies.git
cp -r leah-freqtrade-strategies/strategies/* user_data/strategies/
```

### 2. Backtesting

Run a historical backtest using Freqtrade's CLI:

```bash
# 1. Download daily historical data
freqtrade download-data --config user_data/config.json --timeframe 1d --days 1000

# 2. Backtest the Bitcoin Trend Control Strategy
freqtrade backtesting \
  --config config-templates/config_clock4_spot.example.json \
  --strategy Leah_E015_Trend \
  --timeframe 1d \
  --timerange 20240101-20261024

# 3. Backtest Dual Momentum with BTC Fallback
freqtrade backtesting \
  --config config-templates/config_clock4_spot.example.json \
  --strategy LeahClock4_A2 \
  --timeframe 1d \
  --timerange 20240101-20261024
```

### 3. Dry-Run Execution

To run in simulated live trading (`dry_run: true`):

```bash
freqtrade trade \
  --config config-templates/config_clock4_spot.example.json \
  --strategy LeahClock4_A2
```

---

## License

This project is licensed under the [MIT License](LICENSE).
