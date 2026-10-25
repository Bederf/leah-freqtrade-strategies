"""
LeahClock4_A2 — Dual Momentum Strategy with Benchmark Fallback
=============================================================
Architecture:
  - Macro Gate: BTC daily EMA30 > EMA90. If False -> 100% USDT Cash.
  - Cross-Sectional Ranking: Top-3 altcoins by 30-day momentum (T-1 lag).
  - Dual Filter (Absolute + Benchmark):
      Pass condition: (mom_30d > 0) AND (mom_30d > btc_mom_30d)
  - Benchmark Fallback (Option B):
      If an altcoin slot fails the dual filter, capital falls back to BTC
      rather than sitting in idle cash during a bull regime.
"""

import logging
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

MOM_DAYS = 30
TOP_N = 3
REBALANCE_DOW = 4  # Friday

class LeahClock4_A2(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    process_only_new_candles = True

    stoploss = -0.20
    use_custom_stoploss = False
    max_open_trades = TOP_N
    position_adjustment_enable = False

    minimal_roi = {"0": 100}

    # Warmup candle count
    startup_candle_count = 120

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Own-pair 30d momentum (T-1 closed candle)
        dataframe["mom_30d"] = (
            dataframe["close"].shift(1) / dataframe["close"].shift(MOM_DAYS + 1) - 1
        )

        # BTC informative data
        btc = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        if btc.empty:
            logger.error("BTC/USDT informative empty — fail-closed FLAT")
            dataframe["gate_bull"] = False
            dataframe["btc_mom_30d"] = 0.0
        else:
            b = btc.copy()
            b["ema30"] = b["close"].ewm(span=30, adjust=False).mean()
            b["ema90"] = b["close"].ewm(span=90, adjust=False).mean()
            b["bull"] = b["ema30"] > b["ema90"]
            b["btc_mom"] = b["close"].shift(1) / b["close"].shift(MOM_DAYS + 1) - 1

            dataframe = dataframe.merge(
                b[["date", "bull", "btc_mom"]].rename(
                    columns={"bull": "gate_bull", "btc_mom": "btc_mom_30d"}
                ),
                on="date",
                how="left",
            )
            dataframe["gate_bull"] = dataframe["gate_bull"].fillna(False)
            dataframe["btc_mom_30d"] = dataframe["btc_mom_30d"].fillna(0.0)

        # Cross-sectional momentum rank across whitelist pairs (excluding BTC)
        whitelist = [
            p for p in self.config.get("exchange", {}).get("pair_whitelist", [])
            if p not in ["BTC/USDT", "BTC/USDC"]
        ]

        mom_series = {}
        for pair in whitelist:
            other = self.dp.get_pair_dataframe(pair, self.timeframe)
            if other.empty:
                continue
            o = other[["date", "close"]].copy()
            o["mom_p"] = o["close"].shift(1) / o["close"].shift(MOM_DAYS + 1) - 1
            o = o.dropna(subset=["mom_p"])
            if o.empty:
                continue
            mom_series[pair] = o.set_index("date")["mom_p"]

        if mom_series:
            shared = pd.DataFrame(mom_series)
            ranks = shared.rank(axis=1, ascending=False, method="first")
            n_valid = shared.notna().sum(axis=1)
            complete = (len(mom_series) == len(whitelist)) & (n_valid == len(whitelist))
            usable = ranks.where(complete, other=float("nan"))

            own_col = metadata["pair"]
            if own_col in usable.columns:
                dataframe["rank"] = dataframe["date"].map(usable[own_col].to_dict())
                dataframe["in_top3"] = dataframe["rank"] <= TOP_N
            else:
                dataframe["rank"] = float("nan")
                dataframe["in_top3"] = False
        else:
            dataframe["rank"] = float("nan")
            dataframe["in_top3"] = False

        dataframe["is_friday"] = dataframe["date"].dt.dayofweek == REBALANCE_DOW

        # Dual Momentum filter: positive absolute momentum AND beating BTC
        dataframe["dual_pass"] = (
            (dataframe["mom_30d"] > 0) &
            (dataframe["mom_30d"] > dataframe["btc_mom_30d"])
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_tag"] = ""

        # Case 1: Altcoin entry (Must be Top-3 AND pass Dual Momentum filter)
        if metadata["pair"] not in ["BTC/USDT", "BTC/USDC"]:
            condition = (
                (dataframe["gate_bull"] == True) &
                (dataframe["is_friday"] == True) &
                (dataframe["in_top3"] == True) &
                (dataframe["dual_pass"] == True)
            )
            dataframe.loc[condition, "enter_long"] = 1
            dataframe.loc[condition, "enter_tag"] = "alt_dual_momentum"

        # Case 2: BTC Fallback entry (When bull gate is open, BTC acts as baseline)
        else:
            condition = (
                (dataframe["gate_bull"] == True) &
                (dataframe["is_friday"] == True)
            )
            dataframe.loc[condition, "enter_long"] = 1
            dataframe.loc[condition, "enter_tag"] = "btc_benchmark_fallback"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_tag"] = ""

        # If macro gate turns Bearish -> exit EVERYTHING to USDT Cash immediately
        gate_exit = (dataframe["gate_bull"] == False)

        if metadata["pair"] not in ["BTC/USDT", "BTC/USDC"]:
            # Altcoin exit: Gate closes OR drops out of top-3 OR fails dual momentum
            alt_rotation_exit = (
                (dataframe["is_friday"] == True) &
                ((dataframe["in_top3"] == False) | (dataframe["dual_pass"] == False))
            )
            condition = gate_exit | alt_rotation_exit
            dataframe.loc[condition, "exit_long"] = 1
            dataframe.loc[condition, "exit_tag"] = "alt_exit_or_rotate"
        else:
            # BTC exit: Gate closes
            dataframe.loc[gate_exit, "exit_long"] = 1
            dataframe.loc[gate_exit, "exit_tag"] = "btc_gate_exit"

        return dataframe
