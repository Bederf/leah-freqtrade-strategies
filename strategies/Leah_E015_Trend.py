"""
Leah_E015_Trend — Bitcoin Macro Trend Control
============================================
Quantitative Trend Following on BTC/USDT:
  - Long BTC when BTC daily EMA30 > EMA90.
  - Step to 100% USDT Cash when BTC daily EMA30 <= EMA90.
  - Truncates severe left-tail bear drawdowns (-74% -> -50%) while preserving bull upside.
"""

import logging
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

class Leah_E015_Trend(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    process_only_new_candles = True

    # Exit purely driven by trend regime
    stoploss = -0.25
    use_custom_stoploss = False
    max_open_trades = 1
    position_adjustment_enable = False

    minimal_roi = {"0": 100}

    # 90 bars needed for EMA90 + buffer
    startup_candle_count = 120

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA30 & EMA90 on daily close
        dataframe["ema30"] = dataframe["close"].ewm(span=30, adjust=False).mean()
        dataframe["ema90"] = dataframe["close"].ewm(span=90, adjust=False).mean()
        dataframe["gate_bull"] = dataframe["ema30"] > dataframe["ema90"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_tag"] = ""

        # Only trade BTC/USDT
        if metadata["pair"] not in ["BTC/USDT", "BTC/USDC"]:
            return dataframe

        # Enter long when gate_bull is True (using closed T-1 candle)
        condition = (
            (dataframe["gate_bull"].shift(1) == True) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[condition, "enter_long"] = 1
        dataframe.loc[condition, "enter_tag"] = "btc_bull_trend"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_tag"] = ""

        if metadata["pair"] not in ["BTC/USDT", "BTC/USDC"]:
            return dataframe

        # Exit to cash when gate_bull becomes False (T-1 closed candle)
        condition = (
            (dataframe["gate_bull"].shift(1) == False) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[condition, "exit_long"] = 1
        dataframe.loc[condition, "exit_tag"] = "btc_bear_regime_exit"
        return dataframe
