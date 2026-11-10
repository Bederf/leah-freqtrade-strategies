"""
LeahClock4 — Regime-gated cross-sectional momentum (NO ML)
===========================================================
E013 deployment of the E012-validated structure (2021-2026 full cycle:
+3297%, CAGR +89.8%, Sharpe 1.14, maxDD -49.3%; 2022: -6.5% vs -69.2% BH).

ARCHITECTURE (deliberate — the audit discipline from E010/E011/E012):
  - NO ML, NO feature pipeline, NO model files, NO funding CSV in v1.
    The three bug classes that killed the ML program (tf1h lookahead leak,
    trailing-week return lookahead, apples-oranges compounding) are
    structurally impossible: every input is a published-reference rule on
    raw daily OHLCV.
  - GATE:  BTC daily EMA30 > EMA90 (PyQuantLab reference). BEAR -> flat.
  - SELECT: cross-sectional 30d momentum (T-1 lag), top-3 of whitelist.
    Ranking computed via DataProvider across ALL whitelist pairs — freqtrade
    slices get_pair_dataframe to the current candle in backtest, so no
    cross-pair lookahead (verified against freqtrade source 2026-08-20).
  - CADENCE: weekly rebalance, Friday candle. Signal uses shift(1) (closed
    Thursday candle, T-1); entry executes at Friday close (backtest default
    fills at candle close). Matches E012 forward-week semantics: a position
    entered at Friday close earns the NEXT week's return, never trailing.
  - EXIT: gate closes (regime exit to cash) OR pair drops out of top-3 on a
    Friday (rotation).

V1 scope decisions (E013 spec, approved by Pieter 2026-08-20):
  - funding blend OFF (momentum-only; parity gate vs E012 momentum-only)
  - stoploss -0.20 (real emergency brake, NOT Clock3's -0.99 artifact)
  - spot, long-only, stake 30 USDT, max_open_trades 3

PARITY GATE (mandatory before Phase A): freqtrade backtest 2021-01-01 ->
2026-08-20 must land within ±10pp of E012's +3297% total on the same 8
pairs. If it diverges, STOP — find the discrepancy before any paper trade.
"""
import logging

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# ─── v4.4 SHADOW (additive; observation-only) ───────────────────────────────
# LeahAI.py owns the v4.4 override, but LeahAI is a FreqAI strategy and is not
# deployed — the bot runs this file. This port adds ONLY a probability column and one
# log line so the v4.4 comparison can accumulate during Phase A. It writes to no
# existing column and emits no line matching CLOCK4 / CLOCK4-AB.
import json as _json

_V44_FEATURES_FALLBACK = [
    "vol_ratio_20", "vol_ratio_10", "vol_ratio_5", "vol_ratio_3",
    "vol_ma20", "vol_ma10",
    "%ret_1", "%ret_3", "candle_body", "hl_range",
    "lower_shadow", "upper_shadow", "mom_6", "atr14", "%atr14_rel",
]

logger = logging.getLogger(__name__)

MOM_DAYS = 30
TOP_N = 3
REBALANCE_DOW = 4  # Friday

# ─── RANK SNAPSHOT SIDECAR (Option A) ───────────────────────────────────────
# docs/rank-snapshot-sidecar.md. scripts/rank_snapshot.py builds ONE cross-section
# for the whole universe shortly after the daily boundary, off the per-pair
# processing order. This strategy consumes that snapshot when it is complete and
# current, and otherwise falls back to the in-loop computation unchanged — it
# degrades to today's behaviour, never to a wrong rank. Reading is mtime-cached,
# so the file is parsed once per boundary rather than once per pair.
import os as _os
import time as _time
from pathlib import Path as _Path

_RANK_SNAPSHOT_PATH = _Path(__file__).resolve().parent.parent / "data/rank_snapshot.json"
_RANK_SNAPSHOT_STATE = {"mtime": None, "payload": None}


def _load_rank_snapshot() -> dict | None:
    """Return the snapshot payload, re-reading only when the file changes."""
    try:
        st = _os.stat(_RANK_SNAPSHOT_PATH)
    except OSError:
        return None
    if _RANK_SNAPSHOT_STATE["mtime"] == st.st_mtime:
        return _RANK_SNAPSHOT_STATE["payload"]
    try:
        with open(_RANK_SNAPSHOT_PATH, encoding="utf-8") as fh:
            payload = _json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("[RANKSNAP] unreadable snapshot %s: %s — in-loop fallback",
                       _RANK_SNAPSHOT_PATH, exc)
        payload = None
    _RANK_SNAPSHOT_STATE["mtime"] = st.st_mtime
    _RANK_SNAPSHOT_STATE["payload"] = payload
    return payload

# Macro event risk table (H2 2026) — VERIFIED 2026-08-29 against official
# schedules: BLS CPI release calendar + Federal Reserve FOMC calendar.
# FOMC dates = statement/decision day (2nd day of meeting, 14:00 ET).
# CPI dates = BLS release day (08:30 ET).
HIGH_IMPACT_DATES = {
    "2026-09-11": "red",  # CPI (BLS official: Aug data, Sep 11)
    "2026-09-16": "red",  # FOMC decision (meeting Sep 15-16)
    "2026-10-14": "red",  # CPI (BLS official: Sep data, Oct 14)
    "2026-10-28": "red",  # FOMC decision (meeting Oct 27-28)
    "2026-11-10": "red",  # CPI (BLS official: Oct data, Nov 10)
    "2026-12-09": "red",  # FOMC decision (meeting Dec 8-9)
    "2026-12-10": "red",  # CPI (BLS official: Nov data, Dec 10)
}


# Ex-post shock threshold: BTC weekly |return| p75 over 10.4y (543 weeks):
# median 3.23%, p75 5.86%, sigma 5.79%. 6% ~ p75 = top-quartile high-vol week.
# (4% would tag 44% of weeks — sub-sigma, rejected. Measured 2026-08-29.)
SHOCK_THRESHOLD = 0.06

# ── Rank fetch retry (added 2026-11-10) ─────────────────────────────────────────
# Binance does not publish a final 1d candle at the 00:00:00Z close. The in-loop fallback used
# to fetch ONCE and surrender, so the completeness guard withheld the rank on 8 of 9 nights
# (Nov 2-10) — the strategy looked healthy while never qualifying an entry. The sidecar, which
# fetches five minutes later, got 8/8 complete every one of those nights.
#
# So: poll until the universe is complete, instead of treating the first partial read as final.
# The ranks themselves are UNCHANGED once the data arrives; only the moment of the fetch moves.
# The window below is sized to span the sidecar's proven 5-minute lag. Applied once per candle
# close, not once per pair (see _rank_cache), and never during backtesting.
RANK_FETCH_ATTEMPTS = 7
RANK_FETCH_DELAY_S = 45.0


class LeahClock4(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1d"
    process_only_new_candles = True

    # Risk — E013 SPEC: -0.20 emergency stop (deliberate modification vs E012's
    # no-stop; measured separately against parity in the E013 manifest).
    stoploss = -0.20
    use_custom_stoploss = False
    max_open_trades = TOP_N
    position_adjustment_enable = False

    # ROI: none — exits are regime/rotation based
    minimal_roi = {"0": 100}

    # Warmup: momentum needs shift(31); EMA30/90 uses adjust=False EWM from
    # bar 0 exactly as E012 computed it (no convergence requirement — E012's
    # gate is identical from day one). 35 = 31 (mom) + 4 buffer.
    startup_candle_count = MOM_DAYS + 5

    def informative_pairs(self):
        # BTC only for the regime gate — other whitelist pairs are fetched via get_pair_dataframe.
        # The v4.4 models, however, were TRAINED ON 5-MINUTE BARS (46,034 samples over 159.8
        # days = exactly 288 bars/day), so their features must be computed on a 5m series.
        # Declaring them here makes freqtrade download and refresh that 5m data.
        pairs = [("BTC/USDT", self.timeframe)]
        if getattr(self, "_v44_enabled", False):
            for _p in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"):
                pairs.append((_p, "5m"))
        return pairs

    # ── v4.4 shadow helpers (additive; never touches gate/rank/entry/exit) ──
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._v44_enabled = bool(config.get("force_v44_model", False))
        self._v44_cache: dict = {}
        self._v44_warned: set = set()
        self._v44_last_log: dict = {}
        self._v44_5m_cache: dict = {}
        self._v44_5m_warned: set = set()
        if self._v44_enabled:
            logger.warning("force_v44_model=true — v4.4 shadow observation ENABLED (LeahClock4 v4.4 port)")

    def _v44_load(self, pair: str):
        """Load the v4.4 classifier + scaler + the model's declared feature order."""
        try:
            import joblib
            from pathlib import Path
            pf = pair.replace("/", "_")
            base = pf[:-5] if pf.endswith("_USDT") else pf
            for stem in (base, pf):
                mp = Path(f"user_data/models/leah_v4_4_{stem}_xgb_clf.pkl")
                if not mp.exists():
                    continue
                sp = mp.parent / f"{mp.stem}_scaler.pkl"
                if not sp.exists():
                    continue
                feats = list(_V44_FEATURES_FALLBACK)
                try:
                    with open(mp.parent / f"{mp.stem}_meta.json") as fh:
                        declared = _json.load(fh).get("features") or []
                    if len(declared) == 15:
                        feats = list(declared)
                except Exception:
                    pass
                return joblib.load(mp), joblib.load(sp), feats
        except Exception as exc:
            logger.warning(f"[{pair}] v4.4 load failed: {exc}")
        return None, None, None

    @staticmethod
    def _v44_feature_matrix(df: DataFrame, feats):
        """The 15 v4.4 features from raw OHLCV — same formulas as LeahAI._apply_v44_prediction."""
        import numpy as np
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        op = df["open"].values
        n = len(df)
        out = {}
        pct1 = np.zeros(n)
        pct1[1:] = (close[1:] - close[:-1]) / close[:-1]
        out["%ret_1"] = pct1
        pct3 = np.zeros(n)
        pct3[3:] = (close[3:] - close[:-3]) / close[:-3]
        out["%ret_3"] = pct3
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr = np.zeros(n)
        atr[:14] = np.nan
        atr[14] = tr[:14].mean()
        for i in range(15, n):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
        out["atr14"] = atr
        out["%atr14_rel"] = atr / close
        for win in (3, 5, 10, 20):
            ma = np.zeros(n)
            for i in range(n):
                ma[i] = np.mean(volume[max(0, i - win + 1):i + 1])
            out[f"vol_ratio_{win}"] = volume / np.where(ma == 0, 1, ma)
        for win in (10, 20):
            ma = np.zeros(n)
            for i in range(n):
                ma[i] = np.mean(volume[max(0, i - win + 1):i + 1])
            out[f"vol_ma{win}"] = ma
        out["hl_range"] = (high - low) / close
        out["candle_body"] = (close - op) / close
        max_co = np.maximum(close, op)
        min_co = np.minimum(close, op)
        out["upper_shadow"] = (high - max_co) / close
        out["lower_shadow"] = (min_co - low) / close
        pct6 = np.zeros(n)
        pct6[6:] = (close[6:] - close[:-6]) / close[:-6]
        out["mom_6"] = pct6
        return pd.DataFrame({f: out[f] for f in feats}).fillna(0).values

    def _v44_probabilities(self, dataframe: DataFrame, metadata: dict):
        """v4.4 probability per row, or None when this pair has no model."""
        pair = metadata["pair"]
        if pair not in self._v44_cache:
            self._v44_cache[pair] = self._v44_load(pair)
        model, scaler, feats = self._v44_cache[pair]
        if model is None:
            if pair not in self._v44_warned:
                self._v44_warned.add(pair)
                logger.info(
                    f"[CLOCK4-V44][{pair}] no v4.4 model — shadow skipped "
                    f"(expected: only BTC/ETH/SOL/LINK have models)"
                )
            return None
        # 5m source of truth — see _v44_probs_from_5m for why 1d is wrong for these models.
        return self._v44_probs_from_5m(pair, dataframe, model, scaler, feats)

    def _v44_probs_from_5m(self, pair: str, dataframe: DataFrame, model, scaler, feats):
        """v4.4 probabilities for each DAILY candle, computed from the 5m series.

        The models were trained on 5-minute bars (46,034 samples / 159.8 days = 288 bars/day)
        and carry a MinMaxScaler fitted to 5m feature ranges. Feeding them a 1d dataframe puts
        roughly 38% of feature rows outside that fitted range — vol_ma20 and %atr14_rel sit
        entirely outside it — and the scaler clips, so every probability compresses to about
        half its trained level. Measured before this fix: ETH mean 0.1546 against a trained
        0.3235. Computing on 5m and carrying the values onto the daily candles reproduces the
        trained distribution exactly (delta 0.0000 on all three models).

        Returns None when the 5m series is missing or too short — the caller then withholds
        the prediction rather than silently falling back to the wrong resolution.
        """
        import numpy as np

        try:
            d5 = self.dp.get_pair_dataframe(pair=pair, timeframe="5m")
        except Exception as exc:
            d5 = None
            logger.warning(f"[CLOCK4-V44][{pair}] 5m informative fetch failed: "
                           f"{exc.__class__.__name__}: {exc}")

        # The DataProvider only carries a short recent window of informative candles once a
        # strategy restarts (observed live: 4 daily-equivalent rows), which is far too thin for
        # the distribution statistic the shadow run exists to produce. Layer the on-disk 5m
        # history underneath the live rows: history from the cache, freshest bars from the API.
        #
        # feather returns tz-NAIVE timestamps while the DataProvider returns tz-aware UTC, and
        # concatenating the two yields a mixed-dtype column that merge_asof rejects outright
        # ("incompatible merge keys"). Normalise every series to UTC-aware before they meet.
        def _utc(s):
            s = pd.to_datetime(s)
            return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")

        if d5 is not None and len(d5):
            d5 = d5.copy()
            d5["date"] = _utc(d5["date"])
        try:
            from pathlib import Path as _P
            _dd = _P((getattr(self, "config", {}) or {}).get("datadir")
                     or "user_data/data/binance")
            _f = _dd / f"{pair.replace('/', '_')}-5m.feather"
            if _f.exists():
                _hist = pd.read_feather(_f)
                _hist["date"] = _utc(_hist["date"])
                if d5 is not None and len(d5):
                    d5 = (pd.concat([_hist, d5], ignore_index=True)
                          .drop_duplicates(subset=["date"], keep="last")
                          .sort_values("date").reset_index(drop=True))
                else:
                    d5 = _hist
        except Exception as exc:
            logger.warning(f"[CLOCK4-V44][{pair}] 5m history load skipped: "
                           f"{exc.__class__.__name__}: {exc}")

        if d5 is None or len(d5) < 100:
            if pair not in self._v44_5m_warned:
                self._v44_5m_warned.add(pair)
                logger.warning(
                    f"[CLOCK4-V44][{pair}] 5m informative data unavailable "
                    f"({0 if d5 is None else len(d5)} bars) — v4.4 prediction WITHHELD "
                    f"(1d features are out of population for these models)"
                )
            return None

        # Cache on (pair, last 5m bar, daily length): the feature matrix costs a few seconds
        # over ~80k bars, and the 5m series only advances every 5 minutes.
        key = (pair, str(d5["date"].iloc[-1]), len(dataframe))
        if key in self._v44_5m_cache:
            return self._v44_5m_cache[key]

        p5 = model.predict_proba(scaler.transform(self._v44_feature_matrix(d5, feats)))[:, 1]

        bars = pd.DataFrame({
            "bar_time": d5["date"].values,
            "close_time": d5["date"].add(pd.Timedelta(minutes=5)).values,
            "prob": p5,
        }).sort_values("close_time")
        _ddate = _utc(dataframe["date"])
        daily = pd.DataFrame({
            "date": _ddate.values,
            "window_end": _ddate.add(pd.Timedelta(days=1)).values,
        }).sort_values("window_end")
        merged = pd.merge_asof(daily, bars, left_on="window_end", right_on="close_time",
                               direction="backward", tolerance=pd.Timedelta(days=1))
        merged = merged.sort_values("date")
        probs = merged["prob"].to_numpy(dtype=float)
        if len(probs) != len(dataframe):
            return None
        self._v44_5m_cache[key] = probs
        for stale in [k for k in self._v44_5m_cache if k[0] == pair and k != key]:
            self._v44_5m_cache.pop(stale, None)
        return probs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Own-pair 30d momentum, T-1 lag: close[thu]/close[thu-30] computed on
        # the CLOSED previous candle only (never the forming candle).
        dataframe["mom_30d"] = (
            dataframe["close"].shift(1) / dataframe["close"].shift(MOM_DAYS + 1) - 1
        )

        # BTC regime gate (fail-closed: missing BTC -> not bull)
        btc = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        if btc.empty:
            logger.error("BTC/USDT informative empty — fail-closed FLAT")
            dataframe["gate_bull"] = False
        else:
            b = btc.copy()
            b["ema30"] = b["close"].ewm(span=30, adjust=False).mean()
            b["ema90"] = b["close"].ewm(span=90, adjust=False).mean()
            b["bull"] = b["ema30"] > b["ema90"]
            dataframe = dataframe.merge(
                b[["date", "bull", "close"]].rename(
                    columns={"bull": "gate_bull", "close": "btc_close"}),
                on="date", how="left",
            )
            dataframe["gate_bull"] = dataframe["gate_bull"].fillna(False)

        # Cross-sectional momentum rank across ALL whitelist pairs.
        #
        # FIXED 2026-10-11 (E024 rank fix). Previously each pair ranked inside its OWN
        # dataframe: the competitor columns arrived via a per-pair left-merge at this
        # pair's processing moment, so a pair processed early was ranked against a
        # competitor set whose candles had not been fetched yet. Result: duplicate
        # ranks, N differing across pairs on the SAME candle, and `rank <= TOP_N`
        # over-admitting. Evidence (E024 rank audit, 8 logged cycles): 16 mismatches,
        # every one improving the pair, and the drifted pairs are always a PREFIX of
        # the per-cycle processing order (e.g. 10-08: BNB 1 vs true 5 and DOGE 2 vs
        # true 8 — the first two pairs processed — while the last three were exact).
        #
        # The rank is now computed ONCE over one aligned frame covering every
        # whitelist pair on a common date index, and each pair reads its own rank out
        # of that shared frame by date. Both parties to a comparison therefore always
        # use the same row and the same N, by construction.
        #
        # Completeness guard (fail closed): a rank is only usable when the row carries
        # every whitelist pair. A partial row would rank this pair against a reduced
        # competitor set — the defect being fixed — so it yields NO rank, logs loudly,
        # and cannot enter. (In the 969-date historical window the guard withholds 0
        # rows: it exists for the live staleness case, not for normal history.)
        # Rank source: sidecar snapshot first, in-loop computation as fallback.
        # A rank is only ever taken from a COMPLETE cross-section of the whole
        # universe; anything less is withheld, never faked.
        dataframe["rank_valid"] = False
        if not self._apply_rank_snapshot(dataframe, metadata):
            self._rank_inloop(dataframe, metadata)

        dataframe["is_friday"] = dataframe["date"].dt.dayofweek == REBALANCE_DOW

        # Alpha-vs-beta shadow measurement (Clock 3 lesson): for every would-be
        # entry (gate_bull & Friday & top-3) at T, log the realized 7-day return
        # of the pair vs BTC at T+7. ALL inputs are closed-candle values known
        # at log time — zero lookahead. Fires once per signal, on the following
        # Friday (trailing week = E012 forward-week semantics).
        dataframe["pair_7d"] = dataframe["close"] / dataframe["close"].shift(7) - 1
        if "btc_close" in dataframe.columns:
            dataframe["btc_7d"] = (
                dataframe["btc_close"] / dataframe["btc_close"].shift(7) - 1
            )
            dataframe["sig_7d_ago"] = (
                dataframe["gate_bull"].shift(7)
                & dataframe["is_friday"].shift(7)
                & dataframe["in_top3"].shift(7)
            )

        # Phase A observability: log gate + rank state once per new daily candle
        if len(dataframe) > 0 and not dataframe.empty:
            last = dataframe.iloc[-1]
            bull = bool(last["gate_bull"])
            d = last["date"]
            if hasattr(d, "to_pydatetime"):
                d = d.to_pydatetime()
            rank = float(last["rank"]) if pd.notna(last.get("rank", pd.NA)) else None
            mom = float(last["mom_30d"]) if pd.notna(last["mom_30d"]) else None
            logger.info(
                f"[CLOCK4][{metadata['pair']}] date={d} gate_bull={bull} "
                f"is_friday={bool(last['is_friday'])} rank={rank} mom_30d={mom:.4f}" if mom is not None
                else f"[CLOCK4][{metadata['pair']}] date={d} gate_bull={bull} "
                     f"is_friday={bool(last['is_friday'])} rank={rank} mom_30d=None"
            )
            # outcome log: would-be entry 7d ago -> realized pair vs BTC week
            if "sig_7d_ago" in dataframe.columns:
                sig_prev = bool(last["sig_7d_ago"]) if pd.notna(last.get("sig_7d_ago", pd.NA)) else False
                if sig_prev:
                    p7 = float(last["pair_7d"]) if pd.notna(last.get("pair_7d", pd.NA)) else None
                    b7 = float(last["btc_7d"]) if pd.notna(last.get("btc_7d", pd.NA)) else None
                    if p7 is not None and b7 is not None:
                        sig_date = d - pd.Timedelta(days=7)
                        # macro tag: red if any high-impact event falls inside the
                        # held week (entry Friday close -> outcome Friday close).
                        # Events before the signal Friday's close are baked into
                        # the entry price; only the hold window matters.
                        ev = "green"
                        sd, dd = sig_date.date(), d.date()
                        for ds in HIGH_IMPACT_DATES:
                            ed = pd.Timestamp(ds).date()
                            if sd < ed <= dd:
                                ev = HIGH_IMPACT_DATES[ds]
                                break
                        logger.info(
                            f"[CLOCK4-AB][outcome] pair={metadata['pair']} "
                            f"sig_fri={sd} pair_7d={p7:+.4f} "
                            f"btc_7d={b7:+.4f} alpha={p7 - b7:+.4f} ev={ev} "
                            f"shock={1 if abs(b7) > SHOCK_THRESHOLD else 0}"
                        )
        # ── v4.4 SHADOW OBSERVATION (additive; observation-only) ───────────
        # Everything above is untouched. This block cannot alter gate_bull, rank,
        # in_top3, is_friday, pair_7d/btc_7d/sig_7d_ago, enter_long or exit_long, and it
        # emits no line matching CLOCK4 / CLOCK4-AB. One added column, one added log line.
        if self._v44_enabled:
            try:
                probs = self._v44_probabilities(dataframe, metadata)
            except Exception as exc:
                probs = None
                logger.warning(
                    f"[{metadata['pair']}] v4.4 shadow failed: {exc.__class__.__name__}: {exc}"
                )
            if probs is not None:
                dataframe["v44_prob"] = probs
                # Log once per CANDLE, not once per throttle iteration: populate_indicators runs
                # every ~5s, so an unguarded line would emit ~52k lines/day and bury the log.
                _candle = str(dataframe["date"].iloc[-1]) if "date" in dataframe.columns else "?"
                if self._v44_last_log.get(metadata["pair"]) != _candle:
                    self._v44_last_log[metadata["pair"]] = _candle
                    n55 = int((probs >= 0.55).sum())
                    # Two shares, deliberately. `%>55` is over the full dataframe and is what the
                    # documented 1-10% band describes — but it barely moves as history grows, so it
                    # cannot answer "is this pair in band NOW". `%>55_30` is the last 30 candles
                    # (one month) and is the statistic that tracks band status per pair over the
                    # shadow window (SOL landed at 0.3%, below the 1% floor — see E022).
                    _w = probs[-30:] if len(probs) >= 30 else probs
                    w55 = int((_w >= 0.55).sum())
                    # nan-aware: daily candles predating the 5m series carry no prediction.
                    _v = probs[~pd.isna(probs)]
                    logger.info(
                        f"[CLOCK4-V44][{metadata['pair']}] v4.4 override (5m features) — "
                        f"prob mean={_v.mean():.4f}, "
                        f"std={_v.std():.4f}, %>55={n55}/{len(_v)}, "
                        f"%>55_30={w55}/{len(_w)}, last={_v[-1]:.4f}"
                    )
        return dataframe

    def _apply_rank_snapshot(self, dataframe: DataFrame, metadata: dict) -> bool:
        """Use the sidecar snapshot when it is complete and current. True if applied.

        Spec §2 consumer rule: apply only when `complete` is true AND the snapshot's
        candle_date equals this dataframe's last date. Otherwise return False and let
        the caller fall back — never substitute a rank the snapshot does not assert.
        """
        snap = _load_rank_snapshot()
        if not snap or not snap.get("complete"):
            return False
        pair = metadata["pair"]
        entry = (snap.get("pairs") or {}).get(pair)
        if not entry or entry.get("rank") is None:
            return False
        candle_date = snap.get("candle_date")
        if len(dataframe) == 0 or candle_date is None:
            return False
        last_date = pd.Timestamp(dataframe["date"].iloc[-1]).strftime("%Y-%m-%d")
        if candle_date != last_date:
            # Stale snapshot (boundary not yet regenerated). Fall back rather than
            # applying yesterday's rank to today's candle.
            return False
        mask = pd.to_datetime(dataframe["date"]).dt.strftime("%Y-%m-%d") == candle_date
        dataframe.loc[mask, "rank"] = float(entry["rank"])
        dataframe["in_top3"] = dataframe["rank"] <= TOP_N
        dataframe["rank_valid"] = dataframe["rank"].notna()
        logger.info(
            "[RANKSNAP][%s] rank=%.0f from sidecar candle_date=%s "
            "(complete cross-section, %d pairs)",
            pair, float(entry["rank"]), candle_date, len(snap["pairs"]),
        )
        return True

    # One cross-section per candle close, shared by every pair in the cycle. Without this the
    # retry below would sleep once per pair — eight times over — instead of once per candle.
    _rank_cache: dict = {}

    @staticmethod
    def _is_live_run(dp) -> bool:
        """True only in live/dry-run. A backtest must never sleep on a retry."""
        try:
            return str(dp.runmode.value) in ("live", "dry_run")
        except Exception:
            return False

    def _collect_rank_inputs(self, whitelist: list, dataframe: DataFrame,
                             metadata: dict) -> dict:
        """Fetch every pair's 30d momentum series, polling until the universe is complete.

        Binance does not publish a final 1d candle at the 00:00:00Z close. Fetching once there saw
        4-6 of 8 pairs and withheld the rank on 8 of 9 nights (Nov 2-10), while the sidecar —
        fetching at 00:05 — saw 8/8 every night. This waits for the data instead of concluding it
        is absent.

        Returns the same {pair: mom_series} shape as before, so the ranking and the completeness
        guard downstream are byte-for-byte unchanged. A pair that genuinely has no data still ends
        as a withhold.
        """
        key = (self.timeframe, str(dataframe["date"].iloc[-1]) if len(dataframe) else "?")
        cached = self._rank_cache.get(key)
        if cached is not None:
            return cached

        attempts = RANK_FETCH_ATTEMPTS if self._is_live_run(self.dp) else 1
        mom_series: dict = {}
        for attempt in range(1, attempts + 1):
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

            missing = [p for p in whitelist if p not in mom_series]
            if not missing:
                if attempt > 1:
                    logger.warning(
                        "[CLOCK4][%s] rank universe completed on attempt %d/%d after %.0fs "
                        "— full cross-section of %d pairs",
                        metadata["pair"], attempt, attempts,
                        (attempt - 1) * RANK_FETCH_DELAY_S, len(whitelist))
                break
            if attempt < attempts:
                logger.warning(
                    "[CLOCK4][%s] rank fetch %d/%d incomplete — %d of %d pairs, missing %s; "
                    "waiting %.0fs for the candle to settle",
                    metadata["pair"], attempt, attempts, len(mom_series), len(whitelist),
                    ",".join(sorted(missing)), RANK_FETCH_DELAY_S)
                _time.sleep(RANK_FETCH_DELAY_S)

        self._rank_cache = {key: mom_series}
        return mom_series

    def _rank_inloop(self, dataframe: DataFrame, metadata: dict) -> None:
        """In-loop cross-sectional rank — unchanged fallback behaviour (E024 semantics)."""
        whitelist = [p for p in self.config.get("exchange", {}).get("pair_whitelist", [])
                     if p != "BTC/USDT"]
        mom_series = self._collect_rank_inputs(whitelist, dataframe, metadata)

        if mom_series:
            shared = pd.DataFrame(mom_series)          # one frame, one date index
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
            if not bool(complete.all()):
                missing = sorted(set(whitelist) - set(mom_series))
                warned = getattr(self, "_rank_warned", None)
                if warned is None:
                    warned = self._rank_warned = set()
                key = str(dataframe["date"].iloc[-1]) if len(dataframe) else "?"
                if key not in warned:
                    warned.add(key)
                    logger.warning(
                        "[CLOCK4][%s] rank WITHHELD (fail-closed) — incomplete competitor "
                        "set: %d of %d pairs present on the latest candle%s. No entry is "
                        "taken from a partial rank.",
                        metadata["pair"], int(n_valid.iloc[-1]), len(whitelist),
                        ("; never returned data: " + ",".join(missing)) if missing else "",
                    )
        else:
            dataframe["rank"] = float("nan")
            dataframe["in_top3"] = False
        dataframe["rank_valid"] = dataframe["rank"].notna()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # E012 semantics: top-3 by RELATIVE 30d momentum rank, sign irrelevant.
        # (Rank is cross-sectional — a top-3 pick can have negative raw mom.)
        dataframe.loc[
            (dataframe["gate_bull"])
            & (dataframe["is_friday"])
            & (dataframe["in_top3"]),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # E012 semantics: exit ONLY on regime flip (gate closes) or rotation
        # (pair drops out of top-3 on a Friday). Momentum sign never exits.
        #
        # GUARD 2026-11-07: the rotation clause requires `rank_valid`. A withheld rank
        # (NaN) reads as `in_top3 == False`, so without this guard a single failed
        # candle fetch on a Friday would force-liquidate an open position — a rank
        # failure silently masquerading as "dropped out of the top 3". Exit only on a
        # rank we actually have.
        dataframe.loc[
            (~dataframe["gate_bull"])
            | (dataframe["is_friday"] & dataframe["rank_valid"] & (~dataframe["in_top3"])),
            "exit_long",
        ] = 1
        return dataframe
