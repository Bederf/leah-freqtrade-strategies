#!/usr/bin/env python3
"""Rank Snapshot Sidecar — writer (Option A).

Contract: docs/rank-snapshot-sidecar.md §2.

Produces user_data/data/rank_snapshot.json atomically. Consumers read it in place of
the in-loop cross-sectional rank, which is corrupted by per-pair evaluation order
combined with `process_only_new_candles: true` (a pair evaluated while a competitor's
candle is still in flight writes rank=NaN and is never re-evaluated for that date).

Invariants enforced here (spec §2):
  1. complete=true ONLY when every universe pair returned a 1d candle with the identical
     candle_date.
  2. mom_p == close.shift(1) / close.shift(MOM_DAYS + 1) - 1, on closed candles only.
     No lookahead: a candle counts only once its close time has passed.
  3. rank == DataFrame.rank(axis=1, ascending=False, method="first") over the universe,
     with columns in the order the strategy builds them (config whitelist order), so
     ties break identically. NOT alphabetical tie-breaking.
  4. No partial file ever exists at the final path (write .tmp, fsync, os.replace).
     On failure after retries: write complete=false with the pairs that did resolve and
     a `reason`.
  5. Universe is read from config_lea.json at run time, never hardcoded.

Usage:
    python rank_snapshot.py                  # live: write the snapshot
    python rank_snapshot.py --dry-run        # compute, print, do not write
    python rank_snapshot.py --candle-date 2026-10-30 --dry-run --limit 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "user_data/configs/config_lea.json"
SNAPSHOT = REPO / "user_data/data/rank_snapshot.json"
LOGFILE = REPO / "user_data/logs/rank_snapshot.log"

SCHEMA = 1
TIMEFRAME = "1d"
MOM_DAYS = 30
BACKOFF = [1, 2, 4, 8, 16]          # spec §2 retry policy: max 5 attempts
BINANCE = "https://api.binance.com/api/v3/klines"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    print(line, file=sys.stderr)
    try:
        LOGFILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def universe_from_config(path: Path = CONFIG) -> list[str]:
    """Spec invariant 5: read the universe, never hardcode it.

    Mirrors the strategy: the competitor set is the whitelist minus BTC/USDT, which
    LeahClock4 uses as the regime gate rather than a competitor.
    """
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    whitelist = cfg.get("exchange", {}).get("pair_whitelist", [])
    if not whitelist:
        raise SystemExit("FATAL: config_lea.json has no exchange.pair_whitelist")
    pairs = [p for p in whitelist if p != "BTC/USDT"]
    if not pairs:
        raise SystemExit("FATAL: whitelist contains no competitor pairs")
    return pairs


def fetch_klines(pair: str, limit: int) -> tuple[list, int]:
    """Fetch klines with per-pair exponential backoff. Returns (klines, attempts)."""
    symbol = pair.replace("/", "")
    url = f"{BINANCE}?symbol={symbol}&interval={TIMEFRAME}&limit={limit}"
    last_err = None
    attempts = 0
    for attempt, delay in enumerate(BACKOFF, start=1):     # 5 attempts max
        attempts = attempt
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LeahSidecar/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode())
            if not isinstance(payload, list) or not payload:
                raise ValueError(f"unexpected payload: {str(payload)[:120]}")
            if attempt > 1:
                log(f"pair={pair} attempt={attempt} OK after retries")
            return payload, attempts
        except urllib.error.HTTPError as exc:
            body = exc.read()[:160].decode(errors="replace")
            last_err = f"HTTP {exc.code} {body}"
            log(f"pair={pair} attempt={attempt} FAIL {last_err}")
        except Exception as exc:                            # noqa: BLE001 - log and retry
            last_err = f"{type(exc).__name__}: {exc}"
            log(f"pair={pair} attempt={attempt} FAIL {last_err}")
        if attempt < len(BACKOFF):
            time.sleep(delay)
    raise RuntimeError(f"{pair}: exhausted {attempts} attempts — last error {last_err}")


def closed_series(klines: list, now_ms: int) -> dict[str, float]:
    """Map UTC date -> close for CLOSED candles only (invariant 2, no lookahead)."""
    out: dict[str, float] = {}
    for k in klines:
        open_ms, close_px, close_ms = int(k[0]), float(k[4]), int(k[6])
        if close_ms >= now_ms:          # still forming — must not be used
            continue
        out[datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")] = close_px
    return out


def momentum_at(dates: list[str], closes: dict[str, float], candle_date: str) -> float | None:
    """close.shift(1) / close.shift(MOM_DAYS + 1) - 1 evaluated at candle_date."""
    try:
        idx = dates.index(candle_date)
    except ValueError:
        return None
    if idx - (MOM_DAYS + 1) < 0:
        return None
    return closes[dates[idx - 1]] / closes[dates[idx - (MOM_DAYS + 1)]] - 1.0


def compute(pair_closes: dict[str, dict[str, float]], universe: list[str],
            candle_date: str) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Cross-sectional momentum and ranks at candle_date.

    Ranks are produced by pandas over a one-row frame whose columns are in `universe`
    order — identical call to the strategy, so ties break by column order (invariant 3).
    """
    moms: dict[str, float] = {}
    missing: list[str] = []
    for pair in universe:
        closes = pair_closes.get(pair) or {}
        dates = sorted(closes)
        mom = momentum_at(dates, closes, candle_date)
        if mom is None:
            missing.append(pair)
        else:
            moms[pair] = mom
    if not moms:
        return {}, {}, missing
    frame = pd.DataFrame([{p: moms.get(p, float("nan")) for p in universe}], index=[candle_date])
    ranked = frame.rank(axis=1, ascending=False, method="first").iloc[0]
    ranks = {p: float(ranked[p]) for p in moms}
    return moms, ranks, missing


def atomic_write(path: Path, payload: dict) -> None:
    """Spec invariant 4: tmp + fsync + replace. Never a partial file at the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute and print, do not write")
    ap.add_argument("--candle-date", help="evaluate this YYYY-MM-DD instead of the latest closed")
    ap.add_argument("--limit", type=int, default=MOM_DAYS + 5, help="klines to request per pair")
    ap.add_argument("--json-out", help="write the computed payload here (harness use)")
    args = ap.parse_args()

    universe = universe_from_config()
    now_ms = int(time.time() * 1000)
    log(f"START pairs={len(universe)} limit={args.limit} universe={','.join(universe)}")

    pair_closes: dict[str, dict[str, float]] = {}
    attempts: dict[str, int] = {}
    failed: dict[str, str] = {}
    for pair in universe:
        try:
            klines, n = fetch_klines(pair, args.limit)
            pair_closes[pair] = closed_series(klines, now_ms)
            attempts[pair] = n
        except Exception as exc:                            # noqa: BLE001
            failed[pair] = str(exc)
            log(f"pair={pair} EXHAUSTED {exc}")

    latest = {p: max(c) for p, c in pair_closes.items() if c}
    if args.candle_date:
        candle_date = args.candle_date
    else:
        candle_date = max(latest.values()) if latest else None
    if candle_date is None:
        payload = {"schema": SCHEMA, "candle_date": None,
                   "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "timeframe": TIMEFRAME, "mom_days": MOM_DAYS, "universe": universe,
                   "complete": False, "attempts": attempts, "pairs": {},
                   "reason": "no closed candles retrieved for any pair"}
        log("RESULT complete=false reason=no-candles")
    else:
        mismatched = sorted(p for p, d in latest.items() if d != candle_date)
        moms, ranks, missing = compute(pair_closes, universe, candle_date)
        complete = (not failed and not mismatched and not missing
                    and len(moms) == len(universe))
        reasons = []
        if failed:
            reasons.append("fetch failed: " + ", ".join(sorted(failed)))
        if mismatched:
            reasons.append("candle_date differs: " + ", ".join(
                f"{p}@{latest[p]}" for p in mismatched))
        if missing:
            reasons.append("insufficient history at " + str(candle_date) + ": " + ", ".join(sorted(missing)))
        payload = {
            "schema": SCHEMA,
            "candle_date": candle_date,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeframe": TIMEFRAME,
            "mom_days": MOM_DAYS,
            "universe": universe,
            "complete": bool(complete),
            "attempts": max(attempts.values()) if attempts else 0,
            "pairs": {p: {"mom_p": (moms.get(p) if moms else None),
                          "rank": (ranks.get(p) if complete else None)}
                      for p in universe},
        }
        if not complete:
            payload["reason"] = "; ".join(reasons) if reasons else "incomplete"
            # A rank is only meaningful against the full competitor set. Null them all
            # rather than ship a partial cross-section a consumer might trust.
            for p in payload["pairs"]:
                payload["pairs"][p]["rank"] = None
        log(f"RESULT complete={complete} candle_date={candle_date} "
            f"pairs_ok={len(moms)}/{len(universe)}"
            + ("" if complete else f" reason={' '.join(reasons)}"))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["complete"] else 2

    atomic_write(SNAPSHOT, payload)
    print(f"wrote {SNAPSHOT} complete={payload['complete']} candle_date={payload['candle_date']}")
    return 0 if payload["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
