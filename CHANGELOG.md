# Changelog

## Unreleased

- `LeahClock4`: compute v4.4 model features on the **5-minute** series they were
  trained on (46,034 samples / 159.8 days = 288 bars/day) instead of the daily
  frame. Feeding a daily frame to a scaler fitted on 5m ranges clipped roughly 38%
  of rows and compressed every probability to about half its trained level.
- `LeahClock4`: retry the in-loop rank fetch instead of surrendering on the first
  partial read. Binance does not publish the final 1d candle at the 00:00:00Z
  close, so the completeness guard had withheld the rank on 8 of 9 nights while a
  sidecar reading five minutes later succeeded every night. Rank values are
  unchanged once data arrives; only the moment of the fetch moves.
- Documentation and contributor surfaces are being refined for the public research repository.

## 0.1.0 — 2026-10-25

- Initial public portfolio release of the Leah strategy collection.
- Included trend control, dual-momentum and FreqAI strategy examples with
  example configuration templates.
