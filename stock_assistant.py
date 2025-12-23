import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Deque, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Quote:
    symbol: str
    name: str
    price: float
    prev_close: float
    open_price: float
    high: float
    low: float
    volume: float
    amount: float
    date: str
    time: str
    bid_prices: Tuple[float, float, float, float, float]
    bid_volumes: Tuple[float, float, float, float, float]
    ask_prices: Tuple[float, float, float, float, float]
    ask_volumes: Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class Candle:
    date: str
    open_price: float
    close_price: float
    high: float
    low: float
    volume: float


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def normalize_symbol(raw: str) -> str:
    s = raw.strip().lower()
    if not s:
        return ""

    s = s.replace(".", "").replace("-", "").replace("_", "")
    if s.startswith(("sh", "sz", "bj")) and len(s) >= 8:
        return s[:2] + s[2:8]

    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 6:
        return ""
    digits = digits[:6]

    if digits.startswith(("60", "68")):
        return "sh" + digits
    if digits.startswith(("00", "30")):
        return "sz" + digits
    if digits.startswith(("83", "87", "88", "43")):
        return "bj" + digits
    return "sz" + digits


def fetch_sina_quotes(symbols: Sequence[str], timeout_s: float = 5.0) -> Dict[str, Quote]:
    normalized = [normalize_symbol(s) for s in symbols]
    normalized = [s for s in normalized if s]
    if not normalized:
        return {}

    query = ",".join(normalized)
    url = "https://hq.sinajs.cn/list=" + urllib.parse.quote(query, safe=",")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("gbk", errors="ignore")

    quotes: Dict[str, Quote] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("var hq_str_"):
            continue
        try:
            left, right = line.split("=", 1)
            sym = left.replace("var hq_str_", "").strip()
            data = right.strip().strip(";").strip().strip('"')
            if not data:
                continue
            parts = data.split(",")
            if len(parts) < 10:
                continue
            name = parts[0]
            open_price = _to_float(parts[1])
            prev_close = _to_float(parts[2])
            price = _to_float(parts[3])
            high = _to_float(parts[4])
            low = _to_float(parts[5])
            volume = _to_float(parts[8])
            amount = _to_float(parts[9])
            date = parts[30] if len(parts) > 30 else ""
            tm = parts[31] if len(parts) > 31 else ""
            nan5 = (float("nan"),) * 5
            bid_volumes = (
                tuple(_to_float(parts[i]) for i in (10, 12, 14, 16, 18)) if len(parts) > 18 else nan5
            )
            bid_prices = (
                tuple(_to_float(parts[i]) for i in (11, 13, 15, 17, 19)) if len(parts) > 19 else nan5
            )
            ask_volumes = (
                tuple(_to_float(parts[i]) for i in (20, 22, 24, 26, 28)) if len(parts) > 28 else nan5
            )
            ask_prices = (
                tuple(_to_float(parts[i]) for i in (21, 23, 25, 27, 29)) if len(parts) > 29 else nan5
            )
            quotes[sym.lower()] = Quote(
                symbol=sym.lower(),
                name=name,
                price=price,
                prev_close=prev_close,
                open_price=open_price,
                high=high,
                low=low,
                volume=volume,
                amount=amount,
                date=date,
                time=tm,
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
            )
        except Exception:
            continue
    return quotes


def fetch_kline_data(
    symbol: str,
    scale: str = "day",
    count: int = 320,
    timeout_s: float = 8.0,
) -> List[Candle]:
    sym = normalize_symbol(symbol)
    if not sym:
        return []

    if scale == "day":
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
            + urllib.parse.quote(f"{sym},day,,,{max(count, 1)},qfq")
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            obj = json.loads(raw)
            data = (obj.get("data") or {}).get(sym) or {}
            keys_to_try = ["qfqday", "day"]
            series = []
            for k in keys_to_try:
                if k in data:
                    series = data[k]
                    break

            candles: List[Candle] = []
            if not series:
                return []

            for row in series:
                if not row or len(row) < 6:
                    continue
                dt = str(row[0])
                o = _to_float(str(row[1]))
                c = _to_float(str(row[2]))
                h = _to_float(str(row[3]))
                l = _to_float(str(row[4]))
                v = _to_float(str(row[5]))
                candles.append(Candle(date=dt, open_price=o, close_price=c, high=h, low=l, volume=v))
            return candles
        except Exception:
            return []

    elif scale.startswith("m"):
        sina_scale = scale[1:]
        url = (
            f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?"
            f"symbol={sym}&scale={sina_scale}&ma=no&datalen={max(count, 1)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")

            data = json.loads(raw)
            if not isinstance(data, list):
                return []

            candles: List[Candle] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                dt = item.get("day", "")
                o = _to_float(item.get("open", ""))
                c = _to_float(item.get("close", ""))
                h = _to_float(item.get("high", ""))
                l = _to_float(item.get("low", ""))
                v = _to_float(item.get("volume", ""))
                candles.append(Candle(date=dt, open_price=o, close_price=c, high=h, low=l, volume=v))
            return candles
        except Exception:
            return []

    return []


def fetch_tencent_daily_candles(
    symbol: str,
    count: int = 320,
    timeout_s: float = 8.0,
) -> List[Candle]:
    return fetch_kline_data(symbol, scale="day", count=count, timeout_s=timeout_s)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _safe_pct(numer: float, denom: float) -> float:
    if denom == 0 or math.isnan(denom) or math.isnan(numer):
        return float("nan")
    return numer / denom


def _last_sma(values: Sequence[float], period: int) -> float:
    if period <= 0 or len(values) < period:
        return float("nan")
    window = values[-period:]
    if any(math.isnan(v) for v in window):
        return float("nan")
    return sum(window) / period


def _last_ema(values: Sequence[float], period: int) -> float:
    if period <= 0 or len(values) < period:
        return float("nan")
    if any(math.isnan(v) for v in values[-period:]):
        return float("nan")
    alpha = 2.0 / (period + 1.0)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        if math.isnan(v):
            return float("nan")
        ema = alpha * v + (1.0 - alpha) * ema
    return ema


def _ema_series(values: Sequence[float], period: int) -> List[float]:
    if period <= 0 or len(values) < period:
        return [float("nan")] * len(values)
    alpha = 2.0 / (period + 1.0)
    series: List[float] = [float("nan")] * len(values)
    start = sum(values[:period]) / period
    ema = start
    series[period - 1] = ema
    for i in range(period, len(values)):
        v = values[i]
        if math.isnan(v):
            series[i] = float("nan")
            continue
        ema = alpha * v + (1.0 - alpha) * ema
        series[i] = ema
    return series


def _rsi(values: Sequence[float], period: int = 14) -> float:
    if period <= 0 or len(values) < period + 1:
        return float("nan")
    window = values[-(period + 1) :]
    if any(math.isnan(v) for v in window):
        return float("nan")
    gains = 0.0
    losses = 0.0
    for i in range(1, len(window)):
        delta = window[i] - window[i - 1]
        if delta > 0:
            gains += delta
        elif delta < 0:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        if avg_gain == 0:
            return 50.0
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _max_drawdown(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    peak = float("-inf")
    mdd = 0.0
    for v in values:
        if math.isnan(v) or v <= 0:
            return float("nan")
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < mdd:
            mdd = dd
    return mdd


def _json_safe(obj: object) -> object:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


def _wilder_atr(candles: Sequence[Candle], period: int = 14) -> float:
    if period <= 0 or len(candles) < period + 1:
        return float("nan")
    tr: List[float] = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        prev_c = candles[i - 1].close_price
        if any(math.isnan(x) for x in (h, l, prev_c)):
            return float("nan")
        tr.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    if len(tr) < period:
        return float("nan")
    atr = sum(tr[:period]) / period
    for v in tr[period:]:
        atr = (atr * (period - 1) + v) / period
    return atr


def _adx(candles: Sequence[Candle], period: int = 14) -> Dict[str, float]:
    if period <= 0 or len(candles) < period + 2:
        return {"adx": float("nan"), "di_plus": float("nan"), "di_minus": float("nan")}

    trs: List[float] = []
    dm_plus: List[float] = []
    dm_minus: List[float] = []

    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        prev_h = candles[i - 1].high
        prev_l = candles[i - 1].low
        prev_c = candles[i - 1].close_price
        if any(math.isnan(x) for x in (h, l, prev_h, prev_l, prev_c)):
            return {"adx": float("nan"), "di_plus": float("nan"), "di_minus": float("nan")}
        up_move = h - prev_h
        down_move = prev_l - l
        dmp = up_move if up_move > down_move and up_move > 0 else 0.0
        dmm = down_move if down_move > up_move and down_move > 0 else 0.0
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
        dm_plus.append(dmp)
        dm_minus.append(dmm)

    if len(trs) < period:
        return {"adx": float("nan"), "di_plus": float("nan"), "di_minus": float("nan")}

    tr14 = sum(trs[:period])
    dmp14 = sum(dm_plus[:period])
    dmm14 = sum(dm_minus[:period])

    def di(tr_sum: float, dm_sum: float) -> float:
        if tr_sum == 0:
            return float("nan")
        return 100.0 * (dm_sum / tr_sum)

    di_plus = di(tr14, dmp14)
    di_minus = di(tr14, dmm14)

    def dx(dip: float, dim: float) -> float:
        if math.isnan(dip) or math.isnan(dim) or (dip + dim) == 0:
            return float("nan")
        return 100.0 * (abs(dip - dim) / (dip + dim))

    dxs: List[float] = [dx(di_plus, di_minus)]

    for i in range(period, len(trs)):
        tr14 = tr14 - (tr14 / period) + trs[i]
        dmp14 = dmp14 - (dmp14 / period) + dm_plus[i]
        dmm14 = dmm14 - (dmm14 / period) + dm_minus[i]
        dip = di(tr14, dmp14)
        dim = di(tr14, dmm14)
        dxs.append(dx(dip, dim))
        di_plus = dip
        di_minus = dim

    valid = [v for v in dxs if not math.isnan(v)]
    if len(valid) < period:
        return {"adx": float("nan"), "di_plus": di_plus, "di_minus": di_minus}

    adx = sum(valid[:period]) / period
    for v in valid[period:]:
        adx = (adx * (period - 1) + v) / period

    return {"adx": adx, "di_plus": di_plus, "di_minus": di_minus}


def _obv(closes: Sequence[float], volumes: Sequence[float]) -> float:
    if len(closes) < 2 or len(closes) != len(volumes):
        return float("nan")
    obv = 0.0
    for i in range(1, len(closes)):
        c0, c1, v = closes[i - 1], closes[i], volumes[i]
        if any(math.isnan(x) for x in (c0, c1, v)):
            return float("nan")
        if c1 > c0:
            obv += v
        elif c1 < c0:
            obv -= v
    return obv


def _mfi(candles: Sequence[Candle], period: int = 14) -> float:
    if period <= 0 or len(candles) < period + 1:
        return float("nan")
    window = candles[-(period + 1) :]
    pos = 0.0
    neg = 0.0
    for i in range(1, len(window)):
        prev = window[i - 1]
        cur = window[i]
        tp_prev = (prev.high + prev.low + prev.close_price) / 3.0
        tp_cur = (cur.high + cur.low + cur.close_price) / 3.0
        if any(math.isnan(x) for x in (tp_prev, tp_cur, cur.volume)):
            return float("nan")
        flow = tp_cur * cur.volume
        if tp_cur > tp_prev:
            pos += flow
        elif tp_cur < tp_prev:
            neg += flow
    if neg == 0:
        if pos == 0:
            return 50.0
        return 100.0
    mr = pos / neg
    return 100.0 - (100.0 / (1.0 + mr))


def _rolling_beta_alpha(
    asset_returns: Sequence[float],
    bench_returns: Sequence[float],
) -> Dict[str, float]:
    if len(asset_returns) < 2 or len(asset_returns) != len(bench_returns):
        return {"beta": float("nan"), "corr": float("nan"), "alpha": float("nan")}

    ar = [x for x in asset_returns if not math.isnan(x)]
    br = [x for x in bench_returns if not math.isnan(x)]
    if len(ar) != len(br) or len(ar) < 2:
        return {"beta": float("nan"), "corr": float("nan"), "alpha": float("nan")}

    ma = _mean(ar)
    mb = _mean(br)
    cov = sum((a - ma) * (b - mb) for a, b in zip(ar, br)) / (len(ar) - 1)
    var_b = sum((b - mb) ** 2 for b in br) / (len(br) - 1)
    if var_b == 0:
        return {"beta": float("nan"), "corr": float("nan"), "alpha": float("nan")}
    beta = cov / var_b
    std_a = _stdev(ar)
    std_b = _stdev(br)
    corr = float("nan")
    if not math.isnan(std_a) and not math.isnan(std_b) and std_a != 0 and std_b != 0:
        corr = cov / (std_a * std_b)
    alpha = (ma - beta * mb) * 252.0
    return {"beta": beta, "corr": corr, "alpha": alpha}


def analyze_daily_candles(
    candles: Sequence[Candle],
    bench_candles: Optional[Sequence[Candle]] = None,
) -> Dict[str, object]:
    if not candles:
        return {"error": "no_daily_data"}

    closes = [c.close_price for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    last_close = closes[-1]
    pct_1d = float("nan")
    if len(closes) >= 2:
        pct_1d = _safe_pct(closes[-1] - closes[-2], closes[-2])

    pct_5d = _safe_pct(closes[-1] - closes[-6], closes[-6]) if len(closes) >= 6 else float("nan")
    pct_20d = _safe_pct(closes[-1] - closes[-21], closes[-21]) if len(closes) >= 21 else float("nan")
    pct_60d = _safe_pct(closes[-1] - closes[-61], closes[-61]) if len(closes) >= 61 else float("nan")
    pct_252d = _safe_pct(closes[-1] - closes[-253], closes[-253]) if len(closes) >= 253 else float("nan")

    sma_20 = _last_sma(closes, 20)
    sma_60 = _last_sma(closes, 60)
    ema_20 = _last_ema(closes, 20)
    rsi_14 = _rsi(closes, 14)
    mfi_14 = _mfi(candles, 14)

    atr_14 = _wilder_atr(candles, 14)
    adx_out = _adx(candles, 14)

    boll_mid = float("nan")
    boll_upper = float("nan")
    boll_lower = float("nan")
    boll_bw = float("nan")
    boll_pos = float("nan")
    if len(closes) >= 20:
        w = closes[-20:]
        m = _mean(w)
        s = _stdev(w)
        if not math.isnan(m) and not math.isnan(s):
            boll_mid = m
            boll_upper = m + 2.0 * s
            boll_lower = m - 2.0 * s
            if boll_mid != 0:
                boll_bw = (boll_upper - boll_lower) / boll_mid
            span = boll_upper - boll_lower
            if span != 0:
                boll_pos = (last_close - boll_lower) / span

    macd_line = float("nan")
    macd_signal = float("nan")
    macd_hist = float("nan")
    if len(closes) >= 26:
        ema12_series = _ema_series(closes, 12)
        ema26_series = _ema_series(closes, 26)
        macd_series = []
        for a, b in zip(ema12_series, ema26_series):
            if math.isnan(a) or math.isnan(b):
                macd_series.append(float("nan"))
            else:
                macd_series.append(a - b)
        macd_line = macd_series[-1]
        valid_macd = [v for v in macd_series if not math.isnan(v)]
        signal_series = _ema_series(valid_macd, 9)
        if signal_series and not math.isnan(signal_series[-1]):
            macd_signal = signal_series[-1]
            if not math.isnan(macd_line):
                macd_hist = macd_line - macd_signal

    realized_vol_20 = float("nan")
    if len(closes) >= 21:
        rets = []
        for i in range(-20, 0):
            a = closes[i - 1]
            b = closes[i]
            if a and not math.isnan(a) and not math.isnan(b):
                rets.append((b - a) / a)
        realized_vol_20 = _stdev(rets)

    vol_ann_20 = realized_vol_20 * math.sqrt(252.0) if not math.isnan(realized_vol_20) else float("nan")

    mdd_252 = _max_drawdown(closes[-252:]) if len(closes) >= 2 else float("nan")

    hi_252 = max(highs[-252:]) if len(highs) >= 1 else float("nan")
    lo_252 = min(lows[-252:]) if len(lows) >= 1 else float("nan")
    dist_hi_252 = _safe_pct(last_close - hi_252, hi_252) if not math.isnan(hi_252) else float("nan")
    dist_lo_252 = _safe_pct(last_close - lo_252, lo_252) if not math.isnan(lo_252) else float("nan")
    pos_252 = float("nan")
    if not math.isnan(hi_252) and not math.isnan(lo_252) and hi_252 != lo_252:
        pos_252 = (last_close - lo_252) / (hi_252 - lo_252)

    avg_vol_20 = _mean(volumes[-20:]) if len(volumes) >= 20 else float("nan")
    vol_ratio_20 = _safe_pct(volumes[-1], avg_vol_20) if not math.isnan(avg_vol_20) else float("nan")

    obv = _obv(closes, volumes)

    beta = float("nan")
    corr = float("nan")
    alpha = float("nan")
    rs_20 = float("nan")
    if bench_candles:
        bench_by_date = {c.date: c.close_price for c in bench_candles}
        asset_ret: List[float] = []
        bench_ret: List[float] = []
        prev_asset = None
        prev_bench = None
        for c in candles:
            b = bench_by_date.get(c.date)
            if b is None or math.isnan(b) or math.isnan(c.close_price):
                prev_asset = None
                prev_bench = None
                continue
            if prev_asset is not None and prev_bench is not None:
                asset_ret.append(_safe_pct(c.close_price - prev_asset, prev_asset))
                bench_ret.append(_safe_pct(b - prev_bench, prev_bench))
            prev_asset = c.close_price
            prev_bench = b
        window = 60
        if len(asset_ret) >= window and len(bench_ret) >= window:
            out = _rolling_beta_alpha(asset_ret[-window:], bench_ret[-window:])
            beta = out["beta"]
            corr = out["corr"]
            alpha = out["alpha"]
        if len(asset_ret) >= 20 and len(bench_ret) >= 20:
            rs_20 = sum(asset_ret[-20:]) - sum(bench_ret[-20:])

    return {
        "samples": len(candles),
        "last_date": candles[-1].date,
        "close": last_close,
        "pct_1d": pct_1d,
        "pct_5d": pct_5d,
        "pct_20d": pct_20d,
        "pct_60d": pct_60d,
        "pct_252d": pct_252d,
        "sma_20": sma_20,
        "sma_60": sma_60,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "mfi_14": mfi_14,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "boll_mid": boll_mid,
        "boll_upper": boll_upper,
        "boll_lower": boll_lower,
        "boll_bw": boll_bw,
        "boll_pos": boll_pos,
        "atr_14": atr_14,
        "adx_14": adx_out["adx"],
        "di_plus_14": adx_out["di_plus"],
        "di_minus_14": adx_out["di_minus"],
        "realized_vol_20": realized_vol_20,
        "vol_annualized_20": vol_ann_20,
        "mdd_252": mdd_252,
        "hi_252": hi_252,
        "lo_252": lo_252,
        "dist_hi_252": dist_hi_252,
        "dist_lo_252": dist_lo_252,
        "pos_252": pos_252,
        "avg_vol_20": avg_vol_20,
        "vol_ratio_20": vol_ratio_20,
        "obv": obv,
        "beta_60": beta,
        "corr_60": corr,
        "alpha_60": alpha,
        "rs_20": rs_20,
    }


def analyze_quote(
    quote: Quote,
    price_history: Sequence[Tuple[float, float]],
    interval_s: float,
) -> Dict[str, object]:
    price = quote.price
    prev_close = quote.prev_close
    pct = _safe_pct(price - prev_close, prev_close)

    prices = [p for _, p in price_history if not math.isnan(p)]
    last_n = prices[-240:] if len(prices) > 240 else prices[:]

    def last_ret(k: int) -> float:
        if len(last_n) < k + 1:
            return float("nan")
        base = last_n[-(k + 1)]
        return _safe_pct(last_n[-1] - base, base)

    ret_1 = last_ret(1)
    ret_3 = last_ret(3)
    ret_5 = last_ret(5)
    ret_10 = last_ret(10)
    ret_20 = last_ret(20)
    ret_60 = last_ret(60)

    ret = []
    for i in range(1, len(last_n)):
        a, b = last_n[i - 1], last_n[i]
        if a and not math.isnan(a) and not math.isnan(b):
            ret.append((b - a) / a)

    vol = _stdev(ret)
    vol_annualized = float("nan")
    if ret and interval_s > 0:
        per_day = max(int(4 * 60 * 60 / interval_s), 1)
        vol_annualized = vol * math.sqrt(252 * per_day)

    intraday_range = _safe_pct(quote.high - quote.low, prev_close)
    open_change = _safe_pct(price - quote.open_price, quote.open_price)
    high_change = _safe_pct(quote.high - prev_close, prev_close)
    low_change = _safe_pct(quote.low - prev_close, prev_close)
    vwap = _safe_pct(quote.amount, quote.volume) if quote.volume and not math.isnan(quote.volume) else float("nan")

    sma_5 = _last_sma(last_n, 5)
    sma_10 = _last_sma(last_n, 10)
    sma_20 = _last_sma(last_n, 20)
    sma_60 = _last_sma(last_n, 60)

    ema_12 = _last_ema(last_n, 12)
    ema_26 = _last_ema(last_n, 26)

    rsi_14 = _rsi(last_n, 14)

    macd_line = float("nan")
    macd_signal = float("nan")
    macd_hist = float("nan")
    if len(last_n) >= 26:
        ema12_series = _ema_series(last_n, 12)
        ema26_series = _ema_series(last_n, 26)
        macd_series: List[float] = []
        for a, b in zip(ema12_series, ema26_series):
            if math.isnan(a) or math.isnan(b):
                macd_series.append(float("nan"))
            else:
                macd_series.append(a - b)
        macd_line = macd_series[-1]
        signal_series = _ema_series([v for v in macd_series if not math.isnan(v)], 9)
        if signal_series and not math.isnan(signal_series[-1]):
            macd_signal = signal_series[-1]
            if not math.isnan(macd_line):
                macd_hist = macd_line - macd_signal

    boll_mid = float("nan")
    boll_upper = float("nan")
    boll_lower = float("nan")
    boll_bw = float("nan")
    boll_pos = float("nan")
    if len(last_n) >= 20:
        window = last_n[-20:]
        m = _mean(window)
        s = _stdev(window)
        if not math.isnan(m) and not math.isnan(s):
            boll_mid = m
            boll_upper = m + 2.0 * s
            boll_lower = m - 2.0 * s
            if boll_mid != 0:
                boll_bw = (boll_upper - boll_lower) / boll_mid
            span = boll_upper - boll_lower
            if span != 0:
                boll_pos = (price - boll_lower) / span

    zscore_20 = float("nan")
    if len(last_n) >= 20:
        window = last_n[-20:]
        m = _mean(window)
        s = _stdev(window)
        if not math.isnan(m) and not math.isnan(s) and s != 0:
            zscore_20 = (price - m) / s

    mdd_60 = float("nan")
    if len(last_n) >= 60:
        mdd_60 = _max_drawdown(last_n[-60:])
    elif len(last_n) >= 2:
        mdd_60 = _max_drawdown(last_n)

    return {
        "pct_change": pct,
        "open_change": open_change,
        "high_change": high_change,
        "low_change": low_change,
        "intraday_range": intraday_range,
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "ret_20": ret_20,
        "ret_60": ret_60,
        "sma_5": sma_5,
        "sma_10": sma_10,
        "sma_20": sma_20,
        "sma_60": sma_60,
        "ema_12": ema_12,
        "ema_26": ema_26,
        "rsi_14": rsi_14,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "boll_mid": boll_mid,
        "boll_upper": boll_upper,
        "boll_lower": boll_lower,
        "boll_bw": boll_bw,
        "boll_pos": boll_pos,
        "zscore_20": zscore_20,
        "mdd_60": mdd_60,
        "vwap": vwap,
        "realized_vol": vol,
        "vol_annualized": vol_annualized,
        "samples": len(last_n),
    }


def _fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value * 100:+.2f}%"


def _fmt_num(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.2f}万"
    return f"{value:.0f}"


def _orderbook_to_dict(q: Quote) -> Dict[str, object]:
    bids = [{"price": p, "volume": v} for p, v in zip(q.bid_prices, q.bid_volumes)]
    asks = [{"price": p, "volume": v} for p, v in zip(q.ask_prices, q.ask_volumes)]
    return {"bid": bids, "ask": asks}

def _request_date_time(ts: float) -> Tuple[str, str]:
    dt = datetime.fromtimestamp(float(ts))
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def _format_record_dt(rec: Dict[str, object]) -> str:
    ts = rec.get("ts")
    if ts is not None:
        try:
            dt = datetime.fromtimestamp(float(ts))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return (str(rec.get("date") or "") + " " + str(rec.get("time") or "")).strip()


def _is_auction_time(date_str: str, time_str: str) -> bool:
    if not date_str or not time_str:
        return False
    t = time_str.strip()
    return "09:15:00" <= t < "09:30:00"


def _auction_sample_from_quote(q: Quote, ts: float) -> Dict[str, object]:
    return {
        "ts": ts,
        "t": (q.date + " " + q.time).strip(),
        "symbol": q.symbol,
        "name": q.name,
        "price": q.price,
        "volume": q.volume,
        "amount": q.amount,
        "bid_prices": list(q.bid_prices),
        "bid_volumes": list(q.bid_volumes),
        "ask_prices": list(q.ask_prices),
        "ask_volumes": list(q.ask_volumes),
    }


def _fmt_levels(prices: Sequence[float], volumes: Sequence[float]) -> str:
    out: List[str] = []
    for p, v in zip(prices, volumes):
        if math.isnan(p) or math.isnan(v):
            out.append("NaN")
        else:
            out.append(f"{p:.3f}@{_fmt_num(v)}")
    return " ".join(out)


def _candle_to_dict(c: Candle) -> Dict[str, object]:
    return {
        "t": c.date,
        "o": c.open_price,
        "h": c.high,
        "l": c.low,
        "c": c.close_price,
        "v": c.volume,
    }


def _filter_m5_from_0930(candles: Sequence[Candle], day: str) -> List[Candle]:
    if not day:
        day = datetime.now().strftime("%Y-%m-%d")
    start = f"{day} 09:30:00"
    out: List[Candle] = []
    for c in candles:
        if c.date.startswith(day) and c.date >= start:
            out.append(c)
    return out


def _filter_window(candles: Sequence[Candle], day: str, start_hm: str, end_hm: str) -> List[Candle]:
    if not day:
        day = datetime.now().strftime("%Y-%m-%d")
    start = f"{day} {start_hm}:00"
    end = f"{day} {end_hm}:00"
    out: List[Candle] = []
    for c in candles:
        if not c.date.startswith(day):
            continue
        if c.date < start:
            continue
        if c.date >= end:
            continue
        out.append(c)
    return out


def _auction_summary(candles: Sequence[Candle]) -> Dict[str, object]:
    if not candles:
        return {"samples": 0}
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close_price for c in candles]
    volumes = [c.volume for c in candles]
    total_vol = sum(v for v in volumes if not math.isnan(v))
    vwap = float("nan")
    if total_vol > 0:
        vwap = sum(c * v for c, v in zip(closes, volumes) if not math.isnan(c) and not math.isnan(v)) / total_vol
    return {
        "samples": len(candles),
        "start": candles[0].date,
        "end": candles[-1].date,
        "open": candles[0].open_price,
        "close": candles[-1].close_price,
        "high": max(highs) if highs else float("nan"),
        "low": min(lows) if lows else float("nan"),
        "volume_sum": total_vol,
        "vwap": vwap,
    }


def _parse_codes_from_input(text: str) -> List[str]:
    s = text.replace("，", ",").replace(" ", ",").replace("\t", ",")
    codes = [c.strip() for c in s.split(",") if c.strip()]
    return codes


def run_once(
    codes: Sequence[str],
    timeout_s: float,
    daily: bool,
    daily_len: int,
    bench: str,
) -> int:
    quotes = fetch_sina_quotes(codes, timeout_s=timeout_s)
    if not quotes:
        print("未获取到行情数据，请检查股票代码或网络连接。")
        return 2

    bench_candles = fetch_tencent_daily_candles(bench, count=daily_len) if daily and bench else []

    m5_len = 288

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now}  行情快照")
    for raw in codes:
        sym = normalize_symbol(raw)
        q = quotes.get(sym)
        if not q:
            print(f"- {raw}: 未找到")
            continue

        # Fetch m5 candles to fill history
        m5_candles = fetch_kline_data(sym, scale="m5", count=m5_len, timeout_s=timeout_s)

        history_prices = []
        for c in m5_candles:
            try:
                dt = datetime.strptime(c.date, "%Y-%m-%d %H:%M:%S")
                history_prices.append((dt.timestamp(), c.close_price))
            except Exception:
                pass
        history_prices.append((time.time(), q.price))

        intraday_metrics = analyze_quote(q, price_history=history_prices, interval_s=1.0)

        daily_metrics = {}
        if daily:
            d_candles = fetch_tencent_daily_candles(sym, count=daily_len)
            daily_metrics = analyze_daily_candles(d_candles, bench_candles)

        # m5_candles fetched above
        m5_metrics = analyze_daily_candles(m5_candles, None) if m5_candles else {}

        print(
            f"- {q.name}({q.symbol})  现价:{q.price:.3f}  昨收:{q.prev_close:.3f}  今开:{q.open_price:.3f}  最高:{q.high:.3f}  最低:{q.low:.3f}"
        )
        print(
            f"  成交量:{_fmt_num(q.volume)}  成交额:{_fmt_num(q.amount)}  实时涨跌:{_fmt_pct(intraday_metrics['pct_change'])}"
        )
        print(f"  竞价 买:{_fmt_levels(q.bid_prices, q.bid_volumes)}")
        print(f"  竞价 卖:{_fmt_levels(q.ask_prices, q.ask_volumes)}")

        intraday_text = json.dumps(_json_safe(intraday_metrics), ensure_ascii=False, sort_keys=True, indent=2)
        print("  实时指标：")
        for line in intraday_text.splitlines():
            print("  " + line)

        if m5_metrics:
            m5_text = json.dumps(_json_safe(m5_metrics), ensure_ascii=False, sort_keys=True, indent=2)
            m5_samples = m5_metrics.get("samples")
            print(f"  5分钟K线指标（samples={m5_samples}）：")
            for line in m5_text.splitlines():
                print("  " + line)

        m5_today = _filter_m5_from_0930(m5_candles, q.date)
        print(f"  今日5分钟K线（9:30至今 bars={len(m5_today)}）：")
        for c in m5_today:
            print(
                f"  {c.date}  O:{c.open_price:.3f} H:{c.high:.3f} L:{c.low:.3f} C:{c.close_price:.3f} V:{_fmt_num(c.volume)}"
            )

        m1_candles = fetch_kline_data(sym, scale="m1", count=200, timeout_s=timeout_s)

        auction_bars = _filter_window(m1_candles, q.date, "09:15", "09:30")
        auction_summary = _auction_summary(auction_bars)
        auction_sample = _auction_sample_from_quote(q, time.time()) if _is_auction_time(q.date, q.time) else None
        print(f"  集合竞价（9:15-9:30 bars={len(auction_bars)}）：")
        auction_text = json.dumps(_json_safe(auction_summary), ensure_ascii=False, sort_keys=True, indent=2)
        for line in auction_text.splitlines():
            print("  " + line)
        if auction_sample is not None:
            sample_text = json.dumps(_json_safe(auction_sample), ensure_ascii=False, sort_keys=True, indent=2)
            print("  集合竞价快照：")
            for line in sample_text.splitlines():
                print("  " + line)
        for c in auction_bars:
            print(
                f"  {c.date}  O:{c.open_price:.3f} H:{c.high:.3f} L:{c.low:.3f} C:{c.close_price:.3f} V:{_fmt_num(c.volume)}"
            )

        if daily:
            daily_text = json.dumps(_json_safe(daily_metrics), ensure_ascii=False, sort_keys=True, indent=2)
            daily_samples = daily_metrics.get("samples")
            print(f"  日K指标（samples={daily_samples}）：")
            for line in daily_text.splitlines():
                print("  " + line)
    return 0


def run_stream(
    codes: Sequence[str],
    interval_s: float,
    timeout_s: float,
    clear_screen: bool,
    jsonl: bool,
    ticks: int,
    daily: bool,
    daily_len: int,
    bench: str,
) -> int:
    symbols = [normalize_symbol(c) for c in codes]
    symbols = [s for s in symbols if s]
    if not symbols:
        print("请输入至少一个有效股票代码，例如：600000、000001、sh600000。")
        return 2

    history: Dict[str, Deque[Tuple[float, float]]] = {s: deque(maxlen=240) for s in symbols}
    bench_candles = fetch_tencent_daily_candles(bench, count=daily_len) if daily and bench else []

    m5_len = 288

    auction_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
    remaining = ticks
    while True:
        started = time.time()
        quotes = fetch_sina_quotes(symbols, timeout_s=timeout_s)

        ts = time.time()
        if clear_screen and not jsonl:
            sys.stdout.write("\033[2J\033[H")

        header_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        if not jsonl:
            print(f"{header_time}  实时监控  interval={interval_s:.1f}s  codes={','.join(symbols)}")

        for sym in symbols:
            q = quotes.get(sym)
            if not q:
                if not jsonl:
                    print(f"- {sym}: 未获取到行情")
                continue
            history[sym].append((ts, q.price))

            intraday_metrics = analyze_quote(q, history[sym], interval_s=interval_s)

            daily_metrics = {}
            if daily:
                d_candles = fetch_tencent_daily_candles(sym, count=daily_len)
                daily_metrics = analyze_daily_candles(d_candles, bench_candles)

            m5_candles = fetch_kline_data(sym, scale="m5", count=m5_len, timeout_s=timeout_s)
            m5_metrics = analyze_daily_candles(m5_candles, None) if m5_candles else {}
            m5_bars_today = _filter_m5_from_0930(m5_candles, q.date)
            auction_sample = _auction_sample_from_quote(q, ts) if _is_auction_time(q.date, q.time) else None

            cache_key = (sym, q.date)
            cached = auction_cache.get(cache_key)
            if cached is None:
                m1_candles = fetch_kline_data(sym, scale="m1", count=200, timeout_s=timeout_s)
                auction_bars = _filter_window(m1_candles, q.date, "09:15", "09:30")
                cached = {
                    "bars": [_candle_to_dict(c) for c in auction_bars],
                    "summary": _auction_summary(auction_bars),
                }
                auction_cache[cache_key] = cached

            if jsonl:
                out = {
                    "ts": ts,
                    "symbol": q.symbol,
                    "name": q.name,
                    "price": q.price,
                    "prev_close": q.prev_close,
                    "open": q.open_price,
                    "high": q.high,
                    "low": q.low,
                    "volume": q.volume,
                    "amount": q.amount,
                    "orderbook": _json_safe(_orderbook_to_dict(q)),
                    "intraday_metrics": _json_safe(intraday_metrics),
                    "daily_metrics": _json_safe(daily_metrics) if daily else {},
                    "m5_metrics": _json_safe(m5_metrics) if m5_metrics else {},
                    "m5_bars_today": _json_safe([_candle_to_dict(c) for c in m5_bars_today]),
                    "auction_bars": _json_safe(cached["bars"]),
                    "auction_summary": _json_safe(cached["summary"]),
                    "auction_sample": _json_safe(auction_sample) if auction_sample is not None else None,
                }
                sys.stdout.write(json.dumps(_json_safe(out), ensure_ascii=False) + "\n")
                continue

            print(
                f"- {q.name}({q.symbol})  现价:{q.price:.3f}  昨收:{q.prev_close:.3f}  今开:{q.open_price:.3f}  最高:{q.high:.3f}  最低:{q.low:.3f}"
            )
            print(
                f"  成交量:{_fmt_num(q.volume)}  成交额:{_fmt_num(q.amount)}  实时涨跌:{_fmt_pct(intraday_metrics['pct_change'])}"
            )
            print(f"  竞价 买:{_fmt_levels(q.bid_prices, q.bid_volumes)}")
            print(f"  竞价 卖:{_fmt_levels(q.ask_prices, q.ask_volumes)}")
            intraday_text = json.dumps(_json_safe(intraday_metrics), ensure_ascii=False, sort_keys=True, indent=2)
            print("  实时指标：")
            for line in intraday_text.splitlines():
                print("  " + line)

            if m5_metrics:
                m5_text = json.dumps(_json_safe(m5_metrics), ensure_ascii=False, sort_keys=True, indent=2)
                m5_samples = m5_metrics.get("samples")
                print(f"  5分钟K线指标（samples={m5_samples}）：")
                for line in m5_text.splitlines():
                    print("  " + line)

            print(f"  今日5分钟K线（9:30至今 bars={len(m5_bars_today)}）：")
            for c in m5_bars_today:
                print(
                    f"  {c.date}  O:{c.open_price:.3f} H:{c.high:.3f} L:{c.low:.3f} C:{c.close_price:.3f} V:{_fmt_num(c.volume)}"
                )

            auction_summary = cached["summary"]
            auction_bars = cached["bars"]
            print(f"  集合竞价（9:15-9:30 bars={len(auction_bars)}）：")
            auction_text = json.dumps(_json_safe(auction_summary), ensure_ascii=False, sort_keys=True, indent=2)
            for line in auction_text.splitlines():
                print("  " + line)
            if auction_sample is not None:
                sample_text = json.dumps(_json_safe(auction_sample), ensure_ascii=False, sort_keys=True, indent=2)
                print("  集合竞价快照：")
                for line in sample_text.splitlines():
                    print("  " + line)
            for row in auction_bars:
                dt = str(row.get("t", ""))
                o = float(row.get("o", float("nan")))
                h = float(row.get("h", float("nan")))
                l = float(row.get("l", float("nan")))
                c = float(row.get("c", float("nan")))
                v = float(row.get("v", float("nan")))
                print(f"  {dt}  O:{o:.3f} H:{h:.3f} L:{l:.3f} C:{c:.3f} V:{_fmt_num(v)}")

            if daily:
                daily_text = json.dumps(_json_safe(daily_metrics), ensure_ascii=False, sort_keys=True, indent=2)
                daily_samples = daily_metrics.get("samples")
                print(f"  日K指标（samples={daily_samples}）：")
                for line in daily_text.splitlines():
                    print("  " + line)

        if remaining > 0:
            remaining -= 1
            if remaining == 0:
                return 0

        elapsed = time.time() - started
        sleep_s = max(interval_s - elapsed, 0.0)
        time.sleep(sleep_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="stock_assistant")
    parser.add_argument(
        "--codes",
        default="",
        help="股票代码，逗号分隔，例如 600000,000001 或 sh600000,sz000001",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="刷新间隔秒数，默认 5")
    parser.add_argument("--timeout", type=float, default=5.0, help="网络超时秒数，默认 5")
    parser.add_argument("--once", action="store_true", help="只拉取一次并输出")
    parser.add_argument("--clear", action="store_true", help="每次刷新清屏")
    parser.add_argument("--jsonl", action="store_true", help="按 JSON Lines 输出，便于接入其他程序")
    parser.add_argument("--ticks", type=int, default=0, help="运行 N 次刷新后退出，默认 0 表示持续运行")
    parser.add_argument("--daily", dest="daily", action="store_true", help="抓取日K并计算日K指标")
    parser.add_argument("--no-daily", dest="daily", action="store_false", help="不抓取日K")
    parser.set_defaults(daily=True)
    parser.add_argument("--daily-len", type=int, default=320, help="日K长度，默认 320")
    parser.add_argument("--bench", type=str, default="sh000001", help="基准指数代码，默认 sh000001")
    parser.add_argument("--client", action="store_true", help="启动Python客户端（交互/历史/复制）")
    parser.add_argument("--history-file", type=str, default="history.jsonl", help="历史数据文件，默认 history.jsonl")
    args = parser.parse_args(argv)

    codes = _parse_codes_from_input(args.codes) if args.codes else []
    if not codes and not args.client:
        try:
            text = input("请输入股票代码（逗号分隔，例如 600000,000001）：").strip()
        except EOFError:
            text = ""
        codes = _parse_codes_from_input(text)

    if args.once:
        return run_once(
            codes,
            timeout_s=args.timeout,
            daily=bool(args.daily),
            daily_len=max(int(args.daily_len), 1),
            bench=str(args.bench).strip(),
        )

    interval = max(args.interval, 0.5)
    if args.client:
        return client(
            codes=codes,
            interval_s=interval,
            timeout_s=float(args.timeout),
            ticks=max(int(args.ticks), 0),
            daily=bool(args.daily),
            daily_len=max(int(args.daily_len), 1),
            bench=str(args.bench).strip(),
            history_file=str(args.history_file).strip() or "history.jsonl",
        )
    return run_stream(
        codes=codes,
        interval_s=interval,
        timeout_s=args.timeout,
        clear_screen=args.clear,
        jsonl=args.jsonl,
        ticks=max(args.ticks, 0),
        daily=bool(args.daily),
        daily_len=max(int(args.daily_len), 1),
        bench=str(args.bench).strip(),
    )


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>股票助手</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; margin: 16px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; }
    input, button, select { font-size: 14px; padding: 8px; }
    input { min-width: 280px; }
    button { cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; margin-top: 12px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    .title { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
    .muted { color: #666; font-size: 12px; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f3f3f3; font-size: 12px; }
    .table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    .table th, .table td { border-bottom: 1px solid #eee; padding: 6px 4px; font-size: 12px; text-align: left; }
  </style>
</head>
<body>
  <h2>股票助手</h2>
  <div class="row">
    <input id="codes" placeholder="输入股票代码，例如：600519,000001 或 sh600519" />
    <button id="fetchBtn">获取/刷新</button>
    <input id="filter" placeholder="按股票代码过滤（可选）" />
    <button id="reloadBtn">刷新历史</button>
  </div>
  <div class="muted" id="status"></div>
  <div class="grid" id="cards"></div>

  <script>
    const el = (id) => document.getElementById(id);
    const statusEl = el("status");
    const cardsEl = el("cards");

    function normFilter(s) {
      return (s || "").trim().toLowerCase();
    }

    function fmtNum(x) {
      if (x === null || x === undefined) return "null";
      if (typeof x !== "number") return String(x);
      return Number.isFinite(x) ? x.toString() : "null";
    }

    function renderOrderbook(ob) {
      if (!ob) return "";
      const bid = (ob.bid || []).map(r => `${fmtNum(r.price)}@${fmtNum(r.volume)}`).join(" ");
      const ask = (ob.ask || []).map(r => `${fmtNum(r.price)}@${fmtNum(r.volume)}`).join(" ");
      return `买 ${bid}\\n卖 ${ask}`;
    }

    function renderBars(bars) {
      if (!Array.isArray(bars)) return "";
      return bars.map(b => `${b.t} O:${fmtNum(b.o)} H:${fmtNum(b.h)} L:${fmtNum(b.l)} C:${fmtNum(b.c)} V:${fmtNum(b.v)}`).join("\\n");
    }

    function renderAuctionSamples(samples) {
      if (!Array.isArray(samples)) return "";
      return samples.map(s => {
        const t = s.t || new Date((s.ts || 0) * 1000).toLocaleTimeString();
        const bid1 = Array.isArray(s.bid_prices) && s.bid_prices.length ? s.bid_prices[0] : null;
        const ask1 = Array.isArray(s.ask_prices) && s.ask_prices.length ? s.ask_prices[0] : null;
        return `${t} P:${fmtNum(s.price)} V:${fmtNum(s.volume)} A:${fmtNum(s.amount)} B1:${fmtNum(bid1)} A1:${fmtNum(ask1)}`;
      }).join("\\n");
    }

    function latestSnapshot(snapshots) {
      if (!Array.isArray(snapshots) || snapshots.length === 0) return null;
      return snapshots[snapshots.length - 1];
    }

    function cardForSymbol(symbol, bundle) {
      const snapshots = bundle.snapshots || [];
      const auctionSamples = bundle.auction_samples || [];
      const latest = latestSnapshot(snapshots) || (auctionSamples.length ? auctionSamples[auctionSamples.length - 1] : {});
      const name = latest.name || "";
      const ts = new Date((latest.ts || 0) * 1000).toLocaleString();
      const price = latest.price;
      const orderbook = latest.orderbook;
      const m5BarsToday = latest.m5_bars_today || [];
      const auctionBars = latest.auction_bars || [];

      const wrap = document.createElement("div");
      wrap.className = "card";

      const title = document.createElement("div");
      title.className = "title";
      title.innerHTML = `<div><strong>${name}</strong> <span class="pill">${symbol}</span><div class="muted">${ts} 现价: ${fmtNum(price)}</div></div>`;

      const btns = document.createElement("div");
      const copyLatest = document.createElement("button");
      copyLatest.textContent = "复制最新";
      copyLatest.onclick = async () => {
        const snap = latestSnapshot(snapshots) || latest;
        await navigator.clipboard.writeText(JSON.stringify(snap, null, 2));
        statusEl.textContent = `已复制 ${symbol} 最新数据`;
      };
      const copyAll = document.createElement("button");
      copyAll.textContent = "复制历史";
      copyAll.onclick = async () => {
        await navigator.clipboard.writeText(JSON.stringify({ snapshots, auction_samples: auctionSamples }, null, 2));
        statusEl.textContent = `已复制 ${symbol} 历史数据(snapshots=${snapshots.length}, auction_samples=${auctionSamples.length})`;
      };
      btns.appendChild(copyLatest);
      btns.appendChild(copyAll);
      title.appendChild(btns);

      const pre = document.createElement("pre");
      pre.className = "mono";
      const auctionText = auctionBars.length ? renderBars(auctionBars) : renderAuctionSamples(auctionSamples.slice(-200));
      const auctionCount = auctionBars.length ? `bars=${auctionBars.length}` : `samples=${auctionSamples.length}`;
      pre.textContent = `盘口\\n${renderOrderbook(orderbook)}\\n\\n集合竞价(9:15-9:30) ${auctionCount}\\n${auctionText}\\n\\n今日5分钟(9:30至今) bars=${m5BarsToday.length}\\n${renderBars(m5BarsToday)}`;

      const table = document.createElement("table");
      table.className = "table";
      table.innerHTML = "<thead><tr><th>#</th><th>时间</th><th>现价</th><th>m5bars</th><th>auctionbars</th></tr></thead>";
      const tbody = document.createElement("tbody");
      snapshots.slice(-20).forEach((r, i) => {
        const tr = document.createElement("tr");
        const t = new Date((r.ts || 0) * 1000).toLocaleTimeString();
        const m5n = (r.m5_bars_today || []).length;
        const an = (r.auction_bars || []).length;
        const idxBase = Math.max(snapshots.length - 20, 0);
        tr.innerHTML = `<td>${idxBase + i + 1}</td><td>${t}</td><td>${fmtNum(r.price)}</td><td>${m5n}</td><td>${an}</td>`;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);

      wrap.appendChild(title);
      wrap.appendChild(pre);
      wrap.appendChild(table);
      return wrap;
    }

    async function loadHistory() {
      const filter = normFilter(el("filter").value);
      const qs = filter ? `?symbol=${encodeURIComponent(filter)}` : "";
      const res = await fetch(`/api/history${qs}`);
      const data = await res.json();
      const grouped = new Map();
      for (const r of data.records || []) {
        const sym = (r.symbol || "").toLowerCase();
        if (!sym) continue;
        if (!grouped.has(sym)) grouped.set(sym, { snapshots: [], auction_samples: [] });
        const b = grouped.get(sym);
        const kind = (r.kind || "snapshot").toLowerCase();
        if (kind === "auction_sample") b.auction_samples.push(r);
        else b.snapshots.push(r);
      }
      cardsEl.innerHTML = "";
      for (const [sym, bundle] of grouped.entries()) {
        cardsEl.appendChild(cardForSymbol(sym, bundle));
      }
      statusEl.textContent = `历史记录: ${data.records ? data.records.length : 0}`;
    }

    async function fetchCodes() {
      const codes = el("codes").value.trim();
      if (!codes) return;
      statusEl.textContent = "请求中...";
      const res = await fetch(`/api/fetch?codes=${encodeURIComponent(codes)}`);
      const data = await res.json();
      if (data.error) {
        statusEl.textContent = data.error;
        return;
      }
      statusEl.textContent = `已获取: ${(data.symbols || []).join(", ")}`;
      await loadHistory();
    }

    el("fetchBtn").onclick = fetchCodes;
    el("reloadBtn").onclick = loadHistory;
    loadHistory();
  </script>
</body>
</html>
"""


def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_jsonl_records(path: str, max_lines: int = 5000) -> List[Dict[str, object]]:
    if max_lines <= 0:
        max_lines = 1
    if not os.path.exists(path):
        return []
    buf: Deque[Dict[str, object]] = deque(maxlen=max_lines)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    buf.append(obj)
    except Exception:
        return []
    return list(buf)


def _append_jsonl_record(path: str, rec: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(rec), ensure_ascii=False) + "\n")


def _copy_to_clipboard(text: str) -> bool:
    if not text:
        return False
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
    except Exception:
        return False
    return False


def client(
    codes: Sequence[str],
    interval_s: float,
    timeout_s: float,
    ticks: int,
    daily: bool,
    daily_len: int,
    bench: str,
    history_file: str,
) -> int:
    history_path = os.path.abspath(history_file)
    _ensure_dir(history_path)

    records: List[Dict[str, object]] = _load_jsonl_records(history_path)
    intraday_state: Dict[str, Deque[Tuple[float, float]]] = {}
    last_ts: Dict[str, float] = {}
    bench_candles = fetch_tencent_daily_candles(bench, count=daily_len) if daily and bench else []

    def append_record(rec: Dict[str, object]) -> None:
        records.append(rec)
        _append_jsonl_record(history_path, rec)

    def fetch_and_store(codes_text: str, status_cb: Optional[callable] = None) -> List[str]:
        raw_codes = _parse_codes_from_input(codes_text)
        symbols = [normalize_symbol(c) for c in raw_codes]
        symbols = [s for s in symbols if s]
        if not symbols:
            raise ValueError("请输入至少一个有效股票代码")

        if status_cb is not None:
            status_cb(f"拉取行情... symbols={len(symbols)}")
        quotes = fetch_sina_quotes(symbols, timeout_s=timeout_s)
        ts = time.time()
        req_date, req_time = _request_date_time(ts)
        out_syms: List[str] = []
        for i, sym in enumerate(symbols, start=1):
            q = quotes.get(sym)
            if not q:
                continue

            if status_cb is not None:
                status_cb(f"处理 {i}/{len(symbols)}: {sym}")
            
            # Fetch m5 candles early for history filling and metrics
            m5_candles = fetch_kline_data(sym, scale="m5", count=200, timeout_s=timeout_s)

            state = intraday_state.setdefault(sym, deque(maxlen=240))
            if not state and m5_candles:
                for c in m5_candles:
                    try:
                        dt = datetime.strptime(c.date, "%Y-%m-%d %H:%M:%S")
                        state.append((dt.timestamp(), c.close_price))
                    except Exception:
                        pass

            state.append((ts, q.price))
            interval = ts - last_ts.get(sym, ts - 1.0)
            if interval <= 0:
                interval = 1.0
            last_ts[sym] = ts

            intraday_metrics = analyze_quote(q, state, interval_s=interval)
            # m5_candles fetched above
            m5_metrics = analyze_daily_candles(m5_candles, None) if m5_candles else {}
            m5_bars_today = [_candle_to_dict(c) for c in _filter_m5_from_0930(m5_candles, q.date)]
            m1_candles = fetch_kline_data(sym, scale="m1", count=400, timeout_s=timeout_s)
            auction_bars_c = _filter_window(m1_candles, q.date, "09:15", "09:30")
            auction_bars = [_candle_to_dict(c) for c in auction_bars_c]
            auction_summary = _auction_summary(auction_bars_c)
            if _is_auction_time(q.date, q.time):
                append_record(
                    {
                        "kind": "auction_sample",
                        "ts": ts,
                        "date": req_date,
                        "time": req_time,
                        "quote_date": q.date,
                        "quote_time": q.time,
                        "symbol": q.symbol,
                        "name": q.name,
                        "price": q.price,
                        "volume": q.volume,
                        "amount": q.amount,
                        "bid_prices": list(q.bid_prices),
                        "bid_volumes": list(q.bid_volumes),
                        "ask_prices": list(q.ask_prices),
                        "ask_volumes": list(q.ask_volumes),
                    }
                )

            daily_metrics = {}
            if daily:
                d_candles = fetch_tencent_daily_candles(sym, count=daily_len)
                daily_metrics = analyze_daily_candles(d_candles, bench_candles)

            rec = {
                "kind": "snapshot",
                "ts": ts,
                "date": req_date,
                "time": req_time,
                "quote_date": q.date,
                "quote_time": q.time,
                "symbol": q.symbol,
                "name": q.name,
                "price": q.price,
                "prev_close": q.prev_close,
                "open": q.open_price,
                "high": q.high,
                "low": q.low,
                "volume": q.volume,
                "amount": q.amount,
                "orderbook": _orderbook_to_dict(q),
                "intraday_metrics": intraday_metrics,
                "m5_metrics": m5_metrics,
                "m5_bars_today": m5_bars_today,
                "auction_summary": auction_summary,
                "auction_bars": auction_bars,
                "daily_metrics": daily_metrics,
            }
            append_record(rec)
            out_syms.append(q.symbol)
        return out_syms

    def records_for_symbol(sym: str) -> List[Dict[str, object]]:
        s = normalize_symbol(sym) if sym else ""
        if not s:
            return []
        return [r for r in records if str(r.get("symbol", "")).lower() == s]

    def latest_snapshot(sym: str) -> Optional[Dict[str, object]]:
        for r in reversed(records_for_symbol(sym)):
            kind = str(r.get("kind") or "snapshot").lower()
            if kind != "auction_sample":
                return r
        return None

    if ticks > 0:
        symbols = [normalize_symbol(c) for c in codes]
        symbols = [s for s in symbols if s]
        if not symbols:
            print("请输入至少一个有效股票代码，例如：600000、000001、sh600000。")
            return 2
        codes_text = ",".join(symbols)
        for i in range(ticks):
            got = fetch_and_store(codes_text)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}  已获取: {','.join(got) if got else '无'}")
            if i + 1 < ticks:
                time.sleep(max(interval_s, 0.1))
        return 0

    print("Python 客户端模式")
    print("命令：")
    print("  fetch <codes>     获取/刷新，例如 fetch 600000,000001")
    print("  history <code> [n] 查看历史快照（默认20条）")
    print("  copy latest <code> 复制最新快照到剪贴板")
    print("  copy history <code> 复制该股票全部历史到剪贴板")
    print("  quit              退出")

    if codes:
        try:
            got = fetch_and_store(",".join(codes))
            print(f"已获取: {','.join(got) if got else '无'}")
        except Exception as e:
            print(str(e))

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            return 0
        if not line:
            continue
        low = line.lower()
        if low in ("q", "quit", "exit"):
            return 0
        if low in ("h", "help", "?"):
            print("命令：fetch / history / copy latest / copy history / quit")
            continue

        parts = line.split()
        if parts and parts[0].lower() == "fetch":
            text = line[len(parts[0]) :].strip()
            try:
                got = fetch_and_store(text)
                print(f"已获取: {','.join(got) if got else '无'}")
            except Exception as e:
                print(str(e))
            continue

        if parts and parts[0].lower() == "history":
            code = parts[1] if len(parts) >= 2 else ""
            n = 20
            if len(parts) >= 3:
                try:
                    n = max(int(parts[2]), 1)
                except Exception:
                    n = 20
            rs = records_for_symbol(code)
            if not rs:
                print("无历史")
                continue
            snaps = [r for r in rs if str(r.get("kind") or "snapshot").lower() != "auction_sample"]
            auc = [r for r in rs if str(r.get("kind") or "").lower() == "auction_sample"]
            view = snaps[-n:]
            print(f"symbol={normalize_symbol(code)} snapshots={len(snaps)} auction_samples={len(auc)}")
            for r in view:
                t = _format_record_dt(r)
                price = r.get("price")
                m5n = len(r.get("m5_bars_today") or [])
                an = len(r.get("auction_bars") or [])
                print(f"- {t}  price={price}  m5bars={m5n}  auctionbars={an}")
            continue

        if len(parts) >= 3 and parts[0].lower() == "copy" and parts[1].lower() == "latest":
            code = parts[2]
            r = latest_snapshot(code)
            if not r:
                print("无可复制数据")
                continue
            text = json.dumps(_json_safe(r), ensure_ascii=False, indent=2, sort_keys=True)
            ok = _copy_to_clipboard(text)
            print("已复制到剪贴板" if ok else text)
            continue

        if len(parts) >= 3 and parts[0].lower() == "copy" and parts[1].lower() == "history":
            code = parts[2]
            rs = records_for_symbol(code)
            if not rs:
                print("无可复制数据")
                continue
            text = json.dumps(_json_safe(rs), ensure_ascii=False, indent=2, sort_keys=True)
            ok = _copy_to_clipboard(text)
            print("已复制到剪贴板" if ok else text)
            continue

        try:
            got = fetch_and_store(line)
            print(f"已获取: {','.join(got) if got else '无'}")
        except Exception:
            print("未知命令，输入 help 查看用法")
    return 0


def gui(
    codes: Sequence[str],
    interval_s: float,
    timeout_s: float,
    daily: bool,
    daily_len: int,
    bench: str,
    history_file: str,
) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return gui_qt(
            codes=codes,
            interval_s=interval_s,
            timeout_s=timeout_s,
            daily=daily,
            daily_len=daily_len,
            bench=bench,
            history_file=history_file,
        )

    history_path = os.path.abspath(history_file)
    _ensure_dir(history_path)

    records: List[Dict[str, object]] = _load_jsonl_records(history_path)
    intraday_state: Dict[str, Deque[Tuple[float, float]]] = {}
    last_ts: Dict[str, float] = {}
    bench_candles = fetch_tencent_daily_candles(bench, count=daily_len) if daily and bench else []
    lock = threading.Lock()

    def append_record(rec: Dict[str, object]) -> None:
        with lock:
            records.append(rec)
            _append_jsonl_record(history_path, rec)

    def fetch_and_store(codes_text: str, status_cb: Optional[object] = None) -> List[str]:
        raw_codes = _parse_codes_from_input(codes_text)
        symbols = [normalize_symbol(c) for c in raw_codes]
        symbols = [s for s in symbols if s]
        if not symbols:
            raise ValueError("请输入至少一个有效股票代码")

        if callable(status_cb):
            status_cb(f"拉取行情... symbols={len(symbols)}")
        quotes = fetch_sina_quotes(symbols, timeout_s=timeout_s)
        ts = time.time()
        req_date, req_time = _request_date_time(ts)
        out_syms: List[str] = []
        for i, sym in enumerate(symbols, start=1):
            q = quotes.get(sym)
            if not q:
                continue

            if callable(status_cb):
                status_cb(f"处理 {i}/{len(symbols)}: {sym}")
            
            if callable(status_cb):
                status_cb(f"拉取5分钟K线: {sym}")
            m5_candles = fetch_kline_data(sym, scale="m5", count=200, timeout_s=timeout_s)

            state = intraday_state.setdefault(sym, deque(maxlen=240))
            if not state and m5_candles:
                for c in m5_candles:
                    try:
                        dt = datetime.strptime(c.date, "%Y-%m-%d %H:%M:%S")
                        state.append((dt.timestamp(), c.close_price))
                    except Exception:
                        pass

            state.append((ts, q.price))
            interval = ts - last_ts.get(sym, ts - 1.0)
            if interval <= 0:
                interval = 1.0
            last_ts[sym] = ts

            intraday_metrics = analyze_quote(q, state, interval_s=interval)
            
            # m5_candles fetched above
            m5_metrics = analyze_daily_candles(m5_candles, None) if m5_candles else {}
            m5_bars_today = [_candle_to_dict(c) for c in _filter_m5_from_0930(m5_candles, q.date)]

            if callable(status_cb):
                status_cb(f"拉取1分钟K线: {sym}")
            m1_candles = fetch_kline_data(sym, scale="m1", count=400, timeout_s=timeout_s)
            auction_bars_c = _filter_window(m1_candles, q.date, "09:15", "09:30")
            auction_bars = [_candle_to_dict(c) for c in auction_bars_c]
            auction_summary = _auction_summary(auction_bars_c)
            if _is_auction_time(q.date, q.time):
                append_record(
                    {
                        "kind": "auction_sample",
                        "ts": ts,
                        "date": req_date,
                        "time": req_time,
                        "quote_date": q.date,
                        "quote_time": q.time,
                        "symbol": q.symbol,
                        "name": q.name,
                        "price": q.price,
                        "volume": q.volume,
                        "amount": q.amount,
                        "bid_prices": list(q.bid_prices),
                        "bid_volumes": list(q.bid_volumes),
                        "ask_prices": list(q.ask_prices),
                        "ask_volumes": list(q.ask_volumes),
                    }
                )

            daily_metrics = {}
            if daily:
                if callable(status_cb):
                    status_cb(f"拉取日K: {sym}")
                d_candles = fetch_tencent_daily_candles(sym, count=daily_len)
                daily_metrics = analyze_daily_candles(d_candles, bench_candles)

            rec = {
                "kind": "snapshot",
                "ts": ts,
                "date": req_date,
                "time": req_time,
                "quote_date": q.date,
                "quote_time": q.time,
                "symbol": q.symbol,
                "name": q.name,
                "price": q.price,
                "prev_close": q.prev_close,
                "open": q.open_price,
                "high": q.high,
                "low": q.low,
                "volume": q.volume,
                "amount": q.amount,
                "orderbook": _orderbook_to_dict(q),
                "intraday_metrics": intraday_metrics,
                "m5_metrics": m5_metrics,
                "m5_bars_today": m5_bars_today,
                "auction_summary": auction_summary,
                "auction_bars": auction_bars,
                "daily_metrics": daily_metrics,
            }
            append_record(rec)
            out_syms.append(q.symbol)
        return out_syms

    def grouped_symbols(filter_text: str) -> List[str]:
        f = normalize_symbol(filter_text) if filter_text else ""
        syms = sorted({str(r.get("symbol", "")).lower() for r in records if r.get("symbol")})
        if not f:
            return syms
        return [s for s in syms if s == f]

    def records_for_symbol(sym: str) -> List[Dict[str, object]]:
        s = normalize_symbol(sym) if sym else ""
        if not s:
            return []
        return [r for r in records if str(r.get("symbol", "")).lower() == s]

    def latest_snapshot(sym: str) -> Optional[Dict[str, object]]:
        for r in reversed(records_for_symbol(sym)):
            kind = str(r.get("kind") or "snapshot").lower()
            if kind != "auction_sample":
                return r
        return None

    root = tk.Tk()
    root.title("股票助手（桌面端）")
    root.geometry("1100x700")

    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    codes_var = tk.StringVar(value=",".join(codes) if codes else "")
    filter_var = tk.StringVar(value="")
    auto_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="")

    ttk.Label(top, text="股票代码").pack(side="left")
    codes_entry = ttk.Entry(top, textvariable=codes_var, width=40)
    codes_entry.pack(side="left", padx=(6, 10))

    fetch_btn = ttk.Button(top, text="获取/刷新")
    fetch_btn.pack(side="left")

    ttk.Label(top, text="过滤").pack(side="left", padx=(12, 0))
    filter_entry = ttk.Entry(top, textvariable=filter_var, width=16)
    filter_entry.pack(side="left", padx=(6, 10))

    reload_btn = ttk.Button(top, text="重载历史")
    reload_btn.pack(side="left")

    ttk.Checkbutton(top, text="自动刷新", variable=auto_var).pack(side="left", padx=(12, 0))

    main = ttk.Panedwindow(root, orient="horizontal")
    main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    left = ttk.Frame(main, padding=8)
    mid = ttk.Frame(main, padding=8)
    right = ttk.Frame(main, padding=8)
    main.add(left, weight=1)
    main.add(mid, weight=2)
    main.add(right, weight=3)

    ttk.Label(left, text="股票列表").pack(anchor="w")
    sym_list = tk.Listbox(left, height=25)
    sym_list.pack(fill="both", expand=True, pady=(6, 0))

    ttk.Label(mid, text="快照列表").pack(anchor="w")
    snap_list = tk.Listbox(mid, height=25)
    snap_list.pack(fill="both", expand=True, pady=(6, 0))

    btn_row = ttk.Frame(mid)
    btn_row.pack(fill="x", pady=(10, 0))
    copy_latest_btn = ttk.Button(btn_row, text="复制最新")
    copy_latest_btn.pack(side="left")
    copy_history_btn = ttk.Button(btn_row, text="复制历史")
    copy_history_btn.pack(side="left", padx=(10, 0))

    info_label = ttk.Label(right, text="详情")
    info_label.pack(anchor="w")
    text = tk.Text(right, wrap="none")
    text.pack(fill="both", expand=True, pady=(6, 0))

    status = ttk.Label(root, textvariable=status_var, padding=(10, 4))
    status.pack(fill="x")

    def set_status(msg: str) -> None:
        status_var.set(msg)

    def safe_json(obj: object) -> str:
        return json.dumps(_json_safe(obj), ensure_ascii=False, indent=2, sort_keys=True)

    def refresh_symbol_list() -> None:
        sym_list.delete(0, "end")
        for s in grouped_symbols(filter_var.get()):
            sym_list.insert("end", s)

    def refresh_snap_list(sym: str) -> None:
        snap_list.delete(0, "end")
        rs = records_for_symbol(sym)
        snaps = [r for r in rs if str(r.get("kind") or "snapshot").lower() != "auction_sample"]
        for r in snaps[-200:]:
            t = _format_record_dt(r)
            p = r.get("price")
            snap_list.insert("end", f"{t}  price={p}")

    def show_details_for_snapshot(sym: str, snap_index: int) -> None:
        rs = records_for_symbol(sym)
        snaps = [r for r in rs if str(r.get("kind") or "snapshot").lower() != "auction_sample"]
        if not snaps:
            text.delete("1.0", "end")
            return
        idx = max(len(snaps) - 200 + snap_index, 0)
        if idx < 0 or idx >= len(snaps):
            idx = len(snaps) - 1
        r = snaps[idx]
        auc = [x for x in rs if str(x.get("kind") or "").lower() == "auction_sample"]
        head = {
            "symbol": sym,
            "snapshots": len(snaps),
            "auction_samples": len(auc),
        }
        body = safe_json({"meta": head, "snapshot": r})
        text.delete("1.0", "end")
        text.insert("1.0", body)

    def selected_symbol() -> str:
        sel = sym_list.curselection()
        if not sel:
            return ""
        v = sym_list.get(sel[0])
        return str(v)

    def on_symbol_select(event: object) -> None:
        sym = selected_symbol()
        if not sym:
            return
        if not codes_var.get().strip():
            codes_var.set(sym)
        refresh_snap_list(sym)
        show_details_for_snapshot(sym, snap_list.size() - 1)

    def on_snap_select(event: object) -> None:
        sym = selected_symbol()
        if not sym:
            return
        sel = snap_list.curselection()
        if not sel:
            return
        show_details_for_snapshot(sym, int(sel[0]))

    def reload_history() -> None:
        nonlocal records
        with lock:
            records = _load_jsonl_records(history_path)
        refresh_symbol_list()
        set_status(f"已重载历史 records={len(records)}")

    def do_fetch() -> None:
        codes_text = codes_var.get().strip()
        if not codes_text:
            set_status("请输入股票代码")
            return

        def worker() -> None:
            try:
                got = fetch_and_store(codes_text)
                root.after(0, lambda: after_fetch(got))
            except Exception as e:
                root.after(0, lambda: set_status(str(e)))

        def after_fetch(got: List[str]) -> None:
            refresh_symbol_list()
            set_status(f"已获取: {', '.join(got) if got else '无'}  records={len(records)}")

        threading.Thread(target=worker, daemon=True).start()
        set_status("请求中...")

    def copy_text_to_clipboard(payload: str) -> None:
        if not payload:
            return
        root.clipboard_clear()
        root.clipboard_append(payload)
        root.update()
        set_status("已复制到剪贴板")

    def on_copy_latest() -> None:
        sym = selected_symbol()
        if not sym:
            set_status("请选择股票")
            return
        r = latest_snapshot(sym)
        if not r:
            set_status("无可复制数据")
            return
        copy_text_to_clipboard(safe_json(r))

    def on_copy_history() -> None:
        sym = selected_symbol()
        if not sym:
            set_status("请选择股票")
            return
        rs = records_for_symbol(sym)
        if not rs:
            set_status("无可复制数据")
            return
        copy_text_to_clipboard(safe_json(rs))

    def schedule_auto() -> None:
        if auto_var.get():
            do_fetch()
        root.after(int(max(interval_s, 0.5) * 1000), schedule_auto)

    fetch_btn.configure(command=do_fetch)
    reload_btn.configure(command=reload_history)
    copy_latest_btn.configure(command=on_copy_latest)
    copy_history_btn.configure(command=on_copy_history)
    sym_list.bind("<<ListboxSelect>>", on_symbol_select)
    snap_list.bind("<<ListboxSelect>>", on_snap_select)
    filter_entry.bind("<KeyRelease>", lambda e: refresh_symbol_list())

    refresh_symbol_list()
    if codes_var.get().strip():
        do_fetch()
    schedule_auto()

    root.mainloop()
    return 0


def gui_qt(
    codes: Sequence[str],
    interval_s: float,
    timeout_s: float,
    daily: bool,
    daily_len: int,
    bench: str,
    history_file: str,
) -> int:
    try:
        from PySide6 import QtCore, QtWidgets
    except Exception as e:
        raise RuntimeError(
            "当前Python缺少Tk(_tkinter)，且未安装PySide6。请创建虚拟环境后执行：pip install PySide6"
        ) from e

    history_path = os.path.abspath(history_file)
    _ensure_dir(history_path)

    records: List[Dict[str, object]] = _load_jsonl_records(history_path)
    intraday_state: Dict[str, Deque[Tuple[float, float]]] = {}
    last_ts: Dict[str, float] = {}
    bench_candles = fetch_tencent_daily_candles(bench, count=daily_len) if daily and bench else []
    lock = threading.Lock()

    def append_record(rec: Dict[str, object]) -> None:
        with lock:
            records.append(rec)
            _append_jsonl_record(history_path, rec)

    def fetch_and_store(codes_text: str, status_cb: Optional[object] = None) -> List[str]:
        raw_codes = _parse_codes_from_input(codes_text)
        symbols = [normalize_symbol(c) for c in raw_codes]
        symbols = [s for s in symbols if s]
        if not symbols:
            raise ValueError("请输入至少一个有效股票代码")

        if callable(status_cb):
            status_cb(f"拉取行情... symbols={len(symbols)}")
        quotes = fetch_sina_quotes(symbols, timeout_s=timeout_s)
        ts = time.time()
        req_date, req_time = _request_date_time(ts)
        out_syms: List[str] = []
        for i, sym in enumerate(symbols, start=1):
            q = quotes.get(sym)
            if not q:
                continue

            if callable(status_cb):
                status_cb(f"处理 {i}/{len(symbols)}: {sym}")
            
            if callable(status_cb):
                status_cb(f"拉取5分钟K线: {sym}")
            m5_candles = fetch_kline_data(sym, scale="m5", count=200, timeout_s=timeout_s)

            state = intraday_state.setdefault(sym, deque(maxlen=240))
            if not state and m5_candles:
                for c in m5_candles:
                    try:
                        dt = datetime.strptime(c.date, "%Y-%m-%d %H:%M:%S")
                        state.append((dt.timestamp(), c.close_price))
                    except Exception:
                        pass

            state.append((ts, q.price))
            interval = ts - last_ts.get(sym, ts - 1.0)
            if interval <= 0:
                interval = 1.0
            last_ts[sym] = ts

            intraday_metrics = analyze_quote(q, state, interval_s=interval)
            
            # m5_candles fetched above
            m5_metrics = analyze_daily_candles(m5_candles, None) if m5_candles else {}
            m5_bars_today = [_candle_to_dict(c) for c in _filter_m5_from_0930(m5_candles, q.date)]

            if callable(status_cb):
                status_cb(f"拉取1分钟K线: {sym}")
            m1_candles = fetch_kline_data(sym, scale="m1", count=400, timeout_s=timeout_s)
            auction_bars_c = _filter_window(m1_candles, q.date, "09:15", "09:30")
            auction_bars = [_candle_to_dict(c) for c in auction_bars_c]
            auction_summary = _auction_summary(auction_bars_c)
            if _is_auction_time(q.date, q.time):
                append_record(
                    {
                        "kind": "auction_sample",
                        "ts": ts,
                        "date": req_date,
                        "time": req_time,
                        "quote_date": q.date,
                        "quote_time": q.time,
                        "symbol": q.symbol,
                        "name": q.name,
                        "price": q.price,
                        "volume": q.volume,
                        "amount": q.amount,
                        "bid_prices": list(q.bid_prices),
                        "bid_volumes": list(q.bid_volumes),
                        "ask_prices": list(q.ask_prices),
                        "ask_volumes": list(q.ask_volumes),
                    }
                )

            daily_metrics = {}
            if daily:
                if callable(status_cb):
                    status_cb(f"拉取日K: {sym}")
                d_candles = fetch_tencent_daily_candles(sym, count=daily_len)
                daily_metrics = analyze_daily_candles(d_candles, bench_candles)

            rec = {
                "kind": "snapshot",
                "ts": ts,
                "date": req_date,
                "time": req_time,
                "quote_date": q.date,
                "quote_time": q.time,
                "symbol": q.symbol,
                "name": q.name,
                "price": q.price,
                "prev_close": q.prev_close,
                "open": q.open_price,
                "high": q.high,
                "low": q.low,
                "volume": q.volume,
                "amount": q.amount,
                "orderbook": _orderbook_to_dict(q),
                "intraday_metrics": intraday_metrics,
                "m5_metrics": m5_metrics,
                "m5_bars_today": m5_bars_today,
                "auction_summary": auction_summary,
                "auction_bars": auction_bars,
                "daily_metrics": daily_metrics,
            }
            append_record(rec)
            out_syms.append(q.symbol)
        return out_syms

    def symbols_filtered(filter_text: str) -> List[str]:
        f = normalize_symbol(filter_text) if filter_text else ""
        with lock:
            syms = sorted({str(r.get("symbol", "")).lower() for r in records if r.get("symbol")})
        if not f:
            return syms
        return [s for s in syms if s == f]

    def records_for_symbol(sym: str) -> List[Dict[str, object]]:
        s = normalize_symbol(sym) if sym else ""
        if not s:
            return []
        with lock:
            return [r for r in records if str(r.get("symbol", "")).lower() == s]

    def latest_snapshot(sym: str) -> Optional[Dict[str, object]]:
        for r in reversed(records_for_symbol(sym)):
            kind = str(r.get("kind") or "snapshot").lower()
            if kind != "auction_sample":
                return r
        return None

    def safe_json(obj: object) -> str:
        return json.dumps(_json_safe(obj), ensure_ascii=False, indent=2, sort_keys=True)

    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("股票助手（桌面端）")

    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    root = QtWidgets.QVBoxLayout(central)

    top = QtWidgets.QHBoxLayout()
    root.addLayout(top)

    top.addWidget(QtWidgets.QLabel("股票代码"))
    codes_edit = QtWidgets.QLineEdit(",".join(codes) if codes else "")
    codes_edit.setMinimumWidth(360)
    top.addWidget(codes_edit)

    fetch_btn = QtWidgets.QPushButton("获取/刷新")
    top.addWidget(fetch_btn)

    top.addSpacing(12)
    top.addWidget(QtWidgets.QLabel("过滤"))
    filter_edit = QtWidgets.QLineEdit("")
    filter_edit.setMaximumWidth(200)
    top.addWidget(filter_edit)

    reload_btn = QtWidgets.QPushButton("重载历史")
    top.addWidget(reload_btn)

    top.addSpacing(12)
    auto_chk = QtWidgets.QCheckBox("自动刷新")
    top.addWidget(auto_chk)
    top.addStretch(1)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    root.addWidget(splitter, 1)

    sym_list = QtWidgets.QListWidget()
    snap_list = QtWidgets.QListWidget()
    detail = QtWidgets.QPlainTextEdit()
    detail.setReadOnly(True)
    splitter.addWidget(sym_list)
    splitter.addWidget(snap_list)
    splitter.addWidget(detail)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 2)
    splitter.setStretchFactor(2, 4)

    btn_row = QtWidgets.QHBoxLayout()
    root.addLayout(btn_row)
    copy_latest_btn = QtWidgets.QPushButton("复制最新")
    copy_history_btn = QtWidgets.QPushButton("复制历史")
    btn_row.addWidget(copy_latest_btn)
    btn_row.addWidget(copy_history_btn)
    btn_row.addStretch(1)

    status = QtWidgets.QLabel("")
    root.addWidget(status)

    def set_status(msg: str) -> None:
        status.setText(msg)

    def refresh_symbol_list() -> None:
        sym_list.clear()
        for s in symbols_filtered(filter_edit.text()):
            sym_list.addItem(s)

    def current_symbol() -> str:
        item = sym_list.currentItem()
        return item.text() if item else ""

    def refresh_snapshots(sym: str) -> None:
        snap_list.clear()
        rs = records_for_symbol(sym)
        snaps = [r for r in rs if str(r.get("kind") or "snapshot").lower() != "auction_sample"]
        for r in snaps[-200:]:
            t = _format_record_dt(r)
            p = r.get("price")
            snap_list.addItem(f"{t}  price={p}")

    def show_snapshot(sym: str, row: int) -> None:
        rs = records_for_symbol(sym)
        snaps = [r for r in rs if str(r.get("kind") or "snapshot").lower() != "auction_sample"]
        if not snaps:
            detail.setPlainText("")
            return
        idx = max(len(snaps) - 200 + row, 0)
        if idx < 0 or idx >= len(snaps):
            idx = len(snaps) - 1
        r = snaps[idx]
        auc = [x for x in rs if str(x.get("kind") or "").lower() == "auction_sample"]
        payload = {"meta": {"symbol": sym, "snapshots": len(snaps), "auction_samples": len(auc)}, "snapshot": r}
        detail.setPlainText(safe_json(payload))

    def on_symbol_changed() -> None:
        sym = current_symbol()
        if not sym:
            return
        if not codes_edit.text().strip():
            codes_edit.setText(sym)
        refresh_snapshots(sym)
        if snap_list.count() > 0:
            snap_list.setCurrentRow(snap_list.count() - 1)
            show_snapshot(sym, snap_list.currentRow())

    def on_snapshot_changed() -> None:
        sym = current_symbol()
        if not sym:
            return
        show_snapshot(sym, snap_list.currentRow())

    def reload_history() -> None:
        nonlocal records
        with lock:
            records = _load_jsonl_records(history_path)
        refresh_symbol_list()
        set_status(f"已重载历史 records={len(records)}")

    class _Emitter(QtCore.QObject):
        fetched = QtCore.Signal(list)
        error = QtCore.Signal(str)
        status = QtCore.Signal(str)

    emitter = _Emitter()

    busy = {"v": False}
    current_codes = {"v": ""}

    def after_fetch(got: List[str]) -> None:
        busy["v"] = False
        refresh_symbol_list()
        set_status(f"已获取: {', '.join(got) if got else '无'}  records={len(records)}")
        sym = current_symbol()
        if sym:
            refresh_snapshots(sym)

    def after_error(msg: str) -> None:
        busy["v"] = False
        set_status(msg)

    emitter.fetched.connect(after_fetch)
    emitter.error.connect(after_error)
    emitter.status.connect(set_status)

    def do_fetch() -> None:
        if busy["v"]:
            return
        codes_text = codes_edit.text().strip()
        if not codes_text:
            set_status("请输入股票代码")
            return
        busy["v"] = True
        set_status("请求中...")
        current_codes["v"] = codes_text

        def worker() -> None:
            try:
                def status_cb(msg: str) -> None:
                    emitter.status.emit(msg)

                got = fetch_and_store(codes_text, status_cb=status_cb)
                emitter.fetched.emit(got)
            except Exception as e:
                emitter.error.emit(f"{type(e).__name__}: {e}".strip())

        threading.Thread(target=worker, daemon=True).start()

    def copy_latest() -> None:
        sym = current_symbol()
        if not sym:
            set_status("请选择股票")
            return
        r = latest_snapshot(sym)
        if not r:
            set_status("无可复制数据")
            return
        app.clipboard().setText(safe_json(r))
        set_status("已复制到剪贴板")

    def copy_history() -> None:
        sym = current_symbol()
        if not sym:
            set_status("请选择股票")
            return
        rs = records_for_symbol(sym)
        if not rs:
            set_status("无可复制数据")
            return
        app.clipboard().setText(safe_json(rs))
        set_status("已复制到剪贴板")

    timer = QtCore.QTimer()
    timer.setInterval(int(max(interval_s, 0.5) * 1000))
    timer.timeout.connect(lambda: do_fetch() if auto_chk.isChecked() else None)
    timer.start()

    fetch_btn.clicked.connect(do_fetch)
    reload_btn.clicked.connect(reload_history)
    copy_latest_btn.clicked.connect(copy_latest)
    copy_history_btn.clicked.connect(copy_history)
    filter_edit.textChanged.connect(lambda _: refresh_symbol_list())
    sym_list.currentRowChanged.connect(lambda _: on_symbol_changed())
    snap_list.currentRowChanged.connect(lambda _: on_snapshot_changed())

    refresh_symbol_list()
    if codes_edit.text().strip():
        do_fetch()

    win.resize(1100, 700)
    win.show()
    return int(app.exec())


def serve(
    port: int,
    timeout_s: float,
    daily: bool,
    daily_len: int,
    bench: str,
    history_file: str,
) -> int:
    history_path = os.path.abspath(history_file)
    _ensure_dir(history_path)

    lock = threading.Lock()
    records: List[Dict[str, object]] = []
    intraday_state: Dict[str, Deque[Tuple[float, float]]] = {}
    last_ts: Dict[str, float] = {}

    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except Exception:
                        continue
        except Exception:
            pass

    bench_candles = fetch_tencent_daily_candles(bench, count=daily_len) if daily and bench else []

    def append_record(rec: Dict[str, object]) -> None:
        with lock:
            records.append(rec)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(rec), ensure_ascii=False) + "\n")

    def handle_fetch(codes: str) -> Dict[str, object]:
        raw_codes = _parse_codes_from_input(codes)
        symbols = [normalize_symbol(c) for c in raw_codes]
        symbols = [s for s in symbols if s]
        if not symbols:
            return {"error": "请输入至少一个有效股票代码"}

        quotes = fetch_sina_quotes(symbols, timeout_s=timeout_s)
        ts = time.time()
        req_date, req_time = _request_date_time(ts)
        out_syms: List[str] = []
        for sym in symbols:
            q = quotes.get(sym)
            if not q:
                continue

            state = intraday_state.setdefault(sym, deque(maxlen=240))
            state.append((ts, q.price))
            interval = ts - last_ts.get(sym, ts - 1.0)
            if interval <= 0:
                interval = 1.0
            last_ts[sym] = ts

            intraday_metrics = analyze_quote(q, state, interval_s=interval)
            m5_candles = fetch_kline_data(sym, scale="m5", count=200, timeout_s=timeout_s)
            m5_metrics = analyze_daily_candles(m5_candles, None) if m5_candles else {}
            m5_bars_today = [_candle_to_dict(c) for c in _filter_m5_from_0930(m5_candles, q.date)]

            m1_candles = fetch_kline_data(sym, scale="m1", count=400, timeout_s=timeout_s)
            auction_bars_c = _filter_window(m1_candles, q.date, "09:15", "09:30")
            auction_bars = [_candle_to_dict(c) for c in auction_bars_c]
            auction_summary = _auction_summary(auction_bars_c)
            if _is_auction_time(q.date, q.time):
                append_record(
                    {
                        "kind": "auction_sample",
                        "ts": ts,
                        "date": req_date,
                        "time": req_time,
                        "quote_date": q.date,
                        "quote_time": q.time,
                        "symbol": q.symbol,
                        "name": q.name,
                        "price": q.price,
                        "volume": q.volume,
                        "amount": q.amount,
                        "bid_prices": list(q.bid_prices),
                        "bid_volumes": list(q.bid_volumes),
                        "ask_prices": list(q.ask_prices),
                        "ask_volumes": list(q.ask_volumes),
                    }
                )

            daily_metrics = {}
            if daily:
                d_candles = fetch_tencent_daily_candles(sym, count=daily_len)
                daily_metrics = analyze_daily_candles(d_candles, bench_candles)

            rec = {
                "kind": "snapshot",
                "ts": ts,
                "date": req_date,
                "time": req_time,
                "quote_date": q.date,
                "quote_time": q.time,
                "symbol": q.symbol,
                "name": q.name,
                "price": q.price,
                "prev_close": q.prev_close,
                "open": q.open_price,
                "high": q.high,
                "low": q.low,
                "volume": q.volume,
                "amount": q.amount,
                "orderbook": _orderbook_to_dict(q),
                "intraday_metrics": intraday_metrics,
                "m5_metrics": m5_metrics,
                "m5_bars_today": m5_bars_today,
                "auction_summary": auction_summary,
                "auction_bars": auction_bars,
                "daily_metrics": daily_metrics,
            }
            append_record(rec)
            out_syms.append(q.symbol)
        return {"symbols": out_syms}

    def handle_history(symbol: str) -> Dict[str, object]:
        s = normalize_symbol(symbol) if symbol else ""
        with lock:
            if not s:
                return {"records": records[-2000:]}
            filt = [r for r in records if str(r.get("symbol", "")).lower() == s]
            return {"records": filt[-2000:]}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/fetch":
                qs = parse_qs(parsed.query)
                codes = (qs.get("codes") or [""])[0]
                try:
                    obj = handle_fetch(codes)
                    body = json.dumps(_json_safe(obj), ensure_ascii=False).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                except Exception as e:
                    body = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
                    self._send(500, body, "application/json; charset=utf-8")
                return
            if parsed.path == "/api/history":
                qs = parse_qs(parsed.query)
                symbol = (qs.get("symbol") or [""])[0]
                obj = handle_history(symbol)
                body = json.dumps(_json_safe(obj), ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return

            self._send(404, b"Not found", "text/plain; charset=utf-8")

        def log_message(self, format: str, *args: object) -> None:
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Server running at http://127.0.0.1:{port}/")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
