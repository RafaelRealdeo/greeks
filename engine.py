from __future__ import annotations
import os
import datetime as dt
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Clé API : lue depuis l'environnement uniquement ─────────────────────────
# Créez un fichier .env avec : DATABENTO_API_KEY=db-xxxxx
# Ou exportez la variable avant de lancer le script.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import databento as db

API_KEY = os.environ.get("DATABENTO_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "DATABENTO_API_KEY introuvable.\n"
        "  → Créez un fichier .env avec : DATABENTO_API_KEY=db-votre-clé\n"
        "  → Ou : $env:DATABENTO_API_KEY='db-votre-clé' dans PowerShell"
    )

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

NY = ZoneInfo("America/New_York")

DATASET      = "GLBX.MDP3"
SPOT_SYMBOL  = "NQ.c.0"
# Symbols candidats pour NQ options sur CME GLBX.MDP3.
# Au demarrage on teste tous ceux-ci et on garde uniquement ceux qui resolvent.
OPT_ROOTS_CANDIDATES = [
    # Standards
    "NQ.OPT", "QNE.OPT",
    # Friday weeklies (weeks 1-4)
    "E1A.OPT", "E2A.OPT", "E3A.OPT", "E4A.OPT",
    # Monday weeklies
    "E1B.OPT", "E2B.OPT", "E3B.OPT", "E4B.OPT",
    # Tuesday weeklies
    "E1C.OPT", "E2C.OPT", "E3C.OPT", "E4C.OPT",
    # Wednesday weeklies
    "E1D.OPT", "E2D.OPT", "E3D.OPT", "E4D.OPT",
    # QN* alternative (NQ-specific weekly variants)
    "QN1.OPT", "QN2.OPT", "QN3.OPT", "QN4.OPT",
    # Thursday weeklies non disponibles sur ce dataset Databento
]
# La liste finale est decouverte au demarrage et stockee dans OPT_ROOTS
OPT_ROOTS = ["NQ.OPT", "QNE.OPT"]  # fallback minimum si discovery echoue

RISK_FREE     = 0.053
DIV_YIELD     = 0.0
CONTRACT_MULT = 20       # $20 par point NQ

MAX_DTE_DAYS   = 60
MIN_OI         = 0
MONEYNESS_BAND = 0.20

IV_RECALC_SECS = 2.0    # recalcule IV max toutes les 2s par contrat

# Bootstrap : combien de jours d'historique 1m à charger au démarrage
BOOTSTRAP_DAYS = 4

# Bootstrap IV : fenêtre de recherche du dernier quote connu par contrat
# (couvre le trou du week-end : vendredi soir -> lundi pre-market)
BOOTSTRAP_QUOTE_HOURS = 48

# ── Facteur dealer ───────────────────────────────────────────────────────────
DEALER_OI_FACTOR = 1.0

# ── Gamma flip scan — plage dynamique ────────────────────────────────────────
SCAN_PCT_DEFAULT = 0.08
SCAN_PCT_MIN     = 0.06
SCAN_PCT_MAX     = 0.25
SCAN_N           = 201

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
SQRT_2PI     = math.sqrt(2.0 * math.pi)


# ─────────────────────────────────────────────
# MATH
# ─────────────────────────────────────────────

def _norm_pdf(x):
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

UNDEF_PRICE = 2**63 - 1   # Databento sentinel pour "pas de prix"

def px_to_float(x):
    if x is None: return float("nan")
    if isinstance(x, (int, np.integer)):
        if x == UNDEF_PRICE or x == 0: return float("nan")
        return float(x) / 1e9
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")

def yearfrac(exp_date: dt.date) -> float:
    exp_dt = dt.datetime.combine(exp_date, dt.time(16, 0), tzinfo=NY)
    secs   = max((exp_dt.astimezone(dt.timezone.utc) - dt.datetime.now(dt.timezone.utc)).total_seconds(), 0.0)
    return secs / (365.0 * 24.0 * 3600.0)

def _d1d2(S, K, r, q, T, sigma):
    vs = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vs
    return d1, d1 - vs, vs

def bs_price(S, K, r, q, T, sigma, is_call):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0: return float("nan")
    d1, d2, _ = _d1d2(S, K, r, q, T, sigma)
    dfq, dfr  = math.exp(-q * T), math.exp(-r * T)
    if is_call: return dfq * S * _norm_cdf(d1) - dfr * K * _norm_cdf(d2)
    return dfr * K * _norm_cdf(-d2) - dfq * S * _norm_cdf(-d1)

def bs_all_greeks(S, K, r, q, T, sigma, is_call):
    nan = float("nan")
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        return dict(delta=nan, gamma=nan, vega=nan, theta=nan, charm=nan, vanna=nan, vomma=nan)
    d1, d2, vs = _d1d2(S, K, r, q, T, sigma)
    sqrtT = math.sqrt(T)
    dfq = math.exp(-q * T); dfr = math.exp(-r * T)
    pdf1 = _norm_pdf(d1); cdf1 = _norm_cdf(d1); cdf2 = _norm_cdf(d2)
    if is_call:
        delta     = dfq * cdf1
        theta_raw = -dfq*S*pdf1*sigma/(2*sqrtT) - r*dfr*K*cdf2 + q*dfq*S*cdf1
        charm     = -dfq*(pdf1*(2*(r-q)*T - d2*vs)/(2*T*vs) - q*cdf1)/365.0
    else:
        delta     = dfq * (cdf1 - 1.0)
        theta_raw = -dfq*S*pdf1*sigma/(2*sqrtT) + r*dfr*K*_norm_cdf(-d2) - q*dfq*S*_norm_cdf(-d1)
        charm     = -dfq*(pdf1*(2*(r-q)*T - d2*vs)/(2*T*vs) + q*_norm_cdf(-d1))/365.0
    gamma = dfq * pdf1 / (S * vs)
    vega  = S * dfq * pdf1 * sqrtT / 100.0
    theta = theta_raw / 365.0
    vanna = -dfq * pdf1 * d2 / sigma / 100.0
    vomma = vega * d1 * d2 / sigma / 100.0
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, charm=charm, vanna=vanna, vomma=vomma)

def implied_vol(price_mkt, S, K, r, q, T, is_call, sigma0=0.20):
    if not (price_mkt > 0 and S > 0 and K > 0 and T > 0): return float("nan")
    lo, hi = 1e-4, 5.0
    sigma  = float(np.clip(sigma0, lo, hi))
    for _ in range(10):
        px = bs_price(S, K, r, q, T, sigma, is_call)
        if not math.isfinite(px): break
        diff = px - price_mkt
        if abs(diff) < 1e-7: return sigma
        d1, _, _ = _d1d2(S, K, r, q, T, sigma)
        vega = math.exp(-q*T) * S * _norm_pdf(d1) * math.sqrt(T)
        if vega < 1e-9: break
        sigma -= diff / vega
        if not (lo < sigma < hi): break
    f_lo = bs_price(S, K, r, q, T, lo, is_call) - price_mkt
    f_hi = bs_price(S, K, r, q, T, hi, is_call) - price_mkt
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0: return float("nan")
    for _ in range(50):
        mid   = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, r, q, T, mid, is_call) - price_mkt
        if abs(f_mid) < 1e-8: return mid
        if f_lo * f_mid <= 0: hi, f_hi = mid, f_mid
        else:                 lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class OptionDef:
    raw_symbol : str
    strike     : float
    expiration : dt.date
    is_call    : bool
    oi         : int

@dataclass
class OptionState:
    def __init__(self, dealer_pos=0.0):
        self.bid=float("nan"); self.ask=float("nan"); self.mid=float("nan"); self.iv=float("nan")
        self.delta=float("nan"); self.gamma=float("nan"); self.vega=float("nan"); self.theta=float("nan")
        self.charm=float("nan"); self.vanna=float("nan"); self.vomma=float("nan")
        self.flow_pos=0.0; self.dealer_pos=dealer_pos
        self.last_iv_t=0.0


# ─────────────────────────────────────────────
# CANDLE AGGREGATOR (multi-timeframe OHLCV)
# ─────────────────────────────────────────────

class CandleAggregator:
    """
    Construit des barres OHLCV multi-TF depuis le flux spot tick.

    - bootstrap_from_1m() : précharge depuis l'historique 1m
    - add_tick()          : alimente en live depuis le flux tick
    - get_all()           : retourne les bars pour chaque TF
    """

    TIMEFRAMES = {"1m": 60, "5m": 300, "15m": 900, "1H": 3600}
    MAX_BARS   = 200  # ~3h en 1m / 16h en 5m / 2j en 15m / 8j en 1H

    def __init__(self):
        self._lock   = threading.Lock()
        self.bars    = {tf: [] for tf in self.TIMEFRAMES}
        self.current = {tf: None for tf in self.TIMEFRAMES}

    def bootstrap_from_1m(self, bars_1m):
        """
        Précharge à partir d'une liste de bars 1m {t, o, h, l, c, v}.
        Reconstruit toutes les TF en cascade.
        La dernière bar de chaque TF reste 'current' pour pouvoir être étendue
        en live par add_tick() sans créer de doublon.
        """
        with self._lock:
            self.bars    = {tf: [] for tf in self.TIMEFRAMES}
            self.current = {tf: None for tf in self.TIMEFRAMES}

            for b in bars_1m:
                ts = b["t"]
                for tf, dur in self.TIMEFRAMES.items():
                    bar_start = (ts // dur) * dur
                    cur = self.current[tf]
                    if cur is None:
                        self.current[tf] = {
                            "t": bar_start,
                            "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"],
                            "v": b["v"],
                        }
                    elif cur["t"] == bar_start:
                        if b["h"] > cur["h"]: cur["h"] = b["h"]
                        if b["l"] < cur["l"]: cur["l"] = b["l"]
                        cur["c"] = b["c"]
                        cur["v"] += b["v"]
                    else:
                        # Fermer la bar courante, ouvrir la suivante
                        self.bars[tf].append(cur)
                        if len(self.bars[tf]) > self.MAX_BARS:
                            self.bars[tf].pop(0)
                        self.current[tf] = {
                            "t": bar_start,
                            "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"],
                            "v": b["v"],
                        }

    def add_tick(self, price: float, ts_ns: int):
        """Appelé sur chaque update du spot mid (live)."""
        if not (math.isfinite(price) and price > 0):
            return
        ts_sec = ts_ns / 1e9
        with self._lock:
            for tf, dur in self.TIMEFRAMES.items():
                bar_start = int(ts_sec // dur) * dur
                cur = self.current[tf]
                if cur is None or cur["t"] != bar_start:
                    if cur is not None:
                        self.bars[tf].append(cur)
                        if len(self.bars[tf]) > self.MAX_BARS:
                            self.bars[tf].pop(0)
                    self.current[tf] = {
                        "t": bar_start, "o": price, "h": price,
                        "l": price, "c": price, "v": 1,
                    }
                else:
                    if price > cur["h"]: cur["h"] = price
                    if price < cur["l"]: cur["l"] = price
                    cur["c"] = price
                    cur["v"] += 1

    def get_all(self) -> dict:
        """Retourne toutes les barres (complétées + barre en cours) pour chaque TF."""
        with self._lock:
            out = {}
            for tf in self.TIMEFRAMES:
                bars = list(self.bars[tf])
                if self.current[tf] is not None:
                    bars.append(dict(self.current[tf]))
                out[tf] = bars
            return out





# ─────────────────────────────────────────────
# TRADE STORE (SQLite persistance)
# ─────────────────────────────────────────────

class TradeStore:
    """
    Persistance SQLite des trades pour replay et stats jour.
    WAL pour lectures concurrentes sans bloquer les inserts live.
    Batche les inserts pour limiter le coût I/O sur le flux trade.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS trades (
            ts_ns       INTEGER NOT NULL,
            raw         TEXT    NOT NULL,
            strike      REAL    NOT NULL,
            dte         INTEGER NOT NULL,
            is_call     INTEGER NOT NULL,
            side        TEXT    NOT NULL,
            size        INTEGER NOT NULL,
            price       REAL    NOT NULL,
            premium     REAL    NOT NULL,
            iv          REAL,
            delta       REAL,
            flags       TEXT,
            session_vol INTEGER,
            pct_of_oi   REAL
        );
        CREATE INDEX IF NOT EXISTS idx_trades_ts      ON trades(ts_ns);
        CREATE INDEX IF NOT EXISTS idx_trades_strike  ON trades(strike);
        CREATE INDEX IF NOT EXISTS idx_trades_flags   ON trades(flags);
        CREATE INDEX IF NOT EXISTS idx_trades_premium ON trades(premium);
    """

    COLS = ["ts_ns","raw","strike","dte","is_call","side","size","price",
            "premium","iv","delta","flags","session_vol","pct_of_oi"]

    BATCH_SIZE      = 50
    FLUSH_INTERVAL  = 5.0  # secondes max avant flush forcé même si batch pas plein

    def __init__(self, path: str = "./data/trades.db"):
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        # isolation_level=None -> autocommit; check_same_thread=False car le batch
        # est inséré depuis le thread option et lu depuis Flask.
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(self.SCHEMA)
        self._batch: list[tuple] = []
        self._last_flush = time.time()

    def insert(self, trade: dict) -> None:
        row = (
            int(trade["ts_ns"]), str(trade["raw"]),
            float(trade["strike"]), int(trade["dte"]),
            1 if trade["is_call"] else 0,
            str(trade["side"]), int(trade["size"]),
            float(trade["price"]), float(trade["premium"]),
            float(trade["iv"])    if trade.get("iv")    is not None else None,
            float(trade["delta"]) if trade.get("delta") is not None else None,
            ",".join(trade.get("flags", []) or []),
            int(trade.get("session_vol", 0) or 0),
            float(trade.get("pct_of_oi", 0.0) or 0.0),
        )
        with self._lock:
            self._batch.append(row)
            elapsed = time.time() - self._last_flush
            if len(self._batch) >= self.BATCH_SIZE or elapsed >= self.FLUSH_INTERVAL:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._batch:
            self._last_flush = time.time()
            return
        try:
            self._conn.executemany(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self._batch,
            )
        except sqlite3.Error as e:
            print(f"[trade-store] insert error: {e}")
        self._batch.clear()
        self._last_flush = time.time()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def query(self, since_ns=None, until_ns=None, min_premium=None,
              min_size=None, flag=None, is_call=None, limit=500) -> list[dict]:
        clauses, params = [], []
        if since_ns is not None:
            clauses.append("ts_ns >= ?"); params.append(int(since_ns))
        if until_ns is not None:
            clauses.append("ts_ns <= ?"); params.append(int(until_ns))
        if min_premium is not None:
            clauses.append("premium >= ?"); params.append(float(min_premium))
        if min_size is not None:
            clauses.append("size >= ?"); params.append(int(min_size))
        if flag:
            clauses.append("flags LIKE ?"); params.append(f"%{flag}%")
        if is_call is not None:
            clauses.append("is_call = ?"); params.append(1 if is_call else 0)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT {','.join(self.COLS)} FROM trades{where} "
            f"ORDER BY ts_ns DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._lock:
            self._flush_locked()
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(zip(self.COLS, r))
            d["is_call"] = bool(d["is_call"])
            d["flags"]   = d["flags"].split(",") if d["flags"] else []
            out.append(d)
        return out

    def stats(self, since_ns=None, until_ns=None) -> dict:
        clauses, params = [], []
        if since_ns is not None:
            clauses.append("ts_ns >= ?"); params.append(int(since_ns))
        if until_ns is not None:
            clauses.append("ts_ns <= ?"); params.append(int(until_ns))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT
              COUNT(*),
              COALESCE(SUM(size), 0),
              COALESCE(SUM(premium), 0.0),
              COALESCE(SUM(CASE WHEN side='BUY'  THEN premium ELSE 0 END), 0.0),
              COALESCE(SUM(CASE WHEN side='SELL' THEN premium ELSE 0 END), 0.0),
              COALESCE(SUM(CASE WHEN is_call=1   THEN premium ELSE 0 END), 0.0),
              COALESCE(SUM(CASE WHEN is_call=0   THEN premium ELSE 0 END), 0.0),
              COALESCE(SUM(CASE WHEN flags LIKE '%BLOCK%' THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN flags LIKE '%BIG%'   THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN flags LIKE '%FAST%'  THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN flags LIKE '%\\%OI%' ESCAPE '\\' THEN 1 ELSE 0 END), 0)
            FROM trades{where}
        """
        with self._lock:
            self._flush_locked()
            row = self._conn.execute(sql, params).fetchone()
        keys = ["count","total_size","total_premium","buy_premium","sell_premium",
                "call_premium","put_premium","blocks","bigs","fasts","pct_oi"]
        return dict(zip(keys, row or [0]*len(keys)))

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            try: self._conn.close()
            except Exception: pass


# ─────────────────────────────────────────────
# TRADE TAPE (option flow live)
# ─────────────────────────────────────────────

class TradeTape:
    """
    Buffer rolling des trades options recents avec classification.
    Detecte les SWEEP/BLOCK/BIG/FAST events pour UOA alerts.
    """
    MAX_TRADES = 250
    BLOCK_SIZE = 50           # >= 50 contrats = BLOCK
    BIG_PREMIUM = 50_000      # >= 50k$ premium = BIG
    FAST_WINDOW_NS = 1_000_000_000  # 1s : trades rapproches = FAST
    FAST_MIN_COUNT = 3        # >= 3 trades en 1s = FAST/sweep-like

    def __init__(self, store: "TradeStore | None" = None):
        self._lock = threading.Lock()
        self.trades = []
        # Pour la detection de "FAST" : derniers ts par strike
        self.last_ts_per_strike = {}  # strike -> [ts_ns, ts_ns, ...]
        # Pour le %OI : volume cumule par contrat
        self.volume_per_contract = {}  # raw_symbol -> total_size
        # Alertes UOA recentes (a consommer cote dashboard)
        self.alerts = []
        self.MAX_ALERTS = 30
        # Persistance SQLite optionnelle (mise hors lock).
        self.store = store

    def add_trade(self, raw, opt, side, size, price, ts_ns, st):
        """side: 'B'=buy aggressed (bid-hit), 'A'=sell aggressed (ask-hit)."""
        # Note: 'A' (ask-hit) signifie l'aggresseur a achete au ask -> BUY
        #       'B' (bid-hit) signifie l'aggresseur a vendu au bid -> SELL
        # On normalise en BUY/SELL du point de vue client/aggresseur
        client_side = "BUY" if side == "A" else ("SELL" if side == "B" else "")
        if not client_side or size <= 0 or not math.isfinite(price) or price <= 0:
            return None

        premium = float(size) * float(price) * float(CONTRACT_MULT)
        flags = []
        if size >= self.BLOCK_SIZE:
            flags.append("BLOCK")
        if premium >= self.BIG_PREMIUM:
            flags.append("BIG")

        trade = None
        with self._lock:
            # FAST detection : combien de trades en <1s sur ce strike ?
            now = ts_ns
            tslist = self.last_ts_per_strike.setdefault(opt.strike, [])
            tslist.append(now)
            cutoff = now - self.FAST_WINDOW_NS
            tslist[:] = [t for t in tslist if t >= cutoff]
            if len(tslist) >= self.FAST_MIN_COUNT:
                flags.append("FAST")

            # Volume cumule du contrat (pour le %OI ratio)
            self.volume_per_contract[raw] = self.volume_per_contract.get(raw, 0) + size
            session_vol = self.volume_per_contract[raw]
            pct_of_oi = (session_vol / opt.oi * 100.0) if opt.oi > 0 else 0.0
            if pct_of_oi > 10.0 and size >= 5:
                flags.append("%OI")

            trade = {
                "ts_ns"      : int(ts_ns),
                "raw"        : raw,
                "strike"     : float(opt.strike),
                "dte"        : int((opt.expiration - dt.date.today()).days),
                "is_call"    : bool(opt.is_call),
                "side"       : client_side,
                "size"       : int(size),
                "price"      : float(price),
                "premium"    : float(premium),
                "iv"         : float(st.iv) if (st and math.isfinite(st.iv)) else None,
                "delta"      : float(st.delta) if (st and math.isfinite(st.delta)) else None,
                "flags"      : flags,
                "session_vol": int(session_vol),
                "pct_of_oi"  : float(pct_of_oi),
            }

            self.trades.append(trade)
            if len(self.trades) > self.MAX_TRADES:
                self.trades.pop(0)

            # Alerte UOA si flag majeur
            if flags:
                self.alerts.append(trade)
                if len(self.alerts) > self.MAX_ALERTS:
                    self.alerts.pop(0)

        # Persistance hors lock pour ne jamais bloquer le flux live si I/O lent.
        if trade is not None and self.store is not None:
            try:
                self.store.insert(trade)
            except Exception as e:
                print(f"[trade-tape] persist skipped: {e}")

        return trade

    def get_recent(self, n=120):
        with self._lock:
            return list(self.trades[-n:][::-1])  # plus recents en tete

    def get_alerts(self, n=20):
        with self._lock:
            return list(self.alerts[-n:][::-1])


# ─────────────────────────────────────────────
# BULL / BEAR DELTA FLOW
# ─────────────────────────────────────────────

class BullBearFlow:
    """
    Aggregation du delta-flow signe par session.
    - BUY call  -> +delta * size * spot * mult  (bullish)
    - SELL call -> -delta * size * spot * mult  (bearish)
    - BUY put   -> +delta * size * spot * mult  (delta negatif -> bearish)
    - SELL put  -> -delta * size * spot * mult  (delta negatif -> bullish)
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.bullish = 0.0
        self.bearish = 0.0
        self.last_ts_ns = 0

    def add(self, client_side, delta, size, spot, ts_ns):
        if not math.isfinite(delta) or size <= 0 or not math.isfinite(spot) or spot <= 0:
            return
        sign = 1.0 if client_side == "BUY" else -1.0
        flow_dollar = sign * delta * size * spot * CONTRACT_MULT
        with self._lock:
            if flow_dollar > 0:
                self.bullish += flow_dollar
            else:
                self.bearish += -flow_dollar  # stocke en valeur positive
            self.last_ts_ns = max(self.last_ts_ns, ts_ns)

    def get(self) -> dict:
        with self._lock:
            net = self.bullish - self.bearish
            return {
                "bullish": float(self.bullish),
                "bearish": float(self.bearish),
                "net"    : float(net),
                "ts_ns"  : int(self.last_ts_ns),
            }


# ─────────────────────────────────────────────
# MARKET READER (interprète le snapshot en verdict actionnable)
# ─────────────────────────────────────────────

class MarketReader:
    """
    Convertit un snapshot brut en verdict trader décisionnel.
    Objectif : qu'un débutant qui ne connaît pas Black-Scholes puisse
    décider long/short/no-trade en lisant uniquement la sortie analyze().
    """

    NEAR_FLIP_PCT     = 0.003     # < 0.3% du flip = "près du flip"
    WALL_NEAR_PTS     = 100       # wall < 100 pts = "proche"
    WALL_FAR_PTS      = 200       # wall > 200 pts = "loin"
    PIN_RANGE_PTS     = 80        # < 80 pts entre 2 walls = pin zone
    TRAP_GAP_PTS      = 100       # gap > 100 pts sans wall = piège
    VIEW_RANGE_PCT    = 0.05      # on regarde ±5% autour du spot
    CLUSTER_NEAR_PCT  = 0.02      # cluster γ < 2% du spot = pivot pertinent
    FLOW_WINDOW_NS    = 5 * 60 * 1_000_000_000   # UOA récentes = 5 min
    FLOW_CONFIRM_PREM = 200_000   # $200K min pour confirmer/contredire
    FLIP_MIG_SIGNIF   = 50        # delta flip > 50 pts = migration signifiante
    TREND_SIGNIF_PCT  = 0.30      # trend > 0.3% en 30m = renforce le bias
    ATR_HIGH_PTS      = 30        # ATR(14, 5m) > 30 pts = vol élevée
    DEX_BIG_DOLLAR    = 5e9       # |Net DEX| > $5B = bias dealer notable

    def analyze(self, snap: dict) -> dict:
        if not snap or snap.get("spot") is None:
            return self._empty()

        spot = float(snap["spot"])
        flip = self._safe_float(snap.get("gamma_flip"))
        charm_flip = self._safe_float(snap.get("charm_flip"))
        vanna_flip = self._safe_float(snap.get("vanna_flip"))

        net_gex   = float(snap.get("net_gex")   or 0.0)
        gross_gex = float(snap.get("gross_gex") or 0.0)
        net_dex   = float(snap.get("net_dex")   or 0.0)
        flow_buy  = int(snap.get("flow_buy")  or 0)
        flow_sell = int(snap.get("flow_sell") or 0)
        bb        = snap.get("bb_flow") or {}

        kl = snap.get("key_levels") or {}
        call_walls = list(kl.get("call_walls") or [])
        put_walls  = list(kl.get("put_walls")  or [])
        max_abs_strikes = list(kl.get("max_abs_strikes") or [])

        trades_alerts = snap.get("trades_alerts") or []
        greeks_hist   = snap.get("greeks_history") or []
        candles_d     = snap.get("candles") or {}

        # Dedupe par strike + score d'intensité
        all_walls = self._dedupe_walls(call_walls + put_walls)
        max_oi      = max((float(w.get("oi") or 0)        for w in all_walls), default=0.0)
        max_abs_gex = max((abs(float(w.get("gex") or 0))  for w in all_walls), default=0.0)
        for w in all_walls:
            tier, score = self._strength(w, max_oi, max_abs_gex)
            w["strength"]       = tier
            w["strength_score"] = score
            w["side"] = "CALL" if (w.get("gex") or 0) > 0 else "PUT"

        # Proche résistance / support (premiers walls de chaque côté)
        above = [w for w in all_walls if w["strike"] > spot]
        below = [w for w in all_walls if w["strike"] < spot]
        next_res = min(above, key=lambda w: w["strike"]) if above else None
        next_sup = max(below, key=lambda w: w["strike"]) if below else None

        regime, regime_label, regime_explanation = self._regime(spot, flip, net_gex, bb)
        setup       = self._setup(spot, next_res, next_sup, regime)
        bias, conf  = self._bias(regime, next_res, next_sup, setup, bb)
        traps       = self._traps(spot, all_walls)
        phase, advice = self._phase(snap.get("ts_ns") or 0)

        # ── Nouvelles analyses exploitant + de données du snap ─────────────
        flip_mig    = self._flip_migration(greeks_hist, flip)
        trend_atr   = self._trend_and_atr(candles_d)
        flow_conf   = self._flow_confirmation(trades_alerts, spot, bias, snap.get("ts_ns") or 0)
        cluster     = self._cluster_hint(max_abs_strikes, spot)
        dealer_hint = self._dealer_flow_hint(net_dex, flow_buy, flow_sell)

        # ── Pondération multi-source de la confidence ─────────────────────
        # 1. Setup vs régime
        if (bias == "LONG"  and regime == "BEARISH") or \
           (bias == "SHORT" and regime == "BULLISH"):
            conf *= 0.55
        if regime == "TRANSITION":
            conf *= 0.75
        if phase == "RTH_OPEN":
            conf *= 0.7
        # 2. Trend récent renforce/contredit
        if trend_atr:
            tp = trend_atr["trend_pct"]
            if tp >= self.TREND_SIGNIF_PCT  and bias == "LONG":  conf *= 1.15
            if tp <= -self.TREND_SIGNIF_PCT and bias == "SHORT": conf *= 1.15
            if tp >= self.TREND_SIGNIF_PCT  and bias == "SHORT": conf *= 0.80
            if tp <= -self.TREND_SIGNIF_PCT and bias == "LONG":  conf *= 0.80
        # 3. Option flow confirme/contredit
        if flow_conf:
            conf *= 1.10 if flow_conf["direction"] == "CONFIRM" else 0.70
        # 4. Migration du flip alignée avec le bias
        if flip_mig and flip_mig["direction"] in ("up", "down"):
            if bias == "LONG"  and flip_mig["direction"] == "up":   conf *= 1.08
            if bias == "SHORT" and flip_mig["direction"] == "down": conf *= 1.08
        conf = max(0.10, min(0.95, conf))

        # ── Actions enrichies ─────────────────────────────────────────────
        actions = self._actions(setup, next_res, next_sup, regime, phase)
        # Flow confirmation tout en haut
        if flow_conf:
            ic = "🎯" if flow_conf["direction"] == "CONFIRM" else "⚠️"
            txt = ("Flow confirme : " if flow_conf["direction"] == "CONFIRM" else "Flow contredit : ") \
                  + f"{self._fmt_dollar(flow_conf['premium'])} BUY {flow_conf['label']}"
            det = "UOA récente cohérente avec le bias." if flow_conf["direction"] == "CONFIRM" \
                  else "Méfie-toi, l'option flow va dans l'autre sens."
            actions.insert(0, {"icon": ic, "label": txt, "detail": det})
        # Cluster γ
        if cluster:
            actions.append({
                "icon": "🧲",
                "label": f"Cluster γ à {cluster['strike']:.0f} ({cluster['distance']:+.0f} pts)",
                "detail": "Pivot magnétique — prix souvent piégé proche.",
            })
        # Vol élevée
        if trend_atr and trend_atr["atr"] >= self.ATR_HIGH_PTS:
            actions.append({
                "icon": "📉",
                "label": f"Vol élevée (ATR {trend_atr['atr']:.0f} pts)",
                "detail": "Réduire size, élargir stops.",
            })
        # 0DTE warnings si phase = RTH_CLOSE et des niveaux EPHEMERAL
        if phase == "RTH_CLOSE":
            ephem = [w for w in (next_res, next_sup)
                     if w and (w.get("dte") is not None and w["dte"] <= 1)]
            if ephem:
                actions.append({
                    "icon": "⏰",
                    "label": "0DTE expire à 16h NY",
                    "detail": "Les niveaux EPHEMERAL disparaîtront après la cloche — ne tiens pas overnight sur eux.",
                })
        # Charm/Vanna flips en mentions additionnelles
        if charm_flip is not None and abs(charm_flip - spot) < spot * 0.02:
            actions.append({
                "icon": "🧷",
                "label": f"Charm flip à {charm_flip:.0f}",
                "detail": "Niveau d'attraction des expirations (probable pin à la cloche).",
            })
        if vanna_flip is not None and abs(vanna_flip - spot) < spot * 0.02:
            actions.append({
                "icon": "🌡️",
                "label": f"Vanna flip à {vanna_flip:.0f}",
                "detail": "Sensible à la vol. Si VIX bouge, positionnement change ici.",
            })

        # ── Augmenter regime_explanation avec migration + dealer + trend ──
        parts = [regime_explanation]
        if flip_mig:
            if flip_mig["direction"] == "up":
                parts.append(f"Flip migre ↑ {flip_mig['delta']:+.0f} pts / {flip_mig['minutes']}min "
                             "= positionnement bullish qui se construit.")
            elif flip_mig["direction"] == "down":
                parts.append(f"Flip migre ↓ {flip_mig['delta']:+.0f} pts / {flip_mig['minutes']}min "
                             "= positionnement bearish qui se construit.")
            else:
                parts.append(f"Flip stable depuis {flip_mig['minutes']}min = régime établi.")
        if trend_atr and abs(trend_atr["trend_pct"]) >= self.TREND_SIGNIF_PCT:
            arrow = "↑" if trend_atr["trend_pct"] > 0 else "↓"
            parts.append(f"Trend 5m {arrow} {trend_atr['trend_pct']:+.2f}% (ATR {trend_atr['atr']:.0f} pts).")
        parts.extend(dealer_hint)
        regime_explanation = " ".join(parts)

        return {
            "regime"             : regime,
            "regime_label"       : regime_label,
            "regime_explanation" : regime_explanation,
            "bias"               : bias,
            "bias_confidence"    : conf,
            "next_resistance"    : self._level_summary(next_res, spot, "above"),
            "next_support"       : self._level_summary(next_sup, spot, "below"),
            "key_setup"          : setup,
            "actions"            : actions,
            "traps"              : traps,
            "phase"              : phase,
            "phase_advice"       : advice,
            "walls_scored"       : all_walls,
            "charm_flip"         : charm_flip,
            "vanna_flip"         : vanna_flip,
            "flip_migration"     : flip_mig,
            "trend_atr"          : trend_atr,
            "flow_confirmation"  : flow_conf,
            "cluster_hint"       : cluster,
            "net_dex"            : net_dex,
            "session_flow"       : {"buy": flow_buy, "sell": flow_sell,
                                     "ratio": (flow_buy / max(1, flow_buy + flow_sell))},
        }

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe_walls(walls):
        out = {}
        for w in walls:
            try: s = round(float(w["strike"]), 2)
            except Exception: continue
            cur = out.get(s)
            if cur is None or abs(float(w.get("gex") or 0)) > abs(float(cur.get("gex") or 0)):
                out[s] = dict(w)
        return sorted(out.values(), key=lambda w: w["strike"])

    @staticmethod
    def _strength(w, max_oi, max_abs_gex):
        oi   = float(w.get("oi") or 0)
        gx   = abs(float(w.get("gex") or 0))
        n_oi = (oi / max_oi)         if max_oi > 0      else 0.0
        n_gx = (gx / max_abs_gex)    if max_abs_gex > 0 else 0.0
        score = 0.5 * n_oi + 0.5 * n_gx
        if score >= 0.6: return "STRONG", score
        if score >= 0.3: return "MEDIUM", score
        return "WEAK", score

    @staticmethod
    def _safe_float(v):
        if v is None: return None
        try: f = float(v)
        except (TypeError, ValueError): return None
        return f if math.isfinite(f) else None

    @staticmethod
    def _lifetime(dte):
        if dte is None: return "UNKNOWN"
        if dte <= 1:    return "EPHEMERAL"
        if dte <= 7:    return "WEEKLY"
        return "STRUCTURAL"

    @staticmethod
    def _lifetime_label(lifetime, dte):
        if lifetime == "EPHEMERAL":
            return f"0DTE — fiable jusqu'à 16h NY uniquement (DTE={dte})"
        if lifetime == "WEEKLY":
            return f"Hebdo — valide cette semaine (DTE={dte}j)"
        if lifetime == "STRUCTURAL":
            return f"Structurel — niveau long-terme (DTE={dte}j)"
        return ""

    def _flip_migration(self, greeks_history, current_flip):
        """Compare le flip actuel au flip 30min plus tôt."""
        if current_flip is None or not greeks_history: return None
        hist = list(greeks_history)
        if len(hist) < 5: return None  # besoin d'un mini-historique
        # Les bars sont 1m, donc 30 min = 30 bars max
        window = hist[-30:] if len(hist) >= 30 else hist
        past = [b.get("gamma_flip") for b in window if b.get("gamma_flip") is not None]
        if not past: return None
        old = past[0]
        try: old = float(old)
        except Exception: return None
        if not math.isfinite(old): return None
        delta = current_flip - old
        direction = "up" if delta > self.FLIP_MIG_SIGNIF else \
                    ("down" if delta < -self.FLIP_MIG_SIGNIF else "stable")
        return {
            "delta"    : float(delta),
            "minutes"  : len(window),
            "direction": direction,
            "old_flip" : float(old),
            "new_flip" : float(current_flip),
        }

    def _trend_and_atr(self, candles_dict):
        """Trend récent + ATR(14) sur 5m bars."""
        if not isinstance(candles_dict, dict): return None
        bars_5m = candles_dict.get("5m") or []
        if len(bars_5m) < 14: return None
        bars = bars_5m[-30:] if len(bars_5m) >= 30 else bars_5m
        try:
            c0 = float(bars[0]["c"]); cN = float(bars[-1]["c"])
        except (KeyError, TypeError, ValueError):
            return None
        if c0 <= 0: return None
        trend_pct = (cN / c0 - 1.0) * 100.0
        # True Range
        trs = []
        for i in range(1, len(bars)):
            try:
                h = float(bars[i]["h"]); l = float(bars[i]["l"])
                pc = float(bars[i-1]["c"])
            except (KeyError, TypeError, ValueError):
                continue
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        if not trs: return None
        atr_window = trs[-14:] if len(trs) >= 14 else trs
        atr = sum(atr_window) / len(atr_window)
        return {
            "trend_pct": float(trend_pct),
            "atr"      : float(atr),
            "bars_used": int(len(bars)),
        }

    def _flow_confirmation(self, trades_alerts, spot, bias, now_ns):
        """Détecte si les UOA récentes confirment ou contredisent le bias."""
        if not trades_alerts or not bias or bias == "NEUTRAL":
            return None
        if not now_ns: now_ns = time.time_ns()
        cutoff = now_ns - self.FLOW_WINDOW_NS
        recent = [t for t in trades_alerts
                  if int(t.get("ts_ns", 0)) >= cutoff
                  and t.get("side") == "BUY"]
        if not recent: return None
        # Ne compte que les UOA proches du spot (< 5% de moneyness)
        recent = [t for t in recent if abs(float(t.get("strike", 0)) - spot) <= spot * 0.05]
        if not recent: return None
        call_buy = sum(float(t.get("premium", 0)) for t in recent if t.get("is_call"))
        put_buy  = sum(float(t.get("premium", 0)) for t in recent if not t.get("is_call"))
        # Échantillon de raw symbols pour le UI
        sample = [
            {"strike": t.get("strike"), "size": t.get("size"),
             "premium": t.get("premium"), "is_call": t.get("is_call"),
             "flags": t.get("flags", [])}
            for t in recent[:5]
        ]
        # Direction call vs put
        if call_buy >= self.FLOW_CONFIRM_PREM and call_buy > put_buy * 1.5:
            direction = "CONFIRM" if bias == "LONG" else "CONTRADICT"
            return {"direction": direction, "premium": call_buy, "label": "calls", "samples": sample}
        if put_buy >= self.FLOW_CONFIRM_PREM and put_buy > call_buy * 1.5:
            direction = "CONFIRM" if bias == "SHORT" else "CONTRADICT"
            return {"direction": direction, "premium": put_buy, "label": "puts", "samples": sample}
        return None

    def _cluster_hint(self, max_abs_strikes, spot):
        """Top cluster γ s'il est proche du spot."""
        if not max_abs_strikes: return None
        top = max_abs_strikes[0]
        strike = float(top.get("strike", 0))
        dist = strike - spot
        if abs(dist) > spot * self.CLUSTER_NEAR_PCT: return None
        return {
            "strike"  : strike,
            "distance": float(dist),
            "gex"     : float(top.get("gex", 0)),
            "oi"      : int(top.get("oi", 0) or 0),
            "dte"     : top.get("dte"),
            "strength": top.get("strength", "MEDIUM"),
        }

    def _dealer_flow_hint(self, net_dex, flow_buy, flow_sell):
        """Phrases additionnelles pour regime_explanation."""
        parts = []
        if abs(net_dex) >= self.DEX_BIG_DOLLAR:
            if net_dex > 0:
                parts.append(f"Net DEX {self._fmt_dollar(net_dex)} (dealers nets longs delta — vendront les rallies).")
            else:
                parts.append(f"Net DEX {self._fmt_dollar(net_dex)} (dealers shorts delta — achèteront les dips).")
        total = (flow_buy or 0) + (flow_sell or 0)
        if total > 500:  # éviter le bruit sur très peu de trades
            ratio = (flow_buy or 0) / total
            if ratio >= 0.62:
                parts.append(f"Buying pressure session ({flow_buy:,}B vs {flow_sell:,}S).")
            elif ratio <= 0.38:
                parts.append(f"Selling pressure session ({flow_sell:,}S vs {flow_buy:,}B).")
        return parts

    def _regime(self, spot, flip, net_gex, bb):
        if flip is None:
            return ("TRANSITION", "⚠️ Régime indéterminé",
                    "Pas assez de données options pour identifier le flip. Évite les trades directionnels.")

        dist     = spot - flip
        dist_pct = dist / spot if spot > 0 else 0.0

        if abs(dist_pct) < self.NEAR_FLIP_PCT:
            return ("TRANSITION", "⚠️ Près du flip — prudence",
                    f"Spot ${spot:,.0f} à seulement {dist:+.0f} pts du flip ${flip:,.0f}. "
                    "Régime instable : les dealers basculent de stabilisateurs à amplificateurs. "
                    "Attends que le spot s'éloigne du flip avant de prendre position.")

        if dist > 0 and net_gex > 0:
            extra = self._flow_hint(bb)
            return ("BULLISH", "🟢 Zone bullish stable",
                    f"Spot ${spot:,.0f} au-dessus du flip ${flip:,.0f} ({dist:+.0f} pts). "
                    f"Net GEX {self._fmt_dollar(net_gex)} positif : les dealers stabilisent le marché. "
                    f"Dips achetés, range probable.{extra}")

        if dist < 0 and net_gex < 0:
            extra = self._flow_hint(bb)
            return ("BEARISH", "🔴 Zone bearish volatile",
                    f"Spot ${spot:,.0f} sous le flip ${flip:,.0f} ({dist:+.0f} pts). "
                    f"Net GEX {self._fmt_dollar(net_gex)} négatif : les dealers amplifient les mouvements. "
                    f"Cassures/squeezes probables.{extra}")

        return ("TRANSITION", "⚠️ Signaux mixtes",
                f"Spot à {dist:+.0f} pts du flip mais Net GEX {self._fmt_dollar(net_gex)} contredit. "
                "Attends confirmation avant de trader.")

    def _setup(self, spot, next_res, next_sup, regime):
        if next_res is None or next_sup is None:
            return {
                "type": "AVOID",
                "title": "⏸️ Pas de niveau clair",
                "description": "Murs trop loin ou inexistants. Attends que le spot s'approche d'un niveau identifié.",
                "entry": None, "stop": None, "target": None, "rr": None,
            }

        d_res    = next_res["strike"] - spot
        d_sup    = spot - next_sup["strike"]
        res_str  = next_res["strength"]
        sup_str  = next_sup["strength"]

        # PIN : entre 2 walls solides très proches
        if (d_res + d_sup) <= self.PIN_RANGE_PTS and res_str != "WEAK" and sup_str != "WEAK":
            lo, hi = next_sup["strike"], next_res["strike"]
            mid    = (lo + hi) / 2
            return {
                "type"       : "PIN",
                "title"      : f"📍 PIN PLAY @ {lo:.0f}–{hi:.0f}",
                "description": (f"Spot coincé entre support {lo:.0f} ({sup_str}) et résistance {hi:.0f} ({res_str}). "
                                f"Range étroit ({(hi-lo):.0f} pts). Scalp range OK, swing à éviter."),
                "entry" : round(mid),
                "stop"  : round(lo - 10),
                "target": round(hi - 5),
                "rr"    : round(max(0.0, (hi - 5 - mid) / max(1.0, mid - (lo - 10))), 2),
            }

        # BOUNCE LONG : support proche+fort, résistance loin
        if d_sup <= self.WALL_NEAR_PTS and sup_str == "STRONG" and d_res >= self.WALL_FAR_PTS / 2:
            entry  = round(next_sup["strike"] + 10)
            stop   = round(next_sup["strike"] - 20)
            target = round(next_res["strike"] - 5)
            risk   = max(1, entry - stop)
            reward = max(0, target - entry)
            return {
                "type"       : "BOUNCE",
                "title"      : f"📈 BOUNCE LONG @ {next_sup['strike']:.0f}",
                "description": (f"Support {sup_str} {next_sup['strike']:.0f} à {d_sup:.0f} pts dessous. "
                                f"Résistance loin ({d_res:.0f} pts). Setup bounce favorable."),
                "entry": entry, "stop": stop, "target": target, "rr": round(reward / risk, 2),
            }

        # REJECTION SHORT : résistance proche+forte, support loin
        if d_res <= self.WALL_NEAR_PTS and res_str == "STRONG" and d_sup >= self.WALL_FAR_PTS / 2:
            entry  = round(next_res["strike"] - 10)
            stop   = round(next_res["strike"] + 20)
            target = round(next_sup["strike"] + 5)
            risk   = max(1, stop - entry)
            reward = max(0, entry - target)
            return {
                "type"       : "REJECTION",
                "title"      : f"📉 REJECTION SHORT @ {next_res['strike']:.0f}",
                "description": (f"Résistance {res_str} {next_res['strike']:.0f} à {d_res:.0f} pts dessus. "
                                f"Support loin ({d_sup:.0f} pts). Setup rejet favorable."),
                "entry": entry, "stop": stop, "target": target, "rr": round(reward / risk, 2),
            }

        # BREAKOUT : wall proche mais faible + flip très loin (momentum potentiel)
        if d_res <= self.WALL_NEAR_PTS / 2 and res_str == "WEAK":
            return {
                "type"       : "BREAKOUT",
                "title"      : f"⚡ BREAKOUT possible @ {next_res['strike']:.0f}",
                "description": (f"Résistance faible {next_res['strike']:.0f} à {d_res:.0f} pts. "
                                "Si cassée, accélération vers le prochain wall plausible."),
                "entry" : round(next_res["strike"] + 5),
                "stop"  : round(next_res["strike"] - 15),
                "target": None,
                "rr"    : None,
            }

        return {
            "type"       : "WAIT",
            "title"      : "⏸️ Pas de setup clair",
            "description": (f"Spot entre support {next_sup['strike']:.0f} ({sup_str}) et résistance "
                            f"{next_res['strike']:.0f} ({res_str}). Attends qu'il s'approche d'un mur fort."),
            "entry": None, "stop": None, "target": None, "rr": None,
        }

    def _bias(self, regime, next_res, next_sup, setup, bb):
        s = setup.get("type")
        if s == "PIN":       return "NEUTRAL", 0.55
        if s == "BOUNCE":    return "LONG",    0.78
        if s == "REJECTION": return "SHORT",   0.75
        if s == "BREAKOUT":  return "LONG",    0.55  # spéculatif
        # WAIT/AVOID : se fier au régime de fond + flow
        flow_net = float((bb or {}).get("net") or 0.0)
        if regime == "BULLISH":
            return ("LONG", 0.65) if flow_net >= 0 else ("LONG", 0.55)
        if regime == "BEARISH":
            return ("SHORT", 0.65) if flow_net <= 0 else ("SHORT", 0.55)
        return "NEUTRAL", 0.40

    def _actions(self, setup, next_res, next_sup, regime, phase):
        out = []
        s = setup.get("type")
        if phase == "RTH_OPEN":
            out.append({"icon":"⏳","label":"Attends 10h30 NY","detail":"Volatilité d'ouverture trop élevée, walls peu fiables."})
            return out
        if s == "BOUNCE" and next_sup:
            out.append({"icon":"✅","label":f"Achète les dips vers {next_sup['strike']:.0f}",
                        "detail":f"Support {next_sup['strength']} confirmé."})
            if next_res:
                out.append({"icon":"🎯","label":f"Cible {next_res['strike']:.0f}",
                            "detail":f"Première résistance ({next_res['strength']})."})
            out.append({"icon":"🛑","label":f"Stop sous {setup.get('stop')}", "detail":"Sortie si cassure du support."})
        elif s == "REJECTION" and next_res:
            out.append({"icon":"✅","label":f"Vends les rallies vers {next_res['strike']:.0f}",
                        "detail":f"Résistance {next_res['strength']} confirmée."})
            if next_sup:
                out.append({"icon":"🎯","label":f"Cible {next_sup['strike']:.0f}",
                            "detail":f"Premier support ({next_sup['strength']})."})
            out.append({"icon":"🛑","label":f"Stop au-dessus de {setup.get('stop')}", "detail":"Sortie si breakout."})
        elif s == "PIN":
            out.append({"icon":"🔄","label":"Scalp range uniquement","detail":"Pas de directionnel — fade les extrêmes."})
            out.append({"icon":"⛔","label":"Évite le swing","detail":"Range trop étroit pour tenir une position."})
        elif s == "BREAKOUT":
            out.append({"icon":"⚡","label":f"Attends la cassure confirmée de {setup.get('entry')}",
                        "detail":"Une bougie 5m au-dessus = signal."})
            out.append({"icon":"⚠️","label":"Réduire size","detail":"Setup spéculatif, faux breakouts fréquents."})
        else:  # WAIT/AVOID
            out.append({"icon":"⏸️","label":"Pas de trade","detail":"Aucun setup à risque-rendement favorable maintenant."})
            if regime == "BULLISH":
                out.append({"icon":"👀","label":"Surveille les dips","detail":"Régime bullish — patience d'un retrace."})
            elif regime == "BEARISH":
                out.append({"icon":"👀","label":"Surveille les rebonds","detail":"Régime bearish — patience d'un rally."})
        return out

    def _traps(self, spot, walls):
        view_lo = spot * (1 - self.VIEW_RANGE_PCT)
        view_hi = spot * (1 + self.VIEW_RANGE_PCT)
        in_view = sorted(
            [w for w in walls if view_lo <= w["strike"] <= view_hi],
            key=lambda w: w["strike"],
        )
        if len(in_view) < 2:
            return []
        traps = []
        for a, b in zip(in_view[:-1], in_view[1:]):
            gap = b["strike"] - a["strike"]
            if gap > self.TRAP_GAP_PTS:
                traps.append({
                    "zone"  : [round(float(a["strike"])), round(float(b["strike"]))],
                    "gap"   : round(float(gap)),
                    "reason": (f"Aucun wall significatif entre {a['strike']:.0f} et {b['strike']:.0f} "
                               f"({gap:.0f} pts). Mouvement libre = momentum/squeeze possible."),
                })
        return traps

    def _phase(self, ts_ns):
        if not ts_ns or ts_ns <= 0:
            ts_ns = time.time_ns()
        ts_utc = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.timezone.utc)
        ny     = ts_utc.astimezone(NY)
        h      = ny.hour + ny.minute / 60.0
        if 4.0  <= h <  9.5:  return ("PREMARKET", "🌅 Pre-market — peu d'activité, walls basés sur OI d'hier. Données indicatives seulement.")
        if 9.5  <= h < 10.5:  return ("RTH_OPEN",  "🔔 RTH OPEN — volatilité maximale. Attends 10h30 NY avant de trader.")
        if 10.5 <= h < 14.0:  return ("RTH_MID",   "📊 RTH MID — conditions normales, walls fiables. Idéal pour trader le plan.")
        if 14.0 <= h < 16.0:  return ("RTH_CLOSE", "🔔 RTH CLOSE — 0DTE expirent à 16h NY, pin risk élevé. Méfie-toi des accélérations.")
        if 16.0 <= h < 18.0:  return ("POSTCLOSE", "🌆 Post-close — volumes options chutent, walls statiques.")
        return ("OVERNIGHT", "🌙 Overnight — walls statiques, pas de fresh positioning. Mouvements souvent contre-tendance à l'open.")

    def _level_summary(self, w, spot, direction):
        if w is None: return None
        dte = w.get("dte")
        lifetime = self._lifetime(dte)
        return {
            "strike"        : float(w["strike"]),
            "distance"      : float(w["strike"] - spot),
            "strength"      : w["strength"],
            "strength_score": w["strength_score"],
            "oi"            : int(w.get("oi") or 0),
            "gex"           : float(w.get("gex") or 0),
            "side"          : w.get("side", "?"),
            "dte"           : int(dte) if isinstance(dte, (int, float)) else None,
            "lifetime"      : lifetime,
            "lifetime_label": self._lifetime_label(lifetime, dte),
            "reason"        : self._level_reason(w, direction),
        }

    def _level_reason(self, w, direction):
        s  = w["strength"]
        oi = int(w.get("oi") or 0)
        dte = w.get("dte")
        lifetime = self._lifetime(dte)
        if s == "STRONG":
            base = f"Wall majeur ({oi:,} contrats + GEX élevé). Forte probabilité de réaction."
        elif s == "MEDIUM":
            base = f"Wall correct ({oi:,} contrats). Réaction possible, cassure non exclue."
        else:
            base = f"Wall faible ({oi:,} contrats). Probable test/break."
        if lifetime == "EPHEMERAL":
            base += " ⚠ 0DTE — fiable AUJOURD'HUI seulement, disparaît à 16h NY."
        elif lifetime == "WEEKLY":
            base += " Hebdo, valide cette semaine."
        elif lifetime == "STRUCTURAL":
            base += " Structurel (long terme)."
        return base

    @staticmethod
    def _flow_hint(bb):
        net = float((bb or {}).get("net") or 0.0)
        if abs(net) < 1e6: return ""
        if net > 0: return f" Delta-flow session bullish ({MarketReader._fmt_dollar(net)})."
        return f" Delta-flow session bearish ({MarketReader._fmt_dollar(net)})."

    @staticmethod
    def _fmt_dollar(v):
        try: v = float(v)
        except Exception: return "—"
        a = abs(v); sign = "-" if v < 0 else "+"
        if a >= 1e9: return f"{sign}${a/1e9:.1f}B"
        if a >= 1e6: return f"{sign}${a/1e6:.0f}M"
        if a >= 1e3: return f"{sign}${a/1e3:.0f}K"
        return f"{sign}${a:.0f}"

    def _empty(self):
        return {
            "regime": "UNKNOWN", "regime_label": "— En attente de données",
            "regime_explanation": "Pas encore de snapshot complet.",
            "bias": "NEUTRAL", "bias_confidence": 0.0,
            "next_resistance": None, "next_support": None,
            "key_setup": None, "actions": [], "traps": [],
            "phase": "UNKNOWN", "phase_advice": "—",
            "walls_scored": [], "charm_flip": None,
        }


# ─────────────────────────────────────────────
# GREEKS TIME-SERIES (1m buckets pour le chart intraday)
# ─────────────────────────────────────────────

class GreeksTimeSeries:
    """Stocke les aggregates Greeks par minute pour le chart intraday."""

    def __init__(self, max_minutes: int = 240):
        self._lock = threading.Lock()
        self.bars = []         # liste de bars finalisees
        self.current = None    # bar en cours d'accumulation
        self.max_minutes = max_minutes

    def update(self, ts_ns: int, net_gex: float, gamma_flip,
               net_dex: float, net_vanex: float):
        ts_sec = ts_ns / 1e9
        bar_t = int(ts_sec // 60) * 60
        with self._lock:
            cur = self.current
            if cur is None or cur["t"] != bar_t:
                # Finaliser la barre precedente
                if cur is not None and cur["n"] > 0:
                    finalized = {
                        "t": cur["t"],
                        "net_gex"   : cur["net_gex"]   / cur["n"],
                        "gamma_flip": (cur["flip_sum"] / cur["n_flip"]) if cur["n_flip"] else None,
                        "net_dex"   : cur["net_dex"]   / cur["n"],
                        "net_vanex" : cur["net_vanex"] / cur["n"],
                    }
                    self.bars.append(finalized)
                    if len(self.bars) > self.max_minutes:
                        self.bars.pop(0)
                # Nouvelle barre
                self.current = {
                    "t": bar_t, "n": 0,
                    "net_gex": 0.0, "flip_sum": 0.0, "n_flip": 0,
                    "net_dex": 0.0, "net_vanex": 0.0,
                }
                cur = self.current
            cur["n"] += 1
            cur["net_gex"]   += net_gex
            cur["net_dex"]   += net_dex
            cur["net_vanex"] += net_vanex
            if gamma_flip is not None and math.isfinite(gamma_flip):
                cur["flip_sum"] += float(gamma_flip)
                cur["n_flip"]   += 1

    def get_all(self) -> list:
        with self._lock:
            out = list(self.bars)
            if self.current is not None and self.current["n"] > 0:
                cur = self.current
                out.append({
                    "t": cur["t"],
                    "net_gex"   : cur["net_gex"]   / cur["n"],
                    "gamma_flip": (cur["flip_sum"] / cur["n_flip"]) if cur["n_flip"] else None,
                    "net_dex"   : cur["net_dex"]   / cur["n"],
                    "net_vanex" : cur["net_vanex"] / cur["n"],
                })
            return out


# ─────────────────────────────────────────────
# CHAIN LOADER
# ─────────────────────────────────────────────

def _safe_end(hist) -> dt.datetime:
    info = hist.metadata.get_dataset_range(dataset=DATASET)
    raw  = info["end"] if isinstance(info, dict) else info.end
    if isinstance(raw, str): raw = dt.datetime.fromisoformat(raw)
    if raw.tzinfo is None:   raw = raw.replace(tzinfo=dt.timezone.utc)
    end = raw - dt.timedelta(seconds=30)
    # Le marche est ferme le week-end (CME Globex) : pas de data du samedi
    # ni du dimanche avant reouverture. On recule jusqu'au dernier jour ouvre.
    while end.weekday() >= 5:  # Saturday=5, Sunday=6
        end -= dt.timedelta(days=1)
    return end

def discover_roots() -> list[str]:
    """
    Teste chaque candidat de OPT_ROOTS_CANDIDATES.
    Retourne la liste de ceux qui resolvent (definitions non vides).
    """
    print("[engine] Discovery des roots options NQ...")
    hist = db.Historical(key=API_KEY)
    end = _safe_end(hist)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    working = []
    for root in OPT_ROOTS_CANDIDATES:
        try:
            tmp = hist.timeseries.get_range(
                dataset=DATASET, schema="definition",
                symbols=root, stype_in="parent",
                start=start, end=end,
            ).to_df()
            if not tmp.empty:
                working.append(root)
                print(f"[engine]   ✓ {root}: {len(tmp)} defs")
            else:
                print(f"[engine]   - {root}: vide")
        except Exception as e:
            err = str(e).split(chr(10))[0][:80]
            print(f"[engine]   x {root}: {err}")
    if not working:
        print("[engine] !!! Aucun root resolu, fallback minimum")
        working = ["NQ.OPT", "QNE.OPT"]
    print(f"[engine] {len(working)} roots actifs: {working}")
    return working


def load_chain() -> dict[str, OptionDef]:
    hist  = db.Historical(key=API_KEY)
    end   = _safe_end(hist)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[engine] Chargement chain NQ (defs jusqu'a {end.strftime('%H:%M UTC')})…")

    dfs = []
    for root in OPT_ROOTS:
        tmp = None
        # Retry plus agressif pour NQ.OPT (la chain principale, indispensable)
        max_retries = 8 if root == "NQ.OPT" else 2
        for attempt in range(max_retries):
            try:
                tmp = hist.timeseries.get_range(
                    dataset=DATASET, schema="definition",
                    symbols=root, stype_in="parent",
                    start=start, end=end,
                ).to_df()
                break  # succes
            except Exception as e:
                err_str = str(e)
                is_transient = ("504" in err_str or "timeout" in err_str.lower()
                                or "gateway" in err_str.lower() or "503" in err_str
                                or "502" in err_str or "connection" in err_str.lower())
                if attempt < max_retries - 1 and is_transient:
                    backoff = min(3 + attempt * 2, 15)  # 3s, 5s, 7s, ... cap 15s
                    print(f"[engine]   {root} erreur transitoire (tentative {attempt+1}/{max_retries}) - retry dans {backoff}s")
                    time.sleep(backoff)
                    continue
                print(f"[engine]   {root} ignore: {e}")
                tmp = None
                break
        if tmp is not None and not tmp.empty:
            print(f"[engine]   {root}: {len(tmp)} contrats")
            dfs.append(tmp)

    if not dfs:
        raise RuntimeError("Aucune definition NQ trouvee.")

    defs = pd.concat(dfs, ignore_index=True).drop_duplicates()
    sym_col = "raw_symbol" if "raw_symbol" in defs.columns else "symbol"
    defs["expiration"] = pd.to_datetime(defs["expiration"], utc=True).dt.date
    defs["is_call"]    = defs["instrument_class"].astype(str).str.upper().str.startswith("C")
    defs["strike_f"]   = defs["strike_price"].apply(px_to_float) if "strike_price" in defs.columns else float("nan")

    # OI
    oi_map = {}
    try:
        stat_dfs = []
        for root in OPT_ROOTS:
            try:
                s = hist.timeseries.get_range(
                    dataset=DATASET, schema="statistics",
                    symbols=root, stype_in="parent",
                    start=start, end=end,
                ).to_df()
                if not s.empty: stat_dfs.append(s)
            except Exception: pass
        if stat_dfs:
            stats = pd.concat(stat_dfs, ignore_index=True)
            sc = "raw_symbol" if "raw_symbol" in stats.columns else "symbol"
            try: stats = stats[stats["stat_type"] == db.StatType.OPEN_INTEREST].copy()
            except Exception: pass
            if not stats.empty:
                stats  = stats.drop_duplicates(sc, keep="last")
                oi_map = dict(zip(stats[sc].astype(str), stats["quantity"].astype(int)))
    except Exception as e:
        print(f"[engine] OI indisponible: {e}")

    out = {}
    for _, r in defs.iterrows():
        rs = str(r[sym_col])
        strike = float(r["strike_f"])
        if not math.isfinite(strike) or strike <= 0: continue
        out[rs] = OptionDef(
            raw_symbol=rs, strike=strike,
            expiration=r["expiration"], is_call=bool(r["is_call"]),
            oi=int(oi_map.get(rs, 0)),
        )
    print(f"[engine] {len(out):,} contrats charges (dealer factor={DEALER_OI_FACTOR:.0%}).")
    return out


def load_historical_1m_bars(days: int = BOOTSTRAP_DAYS) -> list[dict]:
    """
    Charge l'historique des bars 1m pour NQ.c.0 depuis Databento.
    Retourne une liste de dicts {t, o, h, l, c, v} triés par t ascendant.
    """
    try:
        hist  = db.Historical(key=API_KEY)
        end   = _safe_end(hist)
        start = end - dt.timedelta(days=days)
        print(f"[engine] Bootstrap candles 1m : {days}j d'historique NQ.c.0...")

        df = hist.timeseries.get_range(
            dataset=DATASET,
            schema="ohlcv-1m",
            symbols=[SPOT_SYMBOL],
            stype_in="continuous",
            start=start, end=end,
        ).to_df()

        if df.empty:
            print("[engine] Bootstrap : aucune barre 1m retournee.")
            return []

        bars = []
        for ts, row in df.iterrows():
            try:
                ts_sec = int(pd.Timestamp(ts).timestamp())
                o = px_to_float(row.get("open"))
                h = px_to_float(row.get("high"))
                l = px_to_float(row.get("low"))
                c = px_to_float(row.get("close"))
                v = int(row.get("volume", 1) or 1)
                if not all(math.isfinite(x) and x > 0 for x in (o, h, l, c)):
                    continue
                bars.append({
                    "t": (ts_sec // 60) * 60,
                    "o": o, "h": h, "l": l, "c": c, "v": v,
                })
            except Exception:
                continue

        bars.sort(key=lambda b: b["t"])
        # Dédoublonner par timestamp
        seen = set(); uniq = []
        for b in bars:
            if b["t"] in seen: continue
            seen.add(b["t"]); uniq.append(b)
        print(f"[engine] Bootstrap : {len(uniq)} bars 1m chargees.")
        return uniq

    except Exception as e:
        print(f"[engine] Bootstrap candles echec: {e}")
        return []


# ─────────────────────────────────────────────
# LIVE ENGINE
# ─────────────────────────────────────────────

class LiveGreeksEngine:
    """
    Streaming live via db.Live() sur GLBX.MDP3.
    """

    def __init__(self):
        # Discovery dynamique des roots options qui resolvent sur Databento
        global OPT_ROOTS
        OPT_ROOTS = discover_roots()
        self.chain = load_chain()
        self.state = {
            raw: OptionState(dealer_pos=-float(opt.oi) * DEALER_OI_FACTOR)
            for raw, opt in self.chain.items()
        }

        self.iid_to_raw: dict[int, str] = {}

        self._lock           = threading.Lock()
        self.spot_mid        = float("nan")
        self.spot_ts_ns      = 0
        self.flow_buy        = 0
        self.flow_sell       = 0
        self.flow_last_ts_ns = 0

        # Agrégateur multi-TF pour le chart daytrading
        self.candles = CandleAggregator()
        # Time-series intraday des Greeks (1m buckets)
        self.greeks_ts = GreeksTimeSeries()
        # Persistance SQLite (peut être désactivée via NQGS_DISABLE_PERSIST=1)
        self.store = None
        if os.environ.get("NQGS_DISABLE_PERSIST", "0") != "1":
            try:
                self.store = TradeStore(os.environ.get("NQGS_TRADES_DB", "./data/trades.db"))
                print(f"[engine] TradeStore actif : {self.store.path}")
            except Exception as e:
                print(f"[engine] TradeStore indisponible ({e}) — persistance désactivée.")
        # Trade tape live + flow bullish/bearish
        self.tape = TradeTape(store=self.store)
        self.flow = BullBearFlow()
        # Interprète le snapshot en verdict trader
        self.reader = MarketReader()
        # Compteurs diagnostic
        self._trade_count = 0
        self._trade_count_captured = 0
        self._last_trade_log = time.time()
        self._rtype_counts = {}      # dict rtype_str -> count
        self._last_rtype_log = time.time()
        # Bootstrap depuis l'historique 1m
        bars_1m = load_historical_1m_bars()
        if bars_1m:
            self.candles.bootstrap_from_1m(bars_1m)
            # Resumé des candles par TF après bootstrap
            for tf, bars in self.candles.get_all().items():
                print(f"[engine]   {tf:>3s}: {len(bars)} candles disponibles")
            # Initialiser le spot mid avec le dernier close pour que
            # snapshot() commence à fonctionner même avant le premier tick live
            last = bars_1m[-1]
            self.spot_mid   = float(last["c"])
            self.spot_ts_ns = int(last["t"]) * 1_000_000_000
            # Bootstrap IV/greeks depuis le dernier quote connu de chaque
            # contrat actif, pour ne pas attendre les premiers ticks live
            # (sinon: dashboard vide en pre-market / faible liquidite)
            self._bootstrap_iv()

    # ── FILTERS ──────────────────────────────────────────────────────────────

    def _is_active(self, opt, spot):
        T = yearfrac(opt.expiration)
        if T <= 0 or T * 365 > MAX_DTE_DAYS: return False
        if abs(opt.strike / spot - 1.0) > MONEYNESS_BAND: return False
        return True

    def _bootstrap_iv(self):
        """
        Pre-charge IV/greeks depuis le dernier quote (mbp-1) connu de chaque
        contrat actif autour du spot courant, pour que le snapshot ne soit
        pas vide en attendant les premiers ticks live.
        """
        spot = self.spot_mid
        if not (math.isfinite(spot) and spot > 0):
            return
        active = {raw: opt for raw, opt in self.chain.items() if self._is_active(opt, spot)}
        if not active:
            return
        print(f"[engine] Bootstrap IV : recherche du dernier quote pour {len(active)} contrats actifs...")
        try:
            hist  = db.Historical(key=API_KEY)
            end   = _safe_end(hist)
            start = end - dt.timedelta(hours=BOOTSTRAP_QUOTE_HOURS)
            df = hist.timeseries.get_range(
                dataset=DATASET, schema="mbp-1",
                symbols=list(active.keys()), stype_in="raw_symbol",
                start=start, end=end,
            ).to_df()
        except Exception as e:
            print(f"[engine] Bootstrap IV indisponible: {e}")
            return
        if df.empty:
            print("[engine] Bootstrap IV : aucun quote recent trouve.")
            return

        sym_col = "raw_symbol" if "raw_symbol" in df.columns else "symbol"
        df = df.sort_index().drop_duplicates(sym_col, keep="last")

        seeded = 0
        for _, row in df.iterrows():
            opt = active.get(str(row[sym_col]))
            if opt is None: continue
            bid = px_to_float(row.get("bid_px_00"))
            ask = px_to_float(row.get("ask_px_00"))
            if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0):
                continue
            mid = 0.5 * (bid + ask)
            T = yearfrac(opt.expiration)
            if T <= 0: continue
            iv = implied_vol(mid, spot, opt.strike, RISK_FREE, DIV_YIELD, T, opt.is_call, 0.20)
            if not math.isfinite(iv): continue
            g = bs_all_greeks(spot, opt.strike, RISK_FREE, DIV_YIELD, T, iv, opt.is_call)

            raw = opt.raw_symbol
            st = self.state.setdefault(raw, OptionState(dealer_pos=-float(opt.oi) * DEALER_OI_FACTOR))
            st.bid, st.ask, st.mid = bid, ask, mid
            st.iv=iv; st.delta=g["delta"]; st.gamma=g["gamma"]
            st.vega=g["vega"]; st.theta=g["theta"]
            st.charm=g["charm"]; st.vanna=g["vanna"]; st.vomma=g["vomma"]
            st.last_iv_t = time.time()
            seeded += 1
        print(f"[engine] Bootstrap IV : {seeded}/{len(active)} contrats seedes depuis le dernier quote.")

    # ── SPOT HANDLER ─────────────────────────────────────────────────────────

    def on_spot(self, rec):
        rtype = getattr(rec, "rtype", None)
        rt_str = str(rtype)

        if "MBP_1" in rt_str or rtype == getattr(db.RType, "MBP_1", None):
            lvls = getattr(rec, "levels", None)
            if lvls:
                bid = px_to_float(lvls[0].bid_px)
                ask = px_to_float(lvls[0].ask_px)
            else:
                bid = px_to_float(getattr(rec, "bid_px_00", None))
                ask = px_to_float(getattr(rec, "ask_px_00", None))

            if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0:
                mid    = 0.5 * (bid + ask)
                ts_ns  = int(getattr(rec, "ts_recv", time.time_ns()))
                with self._lock:
                    self.spot_mid   = mid
                    self.spot_ts_ns = ts_ns
                self.candles.add_tick(mid, ts_ns)

    # ── OPTION HANDLERS ───────────────────────────────────────────────────────

    def on_option(self, rec):
        rtype  = getattr(rec, "rtype", None)
        rt_str = str(rtype)

        # Diagnostic : compter les record types vus
        cls_name = type(rec).__name__
        key = cls_name + "|" + rt_str
        self._rtype_counts[key] = self._rtype_counts.get(key, 0) + 1
        # Log toutes les 30s
        now_s = time.time()
        if now_s - self._last_rtype_log >= 30.0:
            top = sorted(self._rtype_counts.items(), key=lambda x: -x[1])[:8]
            print("[opt-stats] records vus (top types):")
            for k, n in top:
                print(f"           {n:>7}  {k}")
            self._last_rtype_log = now_s

        if rtype == getattr(db.RType, "SYMBOL_MAPPING", None) or "SYMBOL_MAPPING" in rt_str:
            iid = int(getattr(rec, "instrument_id", 0))
            sym = str(getattr(rec, "stype_out_symbol", ""))
            if iid and sym:
                self.iid_to_raw[iid] = sym
            return

        # Detection robuste : par classe ou par rtype
        is_trade = (cls_name.lower().startswith("trade") or
                    "TRADE" in rt_str.upper() or
                    rtype == getattr(db.RType, "TRADE", None))
        is_mbp1 = ("MBP" in cls_name.upper() or "MBP_1" in rt_str or
                   rtype == getattr(db.RType, "MBP_1", None))

        if is_trade:
            self._on_trade(rec)
        elif is_mbp1:
            self._on_quote(rec)

    def _resolve(self, rec):
        iid = int(getattr(rec, "instrument_id", 0))
        raw = self.iid_to_raw.get(iid)
        if not raw:
            raw = str(getattr(rec, "raw_symbol", "") or "")
        return raw, self.chain.get(raw)

    def _on_quote(self, rec):
        raw, opt = self._resolve(rec)
        if opt is None: return

        with self._lock:
            spot = float(self.spot_mid)
        if not (math.isfinite(spot) and spot > 0): return
        if not self._is_active(opt, spot): return

        lvls = getattr(rec, "levels", None)
        if lvls:
            bid = px_to_float(lvls[0].bid_px)
            ask = px_to_float(lvls[0].ask_px)
        else:
            bid = px_to_float(getattr(rec, "bid_px_00", None))
            ask = px_to_float(getattr(rec, "ask_px_00", None))

        if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0): return

        mid = 0.5 * (bid + ask)
        T   = yearfrac(opt.expiration)
        if T <= 0: return

        st = self.state.setdefault(raw, OptionState(dealer_pos=-float(opt.oi) * DEALER_OI_FACTOR))
        st.bid, st.ask, st.mid = bid, ask, mid

        now = time.time()
        if (now - st.last_iv_t) >= IV_RECALC_SECS:
            sigma0 = st.iv if math.isfinite(st.iv) else 0.20
            iv = implied_vol(mid, spot, opt.strike, RISK_FREE, DIV_YIELD, T, opt.is_call, sigma0)
            if math.isfinite(iv):
                g = bs_all_greeks(spot, opt.strike, RISK_FREE, DIV_YIELD, T, iv, opt.is_call)
                st.iv=iv; st.delta=g["delta"]; st.gamma=g["gamma"]
                st.vega=g["vega"]; st.theta=g["theta"]
                st.charm=g["charm"]; st.vanna=g["vanna"]; st.vomma=g["vomma"]
                st.last_iv_t = now

    def _on_trade(self, rec):
        raw, opt = self._resolve(rec)
        if opt is None: return

        with self._lock:
            spot = float(self.spot_mid)
        if not (math.isfinite(spot) and spot > 0): return
        if not self._is_active(opt, spot): return

        sz   = int(getattr(rec, "size", 0) or 0)
        side = str(getattr(rec, "side", "N"))
        if isinstance(side, bytes): side = side.decode()
        if sz <= 0: return

        # Recuperer le prix du trade
        price = px_to_float(getattr(rec, "price", None))
        ts_ns = int(getattr(rec, "ts_recv", time.time_ns()))

        st = self.state.setdefault(raw, OptionState(dealer_pos=-float(opt.oi) * DEALER_OI_FACTOR))
        if side == "B":
            st.flow_pos -= sz
            with self._lock: self.flow_buy += sz
        elif side == "A":
            st.flow_pos += sz
            with self._lock: self.flow_sell += sz
        else: return

        st.dealer_pos = -float(opt.oi) * DEALER_OI_FACTOR + st.flow_pos
        with self._lock:
            self.flow_last_ts_ns = ts_ns

        # Compteurs diagnostic
        self._trade_count += 1

        # Alimenter le trade tape + bull/bear flow
        if math.isfinite(price) and price > 0:
            trade = self.tape.add_trade(raw, opt, side, sz, price, ts_ns, st)
            if trade is not None:
                self._trade_count_captured += 1
                if math.isfinite(st.delta):
                    self.flow.add(trade["side"], st.delta, sz, spot, ts_ns)

        # Log diagnostic toutes les 30s
        now_s = time.time()
        if now_s - self._last_trade_log >= 30.0:
            print(f"[trade-stats] recus={self._trade_count}, captures dans tape={self._trade_count_captured}")
            self._last_trade_log = now_s

    # ── SNAPSHOT ─────────────────────────────────────────────────────────────

    def snapshot(self):
        with self._lock:
            spot=float(self.spot_mid); spot_ts=int(self.spot_ts_ns)
            flow_buy=int(self.flow_buy); flow_sell=int(self.flow_sell)
            flow_ts=int(self.flow_last_ts_ns)

        if not (math.isfinite(spot) and spot > 0): return None

        rows = []
        for raw, opt in self.chain.items():
            st = self.state.get(raw)
            if st is None or not self._is_active(opt, spot): continue
            if not all(math.isfinite(getattr(st, g))
                       for g in ("gamma","iv","delta","charm","vanna","vomma")): continue
            dp=st.dealer_pos; S2=spot**2; M=CONTRACT_MULT
            gex_sign = -1.0 if opt.is_call else 1.0  # FIX: dp est negatif, on inverse pour calls
            rows.append({
                "raw":raw,"strike":opt.strike,"exp":opt.expiration,
                "dte":(opt.expiration-dt.date.today()).days,
                "is_call":opt.is_call,"oi":opt.oi,
                "dealer_pos":dp,"flow_pos":st.flow_pos,"mid":st.mid,
                "iv":st.iv,"delta":st.delta,"gamma":st.gamma,
                "vega":st.vega,"theta":st.theta,"charm":st.charm,
                "vanna":st.vanna,"vomma":st.vomma,
                "gex":gex_sign*dp*st.gamma*S2*M*0.01,
                "dex":dp*st.delta*spot*M,
                "cex":dp*st.charm*spot*M,
                "vanex":dp*st.vanna*spot*M*0.01,
                "vomex":dp*st.vomma*M,
            })

        # Si aucune option active, on retourne quand même un snapshot
        # minimal avec les candles (pour que le chart fonctionne hors-marché)
        if not rows:
            empty_buckets = {b: {"count": 0, "net_gex": 0.0, "gross_gex": 0.0,
                                 "net_dex": 0.0, "top_call_strike": None,
                                 "top_put_strike": None}
                             for b in ("0d", "1-7d", "8-30d", "30+d")}
            empty_snap = {
                "ts_ns":spot_ts,"spot":spot,"flow_buy":flow_buy,
                "flow_sell":flow_sell,"flow_ts_ns":flow_ts,
                "net_gex":0.0,"gross_gex":0.0,
                "net_dex":0.0,"net_cex":0.0,
                "net_vanex":0.0,"net_vomex":0.0,
                "gamma_flip":float("nan"),
                "charm_flip":float("nan"),
                "vanna_flip":float("nan"),
                "options_df":pd.DataFrame(columns=[
                    "raw","strike","exp","dte","is_call","oi","dealer_pos","flow_pos","mid",
                    "iv","delta","gamma","vega","theta","charm","vanna","vomma",
                    "gex","dex","cex","vanex","vomex","abs_gex",
                ]),
                "by_strike_df":pd.DataFrame(columns=["strike","gex","dex","cex","vanex","vomex","oi","abs_gex"]),
                "gamma_curve_df":pd.DataFrame({"spot":[],"net_gex":[]}),
                "candles": self.candles.get_all(),
                "key_levels": {"flip": None, "charm_flip": None, "vanna_flip": None,
                                "call_walls": [], "put_walls": [], "max_abs_strikes": []},
                "by_dte": empty_buckets,
                "greeks_history": self.greeks_ts.get_all(),
                "trades_tape": self.tape.get_recent(120),
                "trades_alerts": self.tape.get_alerts(20),
                "bb_flow": self.flow.get(),
            }
            try:
                empty_snap["market_analysis"] = self.reader.analyze(empty_snap)
            except Exception as e:
                print(f"[reader] error (empty): {e}")
                empty_snap["market_analysis"] = self.reader._empty()
            return empty_snap
        df  = pd.DataFrame(rows)
        df["abs_gex"] = df["gex"].abs()
        bys = df.groupby("strike")[["gex","dex","cex","vanex","vomex","oi","abs_gex"]].sum().sort_index().reset_index()

        median_iv = float(df["iv"].median()) if len(df) > 0 else 0.20
        scan_pct  = float(np.clip(median_iv * 2, SCAN_PCT_MIN, SCAN_PCT_MAX))

        curve_df, flip, charm_flip, vanna_flip = self._gamma_flip_scan(df, spot, scan_pct)

        # ── KEY LEVELS ───────────────────────────────────────────────────────
        bys_abs = bys.assign(abs_gex=lambda d: d["gex"].abs())
        # Per-side breakdown : montre toujours les top calls ET les top puts
        # independamment du net dominant (sinon en regime tres call-heavy on
        # ne voit aucun put wall et inversement).
        df_calls_only = df[df["is_call"]]
        df_puts_only  = df[~df["is_call"]]

        # Calcule le DTE dominant pour un strike (celui qui apporte le plus de |gex|).
        # Permet à MarketReader de tagger chaque niveau EPHEMERAL/WEEKLY/STRUCTURAL.
        def _dom_dte(df_sub, strike):
            sub = df_sub[df_sub["strike"] == strike]
            if sub.empty: return None
            g = sub.groupby("dte")["gex"].apply(lambda x: x.abs().sum())
            return int(g.idxmax()) if len(g) else None

        if len(df_calls_only):
            cps = (df_calls_only
                   .groupby("strike")
                   .agg(gex=("gex", "sum"), oi=("oi", "sum"))
                   .reset_index())
            call_walls = (
                cps[cps["gex"] > 0].nlargest(10, "gex")[["strike", "gex", "oi"]]
                .to_dict(orient="records")
            )
            for w in call_walls:
                w["dte"] = _dom_dte(df_calls_only, w["strike"])
        else:
            call_walls = []
        if len(df_puts_only):
            pps = (df_puts_only
                   .groupby("strike")
                   .agg(gex=("gex", "sum"), oi=("oi", "sum"))
                   .reset_index())
            put_walls = (
                pps[pps["gex"] < 0].nsmallest(10, "gex")[["strike", "gex", "oi"]]
                .to_dict(orient="records")
            )
            for w in put_walls:
                w["dte"] = _dom_dte(df_puts_only, w["strike"])
        else:
            put_walls = []
        # bys contient déjà oi sommé par strike (calls+puts), on l'ajoute ici aussi
        max_abs_strikes = (
            bys_abs.nlargest(5, "abs_gex")[["strike", "gex", "oi"]]
            .to_dict(orient="records")
        )
        for w in max_abs_strikes:
            w["dte"] = _dom_dte(df, w["strike"])
        key_levels = {
            "flip"           : float(flip)       if math.isfinite(flip)       else None,
            "charm_flip"     : float(charm_flip) if math.isfinite(charm_flip) else None,
            "vanna_flip"     : float(vanna_flip) if math.isfinite(vanna_flip) else None,
            "call_walls"     : call_walls,
            "put_walls"      : put_walls,
            "max_abs_strikes": max_abs_strikes,
        }

        # ── Breakdown par bucket DTE (0d / 1-7d / 8-30d / 30+d)
        DTE_BUCKETS = [
            ("0d",     0,  0),
            ("1-7d",   1,  7),
            ("8-30d",  8,  30),
            ("30+d",   31, 9999),
        ]
        by_dte = {}
        for name, lo_d, hi_d in DTE_BUCKETS:
            df_b = df[(df["dte"] >= lo_d) & (df["dte"] <= hi_d)]
            if not len(df_b):
                by_dte[name] = {
                    "count": 0, "net_gex": 0.0, "gross_gex": 0.0,
                    "net_dex": 0.0, "top_call_strike": None, "top_put_strike": None,
                }
                continue
            # Top call strike (par GEX positif des calls seuls)
            df_c = df_b[df_b["is_call"]]
            top_call = None
            if len(df_c):
                gc = df_c.groupby("strike")["gex"].sum()
                gc = gc[gc > 0]
                if len(gc):
                    top_call = float(gc.idxmax())
            # Top put strike (par GEX negatif des puts seuls)
            df_p = df_b[~df_b["is_call"]]
            top_put = None
            if len(df_p):
                gp = df_p.groupby("strike")["gex"].sum()
                gp = gp[gp < 0]
                if len(gp):
                    top_put = float(gp.idxmin())
            by_dte[name] = {
                "count"          : int(len(df_b)),
                "net_gex"        : float(df_b["gex"].sum()),
                "gross_gex"      : float(df_b["gex"].abs().sum()),
                "net_dex"        : float(df_b["dex"].sum()),
                "top_call_strike": top_call,
                "top_put_strike" : top_put,
            }

        # ── Mettre a jour le time-series 1m
        net_gex_total = float(df["gex"].sum())
        net_dex_total = float(df["dex"].sum())
        net_vanex_total = float(df["vanex"].sum())
        self.greeks_ts.update(spot_ts, net_gex_total, flip, net_dex_total, net_vanex_total)

        snap = {
            "ts_ns":spot_ts,"spot":spot,"flow_buy":flow_buy,
            "flow_sell":flow_sell,"flow_ts_ns":flow_ts,
            "net_gex":net_gex_total,"gross_gex":float(df["gex"].abs().sum()),
            "net_dex":net_dex_total,"net_cex":float(df["cex"].sum()),
            "net_vanex":net_vanex_total,"net_vomex":float(df["vomex"].sum()),
            "gamma_flip":flip,"charm_flip":charm_flip,"vanna_flip":vanna_flip,
            "options_df":df,"by_strike_df":bys,"gamma_curve_df":curve_df,
            "candles"        : self.candles.get_all(),
            "key_levels"     : key_levels,
            "by_dte"         : by_dte,
            "greeks_history" : self.greeks_ts.get_all(),
            "trades_tape"    : self.tape.get_recent(120),
            "trades_alerts"  : self.tape.get_alerts(20),
            "bb_flow"        : self.flow.get(),
        }

        # ── MarketReader : verdict trader ───────────────────────────────
        try:
            analysis = self.reader.analyze(snap)
        except Exception as e:
            print(f"[reader] error: {e}")
            analysis = self.reader._empty()
        snap["market_analysis"] = analysis

        # Propage strength tier dans chaque wall (pour les badges UI sans recalcul client)
        strength_map = {round(float(w["strike"]), 2): w.get("strength", "WEAK")
                        for w in analysis.get("walls_scored", [])}
        for wlist in (snap["key_levels"]["call_walls"], snap["key_levels"]["put_walls"],
                      snap["key_levels"]["max_abs_strikes"]):
            for w in wlist:
                w["strength"] = strength_map.get(round(float(w["strike"]), 2), "WEAK")

        return snap

    def _gamma_flip_scan(self, df, spot, scan_pct=SCAN_PCT_DEFAULT):
        """
        Scanne le spot sur ±scan_pct et calcule net_gex, net_cex, net_vanex pour
        chaque point. Retourne le DataFrame de courbes + les 3 zero-crossings
        (gamma_flip, charm_flip, vanna_flip).

        Note q=0 : avec DIV_YIELD=0, les termes q·N(d1)/q·N(-d1) de charm
        s'annulent, donc la formule charm est identique pour calls et puts —
        on n'a pas besoin de calculer N(d1) ici.
        """
        spots = spot * np.linspace(1 - scan_pct, 1 + scan_pct, SCAN_N)
        K       = df["strike"].to_numpy(float)
        sigma   = df["iv"].to_numpy(float)
        pos     = df["dealer_pos"].to_numpy(float)
        is_call = df["is_call"].to_numpy(bool)
        T       = np.array([yearfrac(d) for d in df["exp"]], dtype=float)

        m = (K > 0) & (sigma > 0) & (T > 0) & np.isfinite(pos)
        K, sigma, T, pos, is_call = K[m], sigma[m], T[m], pos[m], is_call[m]

        empty_curve = lambda: pd.DataFrame({
            "spot": spots,
            "net_gex":   np.zeros_like(spots),
            "net_cex":   np.zeros_like(spots),
            "net_vanex": np.zeros_like(spots),
        })
        nan = float("nan")
        if len(K) == 0:
            return empty_curve(), nan, nan, nan

        gex_sign = np.where(is_call, -1.0, 1.0)  # convention SpotGamma
        sqrtT    = np.sqrt(T)
        dfq      = np.exp(-DIV_YIELD * T)

        net_gex   = np.zeros_like(spots)
        net_cex   = np.zeros_like(spots)
        net_vanex = np.zeros_like(spots)
        M = CONTRACT_MULT

        for i, S in enumerate(spots):
            vs  = sigma * sqrtT
            d1  = (np.log(S / K) + (RISK_FREE - DIV_YIELD + 0.5 * sigma ** 2) * T) / vs
            d2  = d1 - vs
            pdf = (1.0 / SQRT_2PI) * np.exp(-0.5 * d1 * d1)

            gamma = dfq * pdf / (S * vs)
            # charm (q=0) : identique calls/puts
            charm = -dfq * pdf * (2.0 * (RISK_FREE - DIV_YIELD) * T - d2 * vs) / (2.0 * T * vs) / 365.0
            vanna = -dfq * pdf * d2 / sigma / 100.0

            net_gex[i]   = np.nansum(gex_sign * pos * gamma * S * S * M * 0.01)
            net_cex[i]   = np.nansum(pos * charm * S * M)
            net_vanex[i] = np.nansum(pos * vanna * S * M * 0.01)

        def _zero_cross(curve):
            idx = np.where(np.diff(np.sign(curve)) != 0)[0]
            if not len(idx): return nan
            # Si plusieurs flips, prendre celui le plus proche du spot courant
            best, best_dist = nan, float("inf")
            for j in idx:
                x0, x1 = spots[j], spots[j + 1]
                y0, y1 = curve[j], curve[j + 1]
                if y1 == y0: continue
                xf = float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))
                d  = abs(xf - spot)
                if d < best_dist:
                    best, best_dist = xf, d
            return best

        gamma_flip = _zero_cross(net_gex)
        charm_flip = _zero_cross(net_cex)
        vanna_flip = _zero_cross(net_vanex)

        curve_df = pd.DataFrame({
            "spot":      spots,
            "net_gex":   net_gex,
            "net_cex":   net_cex,
            "net_vanex": net_vanex,
        })
        return curve_df, gamma_flip, charm_flip, vanna_flip


# ─────────────────────────────────────────────
# STARTER - LIVE STREAMING
# ─────────────────────────────────────────────

def start_live_engine(*args, **kwargs) -> LiveGreeksEngine:
    eng = LiveGreeksEngine()

    def run_spot():
        print("[spot-feed] Connexion live NQ.c.0...")
        while True:
            try:
                live = db.Live(key=API_KEY)
                live.subscribe(
                    dataset=DATASET, schema="mbp-1",
                    symbols=[SPOT_SYMBOL], stype_in="continuous",
                )
                for rec in live:
                    eng.on_spot(rec)
            except Exception as e:
                print(f"[spot-feed] Erreur: {e} - reconnexion dans 5s")
                time.sleep(5)

    def run_options():
        print(f"[opt-feed] Connexion live {OPT_ROOTS}...")
        while True:
            try:
                live = db.Live(key=API_KEY)
                for root in OPT_ROOTS:
                    try:
                        live.subscribe(
                            dataset=DATASET, schema="mbp-1",
                            symbols=root, stype_in="parent",
                        )
                    except Exception as e:
                        print(f"[opt-feed] {root} mbp-1 skip: {e}")
                for root in OPT_ROOTS:
                    try:
                        live.subscribe(
                            dataset=DATASET, schema="trades",
                            symbols=root, stype_in="parent",
                        )
                    except Exception as e:
                        print(f"[opt-feed] {root} trades skip: {e}")
                for rec in live:
                    eng.on_option(rec)
            except Exception as e:
                print(f"[opt-feed] Erreur: {e} - reconnexion dans 5s")
                time.sleep(5)

    threading.Thread(target=run_spot,    daemon=True, name="spot-live").start()
    threading.Thread(target=run_options, daemon=True, name="opt-live").start()

    print("[engine] Feeds live demarres - premier snapshot dans quelques secondes.")
    return eng
