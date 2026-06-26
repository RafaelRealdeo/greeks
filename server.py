"""
server.py — Flask + SocketIO bridge entre l'engine live et le dashboard HTML.

Usage:
    $env:DATABENTO_API_KEY='db-votre-clé'
    python server.py

Puis ouvrir http://127.0.0.1:5050 dans votre navigateur.
Le dashboard reçoit les mises à jour en temps réel via WebSocket.
"""

import os
import math
import time
import threading
import traceback

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from engine import start_live_engine

# ── Config ────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 5050
# Intervalle entre chaque push WebSocket. La latence réelle perçue par le
# client dépend aussi du flux Databento, du réseau et du plan data — réduire
# cette valeur n'élimine pas ces autres sources de délai.
EMIT_INTERVAL = float(os.getenv("NQGS_EMIT_INTERVAL", "0.5"))

# ── Start engine ──────────────────────────────────────────────────────────────
print("[server] Starting engine…")
eng = start_live_engine()

# ── Flask + SocketIO ──────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def _safe_float(v):
    """Convertit en float JSON-safe (remplace NaN/Inf par None)."""
    if v is None: return None
    f = float(v)
    return f if math.isfinite(f) else None


def _serialize_snapshot(snap):
    """Sérialise un snapshot engine en dict JSON-safe."""
    flip = snap["gamma_flip"]

    # by_strike
    bys = snap["by_strike_df"].to_dict(orient="records")

    # gamma curve (downsample à ~67 points)
    curve = snap["gamma_curve_df"].iloc[::3].to_dict(orient="records")

    # Top 50 contrats par |gex| (defensive : retourne [] si df vide ou colonnes manquantes)
    opts_df = snap.get("options_df")
    opts = []
    iv_skew = []
    if opts_df is not None and len(opts_df) > 0 and "gex" in opts_df.columns:
        opts = (
            opts_df
            .assign(abs_gex=lambda d: d["gex"].abs())
            .nlargest(50, "abs_gex")
            [[
                "raw", "strike", "dte", "is_call", "oi",
                "iv", "delta", "gamma", "charm", "vanna", "vomma",
                "gex", "dex", "cex", "vanex", "vomex", "mid"
            ]]
            .to_dict(orient="records")
        )
        # IV skew data
        iv_df = opts_df[["strike", "iv", "is_call", "dte"]].copy()
        iv_df = iv_df[iv_df["iv"] > 0].copy()
        if len(iv_df) > 500:
            iv_df = iv_df.sample(500, random_state=42)
        iv_skew = iv_df.to_dict(orient="records")

    return {
        "status"     : "ok",
        "spot"       : _safe_float(snap["spot"]),
        "net_gex"    : _safe_float(snap["net_gex"]),
        "gross_gex"  : _safe_float(snap["gross_gex"]),
        "net_dex"    : _safe_float(snap["net_dex"]),
        "net_cex"    : _safe_float(snap["net_cex"]),
        "net_vanex"  : _safe_float(snap["net_vanex"]),
        "net_vomex"  : _safe_float(snap["net_vomex"]),
        "gamma_flip" : _safe_float(flip) if math.isfinite(flip) else None,
        "charm_flip" : _safe_float(snap.get("charm_flip")),
        "vanna_flip" : _safe_float(snap.get("vanna_flip")),
        "flow_buy"   : snap["flow_buy"],
        "flow_sell"  : snap["flow_sell"],
        "by_strike"  : bys,
        "gamma_curve": curve,
        "top_options": opts,
        "iv_skew"    : iv_skew,
        "candles"        : snap.get("candles", {}),
        "key_levels"     : snap.get("key_levels", {}),
        "by_dte"         : snap.get("by_dte", {}),
        "greeks_history" : snap.get("greeks_history", []),
        "trades_tape"    : snap.get("trades_tape", []),
        "trades_alerts"  : snap.get("trades_alerts", []),
        "bb_flow"        : snap.get("bb_flow", {}),
        "market_analysis": snap.get("market_analysis", {}),
        "ts_ns"          : snap["ts_ns"],
    }


# ── Routes HTTP (fallback) ───────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/snapshot")
def snapshot():
    """Endpoint HTTP classique (fallback si WebSocket indisponible)."""
    try:
        snap = eng.snapshot()
        if snap is None:
            return jsonify({"status": "waiting", "message": "No spot data yet — market may be closed."})
        return jsonify(_serialize_snapshot(snap))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/status")
def status():
    return jsonify({"status": "running", "symbol": "NQ"})


def _parse_int(name, default=None):
    v = request.args.get(name)
    if v is None or v == "":
        return default
    try: return int(v)
    except ValueError: return default

def _parse_float(name, default=None):
    v = request.args.get(name)
    if v is None or v == "":
        return default
    try: return float(v)
    except ValueError: return default


@app.route("/trades/history")
def trades_history():
    """
    Replay des trades persistés.
    Query params:
      since      : timestamp ns (default = début de la session NY courante)
      until      : timestamp ns
      min_premium: filtre $ premium
      min_size   : filtre lot size
      flag       : BLOCK | BIG | FAST | %OI
      is_call    : 1/0 pour calls/puts seulement
      limit      : max rows (default 500, cap 5000)
    """
    if eng.store is None:
        return jsonify({"status": "disabled", "message": "Persistance désactivée (NQGS_DISABLE_PERSIST=1)"})

    since   = _parse_int("since")
    until   = _parse_int("until")
    minprem = _parse_float("min_premium")
    minsize = _parse_int("min_size")
    flag    = request.args.get("flag") or None
    isc_raw = request.args.get("is_call")
    is_call = None if isc_raw in (None, "") else (isc_raw not in ("0", "false", "False"))
    limit   = max(1, min(_parse_int("limit", 500) or 500, 5000))

    try:
        rows = eng.store.query(
            since_ns=since, until_ns=until,
            min_premium=minprem, min_size=minsize,
            flag=flag, is_call=is_call, limit=limit,
        )
        return jsonify({"status": "ok", "count": len(rows), "trades": rows})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/trades/stats")
def trades_stats():
    """Aggregates des trades persistés. since_ns par défaut = -24h."""
    if eng.store is None:
        return jsonify({"status": "disabled", "message": "Persistance désactivée."})

    since = _parse_int("since")
    until = _parse_int("until")
    if since is None:
        since = int(time.time_ns() - 24 * 3600 * 1_000_000_000)
    try:
        st = eng.store.stats(since_ns=since, until_ns=until)
        st["since_ns"] = since
        if until is not None:
            st["until_ns"] = until
        return jsonify({"status": "ok", **st})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── WebSocket push loop ──────────────────────────────────────────────────────

def _ws_push_loop():
    """Thread qui push un snapshot toutes les EMIT_INTERVAL secondes."""
    time.sleep(3)  # laisser l'engine démarrer
    while True:
        try:
            snap = eng.snapshot()
            if snap is not None:
                payload = _serialize_snapshot(snap)
                socketio.emit("snapshot", payload)
            else:
                socketio.emit("snapshot", {
                    "status": "waiting",
                    "message": "No spot data yet — market may be closed."
                })
        except Exception as e:
            print(f"[ws-push] Erreur sérialisation: {e}")
            traceback.print_exc()
        time.sleep(EMIT_INTERVAL)


threading.Thread(target=_ws_push_loop, daemon=True, name="ws-push").start()


# ── SocketIO events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    print("[ws] Client connecté")
    # Envoie un snapshot immédiat plutôt que d'attendre le prochain tick de
    # la boucle périodique (qui peut survenir jusqu'à EMIT_INTERVAL plus tard).
    try:
        snap = eng.snapshot()
        if snap is not None:
            socketio.emit("snapshot", _serialize_snapshot(snap), to=request.sid)
        else:
            socketio.emit("snapshot", {
                "status": "waiting",
                "message": "No spot data yet — market may be closed."
            }, to=request.sid)
    except Exception as e:
        socketio.emit("snapshot", {"status": "error", "message": str(e)}, to=request.sid)

@socketio.on("disconnect")
def on_disconnect():
    print("[ws] Client déconnecté")

@socketio.on("request_snapshot")
def on_request_snapshot():
    """Le client peut demander un snapshot immédiat."""
    try:
        snap = eng.snapshot()
        if snap is not None:
            socketio.emit("snapshot", _serialize_snapshot(snap))
        else:
            socketio.emit("snapshot", {
                "status": "waiting",
                "message": "No spot data yet — market may be closed."
            })
    except Exception as e:
        socketio.emit("snapshot", {"status": "error", "message": str(e)})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[server] Dashboard running at http://{HOST}:{PORT}")
    print(f"[server] WebSocket push every {EMIT_INTERVAL}s")
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
