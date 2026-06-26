"""
app.py — Sauvegarde périodique des snapshots en Parquet.

Usage:
    $env:DATABENTO_API_KEY='db-votre-clé'
    python app.py
"""

import os
import time
import pandas as pd
from engine import start_live_engine

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUT_DIR     = "./data"
SAVE_EVERY  = 5.0   # seconds

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SAVE SNAPSHOT
# ─────────────────────────────────────────────

def save_snapshot(snap: dict):
    ts   = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"snap_{ts}")

    snap["options_df"].to_parquet(base + "_options.parquet",      index=False)
    snap["by_strike_df"].to_parquet(base + "_strikes.parquet",    index=False)
    snap["gamma_curve_df"].to_parquet(base + "_gamma_curve.parquet", index=False)

    metrics = pd.DataFrame([{
        "ts_ns"      : snap["ts_ns"],
        "spot"       : snap["spot"],
        "net_gex"    : snap["net_gex"],
        "gross_gex"  : snap["gross_gex"],
        "net_dex"    : snap["net_dex"],
        "net_cex"    : snap["net_cex"],
        "net_vanex"  : snap["net_vanex"],
        "net_vomex"  : snap["net_vomex"],
        "gamma_flip" : snap["gamma_flip"],
        "flow_buy"   : snap["flow_buy"],
        "flow_sell"  : snap["flow_sell"],
        "flow_ts_ns" : snap["flow_ts_ns"],
    }])
    metrics.to_parquet(base + "_metrics.parquet", index=False)
    print(f"[app] Saved {base}_*.parquet  |  spot={snap['spot']:.2f}"
          f"  net_gex={snap['net_gex']:+.1f}  flip={snap['gamma_flip']:.2f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    eng       = start_live_engine()
    last_save = 0.0

    while True:
        snap = eng.snapshot()
        if snap and (time.time() - last_save) > SAVE_EVERY:
            save_snapshot(snap)
            last_save = time.time()
        time.sleep(0.25)


if __name__ == "__main__":
    main()
