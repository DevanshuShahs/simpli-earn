"""Read-only ThetaData probe: subscriptions, column names and a few rows. Writes nothing; never prints credentials.

Credentials: a creds file (email on line 1, password on line 2) outside the repo.
    python scripts/theta_probe.py [--creds ~/.thetadata/creds.txt]
(or set THETADATA_API_KEY in the shell and omit the file). Small sample: KLAC around the 2Q26 event (split 2026-06-11).
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def step(name, fn):
    print(f"\n=== {name}")
    try:
        out = fn()
        if isinstance(out, pd.DataFrame):
            print(f"shape={out.shape}")
            print("columns:", list(out.columns))
            print(out.head(4).to_string())
        else:
            print(out)
        return out
    except Exception as e:  # noqa: BLE001 - probe: report and continue
        print(f"FAILED: {type(e).__name__}: {str(e)[:300]}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--creds", default=os.path.expanduser("~/.thetadata/creds.txt"))
    a = ap.parse_args()
    from thetadata import ThetaClient

    creds = Path(a.creds)
    try:
        if creds.exists():
            c = ThetaClient(creds_file=creds, dataframe_type="pandas")
        elif os.getenv("THETADATA_API_KEY"):
            c = ThetaClient(dataframe_type="pandas")
        else:
            sys.exit(f"No creds file at {creds} and THETADATA_API_KEY not set.")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Authentication failed: {type(e).__name__}: {str(e)[:200]}")

    print("subscriptions:", {k: getattr(c, k, "?") for k in ("stock_subscription", "options_subscription", "index_subscription")})
    exps = step("option_list_expirations KLAC", lambda: c.option_list_expirations("KLAC"))
    d0, d1 = dt.date(2026, 7, 15), dt.date(2026, 7, 17)
    exp = None
    if exps is not None:
        col = exps.columns[-1] if isinstance(exps, pd.DataFrame) else None
        vals = sorted(pd.to_datetime(exps[col]).dt.date) if col else []
        exp = next((e for e in vals if e >= dt.date(2026, 8, 14)), None)
        print("probe expiration:", exp, f"({len(vals)} expirations listed)")
    if exp:
        step("option_history_greeks_eod KLAC", lambda: c.option_history_greeks_eod(
            symbol="KLAC", expiration=exp, start_date=d0, end_date=d1, strike="*", right="both"))
        step("option_history_open_interest KLAC", lambda: c.option_history_open_interest(
            symbol="KLAC", expiration=exp, start_date=d0, end_date=d1, strike="*", right="both"))
    step("stock_history_eod KLAC across the 10-for-1 split", lambda: c.stock_history_eod(
        symbol="KLAC", start_date=dt.date(2026, 6, 9), end_date=dt.date(2026, 6, 15)))
    step("index_history_eod VIX", lambda: c.index_history_eod(symbol="VIX", start_date=d0, end_date=d1))
    step("stock_history_eod SMH", lambda: c.stock_history_eod(symbol="SMH", start_date=d0, end_date=d1))


if __name__ == "__main__":
    main()
