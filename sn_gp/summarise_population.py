"""
summarise_population.py  --  print population statistics from
fit/population_measurements.csv. Standalone; no plotting.

    python summarise_population.py
    python summarise_population.py --csv summary.csv     # also write per-type table

Reports: total events, counts per SN type, and per-type + overall statistics
(median, mean, std, N) for every measured parameter -- duration, rest-frame
luminosity, colours, peak-time separation, rise/fade. Plus the specific
Ib+Ic median g-r at +10d your supervisor asked for.
"""

import os
import argparse
import numpy as np
import pandas as pd

import config

CSV = os.path.join(config.FIT_ROOT, "population_measurements.csv")

TYPE_ORDER = ["SN Ic", "SN Ib", "SN Ib/c", "SN Ic-BL", "SN Ic-BL?", "SLSN-I"]

# parameters to summarise: (column, label, rest-frame?)
PARAMS = [
    ("fwhm_r",             "FWHM r [d, rest]"),
    ("fwhm_g",             "FWHM g [d, rest]"),
    ("fwhm_brightest",     "FWHM brightest [d, rest]"),
    ("M_rest_g",           "M rest-g [mag, K-corr]"),
    ("M_r",                "M_r [mag, obs-frame]"),
    ("M_g",                "M_g [mag, obs-frame]"),
    ("color_gr",           "g-r own-peak [mag]"),
    ("color_at_rpeak",     "g-r at r-peak [mag]"),
    ("color_at_gpeak",     "g-r at g-peak [mag]"),
    ("color_10d",          "g-r +10d [mag]"),
    ("dt_peak_g_minus_r",  "t_g - t_r [d]"),
    ("rise_r",             "rise r [d, rest]"),
    ("fade_r",             "fade r [d, rest]"),
]


def _stats(series):
    v = pd.to_numeric(series, errors="coerce").dropna()
    if len(v) == 0:
        return dict(n=0, median=np.nan, mean=np.nan, std=np.nan)
    return dict(n=len(v), median=float(v.median()),
                mean=float(v.mean()), std=float(v.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="also write per-type table to this path")
    a = ap.parse_args()

    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run measure_population.py first")
    df = pd.read_csv(CSV)

    n_total = len(df)
    print("=" * 70)
    print(f"POPULATION SUMMARY   ({n_total} events total)")
    print("=" * 70)

    # ---- counts per type ----
    print("\nCounts by type:")
    counts = df["type"].value_counts()
    for t in TYPE_ORDER:
        c = int(counts.get(t, 0))
        print(f"  {t:12s}: {c:4d}  ({100*c/n_total:.1f}%)")
    other = int((~df["type"].isin(TYPE_ORDER)).sum())
    if other:
        print(f"  {'other':12s}: {other:4d}  ({100*other/n_total:.1f}%)")

    # ---- per-parameter, per-type statistics ----
    rows = []
    for col, label in PARAMS:
        if col not in df.columns:
            continue
        print(f"\n{label}   (column: {col})")
        print(f"  {'type':12s} {'N':>4s} {'median':>9s} {'mean':>9s} {'std':>8s}")
        # overall
        s = _stats(df[col])
        print(f"  {'ALL':12s} {s['n']:>4d} {s['median']:>9.3f} {s['mean']:>9.3f} {s['std']:>8.3f}")
        rows.append(dict(param=label, type="ALL", **s))
        # per type
        for t in TYPE_ORDER:
            s = _stats(df.loc[df["type"] == t, col])
            if s["n"] == 0:
                continue
            print(f"  {t:12s} {s['n']:>4d} {s['median']:>9.3f} {s['mean']:>9.3f} {s['std']:>8.3f}")
            rows.append(dict(param=label, type=t, **s))

    # ---- the specific ask: Ib + Ic median g-r at +10d ----
    print("\n" + "=" * 70)
    print("SUPERVISOR ASK: median g-r at +10d (rest-frame), Ib + Ic ONLY")
    print("=" * 70)
    if "color_10d" in df.columns:
        ibic = df[df["type"].isin(["SN Ib", "SN Ic"])]
        v = pd.to_numeric(ibic["color_10d"], errors="coerce").dropna()
        # also each separately for context
        for t in ["SN Ib", "SN Ic"]:
            vt = pd.to_numeric(df.loc[df["type"] == t, "color_10d"], errors="coerce").dropna()
            if len(vt):
                print(f"  {t:8s}: median = {vt.median():.3f}  (n={len(vt)})")
        if len(v):
            print(f"  Ib+Ic combined: median g-r(+10d) = {v.median():.3f} mag  "
                  f"(n={len(v)}, mean={v.mean():.3f}, std={v.std():.3f})")
        else:
            print("  no finite color_10d for Ib/Ic")
    else:
        print("  color_10d column missing -- re-run measure_population.py")

    # ---- flags worth noting ----
    if "kcorr_wave_extrap" in df.columns:
        n_ex = int(df["kcorr_wave_extrap"].fillna(False).sum())
        print(f"\nNote: {n_ex} events have M_rest_g from wavelength EXTRAPOLATION "
              f"(z high enough that rest-g falls past r); treat their luminosity with caution.")

    if a.csv:
        pd.DataFrame(rows).to_csv(a.csv, index=False)
        print(f"\nwrote per-type table -> {a.csv}")


if __name__ == "__main__":
    main()