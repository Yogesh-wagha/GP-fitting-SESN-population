"""
batch_fit.py  --  fit a random 100 BTS events x 6 (mean,kernel) configs.

Lives in the sn_gp/ folder next to run.py. Calls run.py once per fit (isolated:
a crash/NaN in one fit can't kill the batch), skips fits already done (so it
RESUMES after a SLURM timeout), and compiles all metrics into one CSV.

Selection: events come from BTS.csv, cuts/remarks from manual_cuts.csv.
  - skip events whose remark contains 'peak' or 'epoch'
  - skip events with no photometry CSV
  - lower_cut -> --left,  upper_cut -> --right   (only when present)
The chosen 100 (and their order) are frozen in selected_events.csv so the
website and summary always use the same set/order.

Run:
    python3 batch_fit.py --dry-run     # print the commands, fit nothing
    python3 batch_fit.py               # run everything (resumable)
    python3 batch_fit.py --compile-only
"""

import os
import sys
import json
import random
import argparse
import subprocess
import pandas as pd

import config                      # from sn_gp (gives config.CSV_DIR)

HERE    = os.path.dirname(os.path.abspath(__file__))
RUN_PY  = os.path.join(HERE, "run.py")
FIGS    = os.path.join(HERE, "figs")

GP_SN   = os.path.expanduser("~/GP_SN")
BTS_LIST = os.path.join(GP_SN, "BTS.csv")
CUTS_CSV = os.path.join(GP_SN, "review_site", "manual_cuts.csv")
RESULTS  = os.path.join(GP_SN, "fit_results")
SELECTED = os.path.join(RESULTS, "selected_events.csv")
METRICS  = os.path.join(RESULTS, "fit_metrics.csv")

N_EVENTS = 200
SEED     = 41
TIMEOUT  = 900          # seconds per single fit

# (mean, kernel) -- the six groups, in the order shown on the website
CONFIGS = [
    ("constant", "gibbs"),
    ("constant", "changepoint"),      # 3-segment
    ("constant", "changepoint_1"),    # 2-segment
    ("constant", "matern32"),
]


def load_cuts():
    d = pd.read_csv(CUTS_CSV, dtype=str).fillna("")
    out = {}
    for _, r in d.iterrows():
        out[r["ZTFID"]] = dict(
            remarks=r.get("remarks", "").lower(),
            lower=str(r.get("lower_cut", "")).strip(),
            upper=str(r.get("upper_cut", "")).strip(),
        )
    return out


def csv_exists(name):
    return os.path.exists(os.path.join(config.CSV_DIR, f"{name}.csv"))


def build_pool(cuts):
    names = None
    if os.path.exists(BTS_LIST):
        b = pd.read_csv(BTS_LIST)
        col = "ZTFID" if "ZTFID" in b.columns else b.columns[0]
        names = set(b[col].dropna().astype(str))

    pool = []
    for z, info in cuts.items():
        if "peak" in info["remarks"] or "epoch" in info["remarks"]:
            continue                               # exclude bad-peak / too-few-epoch
        if names is not None and z not in names:
            continue
        if not csv_exists(z):
            continue
        pool.append(z)
    return sorted(pool)


def select_events(cuts):
    b = pd.read_csv(BTS_LIST)
    col = "ZTFID" if "ZTFID" in b.columns else b.columns[0]
    names = [str(z) for z in b[col].dropna()]
    # keep only those with photometry on disk (can't fit what has no CSV)
    return sorted([z for z in names if csv_exists(z)])


def fit_one(name, mean, kernel, cuts, dry=False):
    info = cuts.get(name, {})
    cmd = [sys.executable, RUN_PY, "--name", name,
           "--kernel", kernel, "--mean_func", mean, "--gri"]
    if info.get("lower"): cmd += ["--left", info["lower"]]
    if info.get("upper"): cmd += ["--right", info["upper"]]
    ov = config.load_cp_override(name)                 # (d1, d2) or None
    if ov is not None:
        d1, d2 = ov
        s = str(d1) + ("," + str(d2) if d2 is not None else "")
        cmd += ["--cp_loc", s]
    jpath = os.path.join(config.JSON_DIR, f"{name}_{kernel}_{mean}.json")

    if os.path.exists(jpath):
        return "cached"
    if dry:
        print("   " + " ".join(cmd))
        return "dry"
    try:
        subprocess.run(cmd, timeout=TIMEOUT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "ok" if os.path.exists(jpath) else "no_output"
    except subprocess.TimeoutExpired:
        return "timeout"
    except subprocess.CalledProcessError:
        return "error"


def compile_metrics(events):
    rows = []
    for z in events:
        for mean, kernel in CONFIGS:
            jp = os.path.join(FIGS, f"{z}_{kernel}_{mean}.json")
            row = dict(ZTFID=z, mean=mean, kernel=kernel, group=f"{kernel}_{mean}")
            if os.path.exists(jp):
                with open(jp) as f:
                    row.update(json.load(f))
                row["status"] = "ok"
            else:
                row["status"] = "missing"
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(METRICS, index=False)
    n_ok = int((df.status == "ok").sum())
    print(f"metrics -> {METRICS}   ({n_ok}/{len(df)} fits present)")
    return df

def write_best_fit(events):
    rows = []
    for z in events:
        best = None
        for mean, kernel in CONFIGS:
            jp = os.path.join(config.JSON_DIR, f"{z}_{kernel}_{mean}.json")
            if not os.path.exists(jp): continue
            r = json.load(open(jp))
            if best is None or r["BIC"] < best["BIC"]:
                best = dict(ZTFID=z, best_kernel=kernel, AIC=r["AIC"], BIC=r["BIC"])
        if best: rows.append(best)
    out = os.path.join(config.FIT_ROOT, "best_fit.csv")
    pd.DataFrame(rows, columns=["ZTFID","best_kernel","AIC","BIC"]).to_csv(out, index=False)
    print(f"best-fit summary -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--compile-only", action="store_true",
                    help="just rebuild fit_metrics.csv from existing JSONs")
    args = ap.parse_args()

    cuts = load_cuts()
    events = select_events(cuts)

    if args.compile_only:
        compile_metrics(events)
        return

    total, done = len(events) * len(CONFIGS), 0
    for z in events:
        for mean, kernel in CONFIGS:
            done += 1
            st = fit_one(z, mean, kernel, cuts, dry=args.dry_run)
            print(f"[{done}/{total}] {z}  {mean}+{kernel}  -> {st}", flush=True)

    if not args.dry_run:
        compile_metrics(events)

    write_best_fit(events)

if __name__ == "__main__":
    main()
