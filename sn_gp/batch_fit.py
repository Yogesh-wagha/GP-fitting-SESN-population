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

REVIEW = os.path.join(config.FIT_ROOT, "fit_review.csv")

def rework_plan():
    """Return {ZTFID: [kernels_to_refit]} for events that have any remark.
       kernels = those with verdict 'good'; if none good -> all four kernels."""
    d = pd.read_csv(REVIEW, dtype=str).fillna("")
    all_kernels = [k for _, k in CONFIGS]          # gibbs, changepoint, changepoint_1, matern32
    flagged, goods = {}, {}
    for _, r in d.iterrows():
        z = r["ZTFID"]
        grp = r["group"]                           # e.g. 'gibbs_constant'
        kern = grp.rsplit("_constant", 1)[0]       # -> 'gibbs' / 'changepoint' / 'changepoint_1' / 'matern32'
        goods.setdefault(z, set()); flagged.setdefault(z, False)
        if r["remark"].strip():
            flagged[z] = True
        if r["verdict"].strip().lower() == "good":
            goods[z].add(kern)
    return {z: (sorted(goods[z]) if goods[z] else list(all_kernels))
            for z, f in flagged.items() if f}

def select_events(cuts):
    b = pd.read_csv(BTS_LIST)
    col = "ZTFID" if "ZTFID" in b.columns else b.columns[0]
    names = [str(z) for z in b[col].dropna()]
    # keep only those with photometry on disk (can't fit what has no CSV)
    return sorted([z for z in names if csv_exists(z)])


def fit_one(name, mean, kernel, cuts, dry=False, force=False):
    info = cuts.get(name, {})
    cmd = [sys.executable, RUN_PY, "--name", name,
           "--kernel", kernel, "--mean_func", mean, "--gri"]
    if info.get("lower"): cmd += ["--left", info["lower"]]
    if info.get("upper"): cmd += ["--right", info["upper"]]
    jpath = os.path.join(config.JSON_DIR, f"{name}_{kernel}_{mean}.json")
    if os.path.exists(jpath) and not force:
        return "cached"
    if dry:
        print("   " + " ".join(cmd)); return "dry"
    try:
        subprocess.run(cmd, timeout=TIMEOUT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return "ok" if os.path.exists(jpath) else "no_output"
    except subprocess.TimeoutExpired:
        return "timeout"
    except subprocess.CalledProcessError as e:
        print(f"    ERR {name} {kernel}: {e.stderr.decode()[-300:]}")
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
    ap.add_argument("--rework", action="store_true",
        help="re-fit only events with a remark in fit_review.csv; only their "
             "'good' kernels (all four if none good); cuts from manual_cuts.csv; "
             "overwrites existing PNG/JSON.")
    args = ap.parse_args()

    cuts = load_cuts()
    events = select_events(cuts)

    if args.compile_only:
        compile_metrics(events)
        return

    # ---- REWORK branch (insert here) ----
    if args.rework:
        plan = rework_plan()
        total = sum(len(v) for v in plan.values())
        print(f"REWORK: {len(plan)} events, {total} (event,kernel) fits\n")
        for z, kernels in sorted(plan.items()):
            info = cuts.get(z, {})
            l, r = info.get("lower", ""), info.get("upper", "")
            print(f"  {z:16s} kernels={kernels}  left={l or '-'} right={r or '-'}")
        if args.dry_run:
            return
        done = 0
        for z, kernels in sorted(plan.items()):
            for kern in kernels:
                done += 1
                st = fit_one(z, "constant", kern, cuts, force=True)   # overwrite
                print(f"[{done}/{total}] {z}  {kern}  -> {st}", flush=True)
        compile_metrics(events)
        write_best_fit(events)
        return
    # ---- end REWORK branch ----

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
