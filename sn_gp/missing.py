"""
find_missing.py  --  list events that are in fit_review.csv but NOT in the
population measurements, and say WHY each was excluded. Useful for catching
events where a 'good' verdict was forgotten during scanning.

    python find_missing.py                 # print table
    python find_missing.py --csv missing_events.csv   # also write a csv
"""

import os
import argparse
import numpy as np
import pandas as pd

import config
import measure_population as mp

MEAS = os.path.join(config.FIT_ROOT, "population_measurements.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="also write the list to this path")
    a = ap.parse_args()

    review = mp.load_review()          # has columns group, ZTFID, remark, verdict, kernel
    bic = mp.load_bic()

    # events actually measured (present in the population)
    measured = set()
    if os.path.exists(MEAS):
        measured = set(pd.read_csv(MEAS)["ZTFID"].astype(str))

    rows = []
    for z, grp in review.groupby("ZTFID"):
        if z in measured:
            continue                                  # it's in the population; fine

        verdicts = [v.strip().lower() for v in grp["verdict"]]
        remarks = " ".join(grp["remark"].str.lower())
        vset = set(verdicts)

        # classify WHY it's missing
        has_good = "good" in vset
        has_keep = "keep" in vset
        has_epoch_halfr = ("epoch" in remarks) or ("half_r" in remarks)

        # does any fit reconstruct-able JSON exist? (BIC present == JSON exists)
        kernels_with_json = [k for k in ("gibbs", "changepoint", "changepoint_1", "matern32")
                             if np.isfinite(bic.get((z, k), np.inf))]

        if has_epoch_halfr:
            reason = "dropped: epoch/half_r remark"
        elif not kernels_with_json:
            reason = "no JSON on disk for any kernel (never fit / all failed)"
        elif has_good:
            # has a good but still not measured -> likely reconstruct failure or
            # the good kernel's JSON missing
            reason = "HAS good but missing -> check reconstruct failure"
        elif has_keep:
            reason = "best available is 'keep' (no good) -> currently DROPPED?"
        else:
            reason = "ALL bad/blank -> forgotten 'good'?"

        rows.append(dict(
            ZTFID=z,
            verdicts=",".join(verdicts),
            remark=grp["remark"].str.strip().replace("", np.nan).dropna().unique().tolist(),
            kernels_with_json=",".join(kernels_with_json) if kernels_with_json else "-",
            reason=reason,
        ))

    df = pd.DataFrame(rows).sort_values("reason")
    if df.empty:
        print("no events in review are missing from the population.")
        return

    # print grouped by reason
    for reason, sub in df.groupby("reason"):
        print(f"\n=== {reason}  ({len(sub)}) ===")
        for _, r in sub.iterrows():
            print(f"  {r['ZTFID']:16s} verdicts=[{r['verdicts']}]  "
                  f"json=[{r['kernels_with_json']}]  remark={r['remark']}")

    print(f"\ntotal missing: {len(df)}")
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()