"""
summarize.py  --  compare the 6 methods across all fitted events.

Reads fit_results/fit_metrics.csv and writes plots to fit_results/summary/.
Also prints a comparison table.

Which method is "best for almost all LCs" is answered three ways:
  - BIC/AIC distribution  : where each method sits overall (lower = better)
  - BIC win count         : on how many events each method has the LOWEST BIC
  - mean BIC rank         : average ranking across events (1 = best)
  - success rate          : how often a method fit at all (robustness)
A good method is low-BIC, wins often, ranks near 1, AND rarely fails.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.expanduser("~/GP_SN/fit_results")
METRICS = os.path.join(RESULTS, "fit_metrics.csv")
SUMDIR  = os.path.join(RESULTS, "summary")
os.makedirs(SUMDIR, exist_ok=True)

CONFIGS = [
    ("constant", "gibbs"),("constant", "changepoint"),
    ("constant", "matern32"), ("bazin", "matern32"),
    ("bazin", "gibbs"), ("bazin", "changepoint"),
]
# CONFIGS = [("constant", "gibbs")]
GROUPS = [f"{k}_{m}" for m, k in CONFIGS]
LABELS = [f"{m}+{k}" for m, k in CONFIGS]


def _bar(values, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(LABELS, values)
    ax.set_ylabel(ylabel); ax.set_title(title)
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(SUMDIR, fname), dpi=150); plt.close(fig)


def main():
    df = pd.read_csv(METRICS)
    ok = df[df.status == "ok"].copy()
    for c in ("AIC", "BIC", "lnL", "k"):
        ok[c] = pd.to_numeric(ok.get(c), errors="coerce")

    # --- AIC / BIC distributions ---
    for metric in ("BIC", "AIC"):
        data = [ok.loc[ok.group == g, metric].dropna().values for g in GROUPS]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data, tick_labels=LABELS, showfliers=False)
        ax.set_ylabel(metric); ax.set_title(f"{metric} distribution by method")
        plt.xticks(rotation=30, ha="right"); fig.tight_layout()
        fig.savefig(os.path.join(SUMDIR, f"{metric}_box.png"), dpi=150); plt.close(fig)

    # --- BIC win count (lowest BIC per event, using whatever is available) ---
    piv = ok.pivot_table(index="ZTFID", columns="group", values="BIC").reindex(columns=GROUPS)
    wins = {g: 0 for g in GROUPS}
    for _, row in piv.iterrows():
        if row.notna().any():
            wins[row.idxmin()] += 1
    _bar([wins[g] for g in GROUPS], "# events with lowest BIC",
         "BIC win count", "BIC_wins.png")

    # --- mean rank on complete cases (events where all 6 succeeded) ---
    comp = piv.dropna()
    if len(comp):
        ranks = comp.rank(axis=1, method="min")
        meanrank = ranks.mean()
        _bar([meanrank[g] for g in GROUPS], "mean BIC rank (1 = best)",
             f"mean rank over {len(comp)} complete events", "BIC_rank.png")
    else:
        meanrank = pd.Series({g: np.nan for g in GROUPS})

    # --- success rate (robustness) ---
    cov = df.groupby("group").apply(lambda d: int((d.status == "ok").sum())).reindex(GROUPS).fillna(0)
    _bar(cov.values, "# successful fits", "fit success per method", "coverage.png")

    # --- table ---
    print(f"\n{'method':18s} {'n_ok':>5s} {'medBIC':>9s} {'medAIC':>9s} "
          f"{'wins':>5s} {'rank':>6s}")
    for g, l in zip(GROUPS, LABELS):
        sub = ok[ok.group == g]
        med_b = sub.BIC.median() if len(sub) else np.nan
        med_a = sub.AIC.median() if len(sub) else np.nan
        print(f"{l:18s} {len(sub):5d} {med_b:9.1f} {med_a:9.1f} "
              f"{wins[g]:5d} {meanrank.get(g, np.nan):6.2f}")
    print(f"\nplots -> {SUMDIR}")


if __name__ == "__main__":
    main()
