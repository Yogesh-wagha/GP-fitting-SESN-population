"""
plot_population_html.py  --  interactive duration-luminosity plots (Plotly).

Writes standalone .html files (no server / display needed): generate on the
cluster, scp the .html to your laptop, open it, hover any point to see the
supernova name, redshift, type, kernel and filter. Ideal for chasing outliers.

    python plot_html.py --plot durlum_r
    python plot_html.py --plot durlum_g
    python plot_html.py --plot durlum_brightest
    python plot_html.py --plot all

Colour = coarse type group (your scheme); marker shape = fine subtype.
Duration = rest-frame FWHM [days]; luminosity = peak absolute mag (y inverted).
Not K-corrected.
"""

import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config

CSV = os.path.join(config.FIT_ROOT, "population_measurements.csv")
OUTDIR = os.path.join(config.FIT_ROOT, "population_plots")
os.makedirs(OUTDIR, exist_ok=True)

# colour per type (your scheme) + a distinct marker shape per fine subtype
"#FF0000"
TYPE_COLOR = {
    "SN Ic-BL":  "#8A0D67",
    "SN Ic-BL?": "#8A0D67",
    "SN Ic":     "#1017F1",
    "SN Ib":     "#1017F1",
    "SN Ib/c":   "#1017F1",
    "SLSN-I":    "#1FCFD8",
}
TYPE_SYMBOL = {
    "SN Ic-BL":  "square",
    "SN Ic-BL?": "square",
    "SN Ic":     "circle",
    "SN Ib":     "triangle-up",
    "SN Ib/c":   "diamond",
    "SLSN-I":    "star",
}
TYPE_ORDER = ["SN Ic", "SN Ib", "SN Ib/c", "SN Ic-BL", "SN Ic-BL?", "SLSN-I"]

CONSTRAINING_DAYS = 16.0
FAST_RISE_DAYS = 8.0

def _load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run measure_population.py first")
    return pd.read_csv(CSV)


def durlum_html(df, band, show_limits=False):
    if band == "brightest":
        xcol, ycol, title = "fwhm_brightest", "M_brightest", "Duration-luminosity (brightest of g/r)"
        fcol, ll_col, lim_col = "brightest_band", "fwhm_brightest_ll", "brightest_is_limit"
    else:
        xcol, ycol, title = f"fwhm_{band}", f"M_{band}", f"Duration-luminosity ({band} band)"
        fcol, ll_col, lim_col = f"{band}_filter", f"fwhm_{band}_ll", f"{band}_is_limit"

    fig = go.Figure()

    def _hovertemplate(is_limit):
        xlab = "FWHM ≥ %{x:.1f} d  (lower limit)" if is_limit else "FWHM: %{x:.1f} d"
        return ("<b>%{customdata[0]}</b><br>"
                "type: %{customdata[2]}<br>"
                "z: %{customdata[1]}<br>"
                "kernel: %{customdata[3]}<br>"
                "filter: %{customdata[4]}<br>"
                + xlab + "<br>"
                "M: %{y:.2f}<extra></extra>")

    def _custom(sub):
        return np.stack([sub["ZTFID"].astype(str), sub["z"].round(4).astype(str),
                         sub["type"].astype(str), sub["kernel"].astype(str),
                         sub[fcol].astype(str)], axis=-1)

    for t in TYPE_ORDER:
        base = (df["type"] == t) & np.isfinite(df[ycol])

        # finite (fully-measured) FWHM -> filled marker, use xcol
        meas = base & np.isfinite(df[xcol]) & ~df[lim_col].fillna(False)
        sub = df[meas]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub[xcol], y=sub[ycol], mode="markers", name=t,
                legendgroup=t,
                marker=dict(size=6, color=TYPE_COLOR[t], symbol=TYPE_SYMBOL[t],
                            line=dict(width=0.6, color="black")),
                customdata=_custom(sub), hovertemplate=_hovertemplate(False)))

        # constraining lower limits -> OPEN marker, use ll_col
        if show_limits:
            lim = base & df[lim_col].fillna(False) & np.isfinite(df[ll_col])
            constraining = df[ll_col] >= CONSTRAINING_DAYS
            if f"rise_{band}" in df.columns:
                fast = np.isfinite(df[f"rise_{band}"]) & (df[f"rise_{band}"] < FAST_RISE_DAYS)
                lim &= (constraining | fast)
            else:
                lim &= constraining
            sub = df[lim]
            if not sub.empty:
                open_sym = TYPE_SYMBOL[t] if TYPE_SYMBOL[t].endswith("-open") \
                           else TYPE_SYMBOL[t] + "-open"
                fig.add_trace(go.Scatter(
                    x=sub[ll_col], y=sub[ycol], mode="markers",
                    name=f"{t} (limit)", legendgroup=t, showlegend=False,
                    marker=dict(size=7, color=TYPE_COLOR[t], symbol=open_sym,
                                line=dict(width=1.2, color=TYPE_COLOR[t])),
                    customdata=_custom(sub), hovertemplate=_hovertemplate(True)))

    fig.update_layout(
        title=title,
        xaxis_title="rest-frame FWHM [days]",
        yaxis_title="peak absolute magnitude",
        yaxis=dict(autorange="reversed"),
        legend_title="type", template="simple_white",
        width=850, height=650,
    )
    suffix = "_lim" if show_limits else ""
    out = os.path.join(OUTDIR, f"durlum_{band}{suffix}.html")
    fig.write_html(out, include_plotlyjs=True)   # embedded, works offline
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", required=True,
        choices=["durlum_g", "durlum_r", "durlum_brightest", "all"])
    ap.add_argument("--limits", action="store_true",
        help="overlay constraining lower-limit events as open markers")
    a = ap.parse_args()
    df = _load()
    which = ["durlum_g", "durlum_r", "durlum_brightest"] if a.plot == "all" else [a.plot]
    for p in which:
        durlum_html(df, p.split("_")[1], show_limits=a.limits)


if __name__ == "__main__":
    main()
