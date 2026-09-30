"""
morph_review.py  --  web app for visually classifying light-curve MORPHOLOGY.

Separate from review_fits.py (which judges fit quality). This one is about the
SHAPE of the light curve: bumps, double peaks, plateaus, undulations.

  * plots are in MAGNITUDE space (y inverted), not flux
  * two panels per event: (a) the analysis bands g/r (+ i if present),
    (b) every filter in the fit
  * measured parameters shown alongside so you judge shape with the numbers visible
  * one primary shape (radio) + any number of feature flags (checkboxes) + notes

    python morph_review.py            # then open http://localhost:5003
    ssh -L 5003:localhost:5003 ariywagh@prospero...   # if running on the cluster

Writes fit/morphology_review.csv  (ZTFID, shape, features, notes).
"""

import os
import io
import csv
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, render_template_string

import config
import reconstruct
import measure_population as mp

PORT = 5003
MEAS = os.path.join(config.FIT_ROOT, "population_measurements.csv")
OUT = os.path.join(config.FIT_ROOT, "morphology_review.csv")
PNG_CACHE = os.path.join(config.FIT_ROOT, "morph_png")
os.makedirs(PNG_CACHE, exist_ok=True)

F0_mJy = 3631e3

# ---- the classification vocabulary -------------------------------------
SHAPES = [
    ("single",     "Single smooth peak (normal)"),
    ("double",     "Double peak"),
    ("earlybump",  "Early bump then main peak (shock cooling?)"),
    ("plateau",    "Plateau / flat top"),
    ("linear",     "Linear decline (no clear peak curvature)"),
    ("irregular",  "Irregular / no clean shape"),
    ("unclear",    "Too sparse to judge"),
]
FEATURES = [
    ("bumpy",       "Bumpy / undulating"),
    ("shoulder",    "Shoulder or knee on decline"),
    ("rebright",    "Late re-brightening"),
    ("sharp",       "Sharp / narrow peak"),
    ("broad",       "Broad / rounded peak"),
    ("fastdecline", "Fast decline"),
    ("slowtail",    "Slow, shallow tail"),
    ("asymmetric",  "Strongly asymmetric (fast rise, slow fade)"),
    ("colorodd",    "g and r shapes disagree"),
    ("noisy",       "Noisy / poor sampling"),
]

BAND_COLOR = {"g": "#1b9e2f", "r": "#d62728", "i": "#8c564b",
              "u": "#7570b3", "z": "#555555"}

# parameters shown in the side panel: (column, label, format)
SHOW_PARAMS = [
    ("type",               "Type",               "{}"),
    ("z",                  "Redshift",           "{:.4f}"),
    ("kernel",             "Kernel",             "{}"),
    ("g_filter",           "g filter used",      "{}"),
    ("r_filter",           "r filter used",      "{}"),
    ("fwhm_r",             "FWHM r (rest)",      "{:.1f} d"),
    ("fwhm_g",             "FWHM g (rest)",      "{:.1f} d"),
    ("rise_r",             "rise r (rest)",      "{:.1f} d"),
    ("fade_r",             "fade r (rest)",      "{:.1f} d"),
    ("rise_frac_r",        "rise / FWHM (r)",    "{:.2f}"),
    ("M_rest_g",           "M rest-g (K-corr)",  "{:.2f}"),
    ("M_rest_g_mw",        "M rest-g (K+MW)",    "{:.2f}"),
    ("M_r",                "M_r (obs)",          "{:.2f}"),
    ("M_g",                "M_g (obs)",          "{:.2f}"),
    ("color_at_rpeak",     "g-r at r-peak",      "{:.3f}"),
    ("color_10d",          "g-r at +10d",        "{:.3f}"),
    ("dt_peak_g_minus_r",  "t_g - t_r",          "{:.2f} d"),
    ("dm15_r",             "Δm15 r",             "{:.2f}"),
    ("tail_slope_r",       "tail slope r",       "{:.4f} mag/d"),
    ("fwhm_ratio_gr",      "FWHM g/r",           "{:.2f}"),
    ("color_rate",         "d(g-r)/dt",          "{:.4f} mag/d"),
]
FLAG_PARAMS = [
    ("r_is_limit",        "r FWHM is a LOWER LIMIT"),
    ("g_is_limit",        "g FWHM is a LOWER LIMIT"),
    ("kcorr_wave_extrap", "M_rest_g from wavelength EXTRAPOLATION"),
]

app = Flask(__name__)


# ---- data ---------------------------------------------------------------
def load_events():
    df = pd.read_csv(MEAS)
    return df

def load_tags():
    """ZTFID -> dict(shape, features, notes)"""
    tags = {}
    if os.path.exists(OUT):
        d = pd.read_csv(OUT, dtype=str).fillna("")
        for _, r in d.iterrows():
            tags[r["ZTFID"]] = dict(shape=r.get("shape", ""),
                                    features=r.get("features", ""),
                                    notes=r.get("notes", ""))
    return tags

def save_tag(ztfid, shape, features, notes):
    tags = load_tags()
    tags[ztfid] = dict(shape=shape, features=";".join(features), notes=notes)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ZTFID", "shape", "features", "notes"])
        for k, v in sorted(tags.items()):
            w.writerow([k, v["shape"], v["features"], v["notes"]])


# ---- plotting (MAGNITUDE space) ----------------------------------------
def _letter(b):
    n = str(b).split("::")[-1]
    for p in ("sdss", "ztf", "atlas", "ps1"):
        if n.startswith(p):
            n = n[len(p):]
    return n[:1] if n else "?"

def _flux_to_mag(f):
    f = np.asarray(f, float)
    out = np.full_like(f, np.nan)
    ok = f > 0
    out[ok] = -2.5 * np.log10(f[ok] / F0_mJy)
    return out

def _draw_panel(ax, fit, bands, title):
    any_drawn = False
    yvals = []                       # collect ONLY mean + data mags for the scale
    for b in bands:
        col = BAND_COLOR.get(_letter(b), "grey")
        try:
            tg, mu, sd = fit.predict_band(b, n=600)
        except Exception:
            continue
        m = _flux_to_mag(mu)
        m_hi = _flux_to_mag(mu - sd)     # fainter edge (can diverge -> excluded from scale)
        m_lo = _flux_to_mag(mu + sd)     # brighter edge
        good = np.isfinite(m)
        if good.any():
            ax.plot(tg[good], m[good], color=col, lw=1.5, label=str(b))
            yvals.append(m[good])                       # mean counts toward scale
            band_ok = good & np.isfinite(m_hi) & np.isfinite(m_lo)
            if band_ok.any():
                ax.fill_between(tg[band_ok], m_lo[band_ok], m_hi[band_ok],
                                color=col, alpha=0.18, linewidth=0)
            any_drawn = True
        ph, fl, fe = fit.band_data(b)
        mm = _flux_to_mag(fl)
        with np.errstate(invalid="ignore", divide="ignore"):
            merr = (2.5 / np.log(10)) * (np.asarray(fe, float) /
                                         np.asarray(fl, float))
        ok = np.isfinite(mm)
        if ok.any():
            ax.errorbar(np.asarray(ph)[ok], mm[ok], yerr=np.abs(merr)[ok],
                        fmt="o", ms=4, color=col, mfc="white", mec=col,
                        elinewidth=0.8, capsize=2, zorder=3)
            yvals.append(mm[ok])                        # data counts toward scale
            any_drawn = True

    if not any_drawn:
        ax.text(0.5, 0.5, "nothing to plot", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    else:
        v = np.concatenate(yvals)
        v = v[np.isfinite(v)]
        if v.size:
            lo, hi = float(v.min()), float(v.max())     # lo = brightest
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            ax.set_ylim(hi + pad, lo - pad)             # inverted: bright at top
    ax.set_xlabel("phase [days]")
    ax.set_ylabel("apparent magnitude (AB)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)

def make_png(ztfid, kernel, g_filter, r_filter):
    """Two-panel magnitude-space figure; cached to disk."""
    path = os.path.join(PNG_CACHE, f"{ztfid}_{kernel}.png")
    if os.path.exists(path):
        return path
    fit = reconstruct.load_fit(
        os.path.join(config.JSON_DIR, f"{ztfid}_{kernel}_constant.json"))
    allb = list(fit.obj["bands"])
    # panel A: the analysis g and r, plus any i band
    gri = [b for b in (g_filter, r_filter) if b and b in allb]
    gri += [b for b in allb if _letter(b) == "i" and b not in gri]
    if not gri:
        gri = [b for b in allb if _letter(b) in ("g", "r", "i")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    _draw_panel(axes[0], fit, gri, f"analysis bands (g/r{' + i' if any(_letter(b)=='i' for b in gri) else ''})")
    _draw_panel(axes[1], fit, allb, f"all filters ({len(allb)})")
    fig.suptitle(f"{ztfid}   kernel = {kernel}   [magnitude space]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---- template -----------------------------------------------------------
PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Morphology review</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:14px;background:#fafafa}
 .top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:8px}
 .name{font-size:20px;font-weight:700}
 .chip{background:#eee;border-radius:10px;padding:2px 9px;font-size:13px}
 .wrap{display:flex;gap:16px;align-items:flex-start}
 .plot{flex:3}
 .plot img{width:100%;border:1px solid #ddd;background:#fff}
 .side{flex:1;min-width:290px}
 table.params{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
 table.params td{border-bottom:1px solid #eee;padding:3px 6px}
 table.params td.k{color:#555;width:52%}
 table.params td.v{font-weight:600;text-align:right}
 .flag{background:#ffe9e9;color:#a00;border-radius:6px;padding:3px 7px;
       font-size:12px;display:inline-block;margin:2px 0}
 fieldset{border:1px solid #ddd;background:#fff;margin:10px 0;padding:8px}
 legend{font-weight:700;font-size:14px}
 label{display:block;font-size:14px;padding:3px 0;cursor:pointer}
 .feat label{display:inline-block;width:48%}
 .nav{margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 button,input[type=submit]{padding:7px 14px;font-size:15px;cursor:pointer}
 .save{background:#2e7d32;color:#fff;border:0;border-radius:5px}
 .done{color:#2e7d32;font-weight:700}
 input[type=text]{padding:6px;font-size:14px}
</style></head><body>

<div class="top">
  <span class="name">{{ ztfid }}</span>
  <span class="chip">{{ i+1 }} / {{ n }}</span>
  <span class="chip">{{ row['type'] }}</span>
  <span class="chip">z = {{ '%.4f'|format(row['z']) if row['z']==row['z'] else '?' }}</span>
  {% if tagged %}<span class="done">tagged ✓</span>{% endif %}
  <form method="get" action="{{ url_for('goto') }}" style="margin-left:auto">
    <input type="text" name="q" placeholder="ZTFID or number" size="16">
    <input type="submit" value="go">
  </form>
</div>

<div class="wrap">
  <div class="plot">
    <img src="{{ url_for('plot', ztfid=ztfid) }}?k={{ kernel }}">
  </div>

  <div class="side">
    <form method="post" action="{{ url_for('save') }}">
      <input type="hidden" name="ztfid" value="{{ ztfid }}">
      <input type="hidden" name="idx" value="{{ i }}">

      <fieldset><legend>Primary shape</legend>
        {% for key,lab in shapes %}
        <label><input type="radio" name="shape" value="{{ key }}"
          {% if cur.shape==key %}checked{% endif %}> {{ lab }}</label>
        {% endfor %}
      </fieldset>

      <fieldset class="feat"><legend>Features (any)</legend>
        {% for key,lab in features %}
        <label><input type="checkbox" name="features" value="{{ key }}"
          {% if key in cur_features %}checked{% endif %}> {{ lab }}</label>
        {% endfor %}
      </fieldset>

      <fieldset><legend>Notes</legend>
        <input type="text" name="notes" value="{{ cur.notes }}" style="width:98%">
      </fieldset>

      <div class="nav">
        <button type="submit" name="go" value="prev">&larr; Prev</button>
        <input class="save" type="submit" name="go" value="Save &amp; Next">
        <button type="submit" name="go" value="stay">Save</button>
        <button type="submit" name="go" value="next">Skip &rarr;</button>
      </div>
    </form>

    {% for col,lab in flags %}{% if row[col] %}
      <div class="flag">{{ lab }}</div>
    {% endif %}{% endfor %}

    <table class="params">
      {% for lab,val in params %}
      <tr><td class="k">{{ lab }}</td><td class="v">{{ val }}</td></tr>
      {% endfor %}
    </table>
  </div>
</div>
</body></html>
"""


# ---- routes -------------------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("event", i=0))

@app.route("/event/<int:i>")
def event(i):
    df = load_events()
    n = len(df)
    i = max(0, min(i, n - 1))
    row = df.iloc[i]
    tags = load_tags()
    cur = tags.get(row["ZTFID"], dict(shape="", features="", notes=""))

    params = []
    for col, lab, fmt in SHOW_PARAMS:
        if col not in df.columns:
            continue
        v = row[col]
        try:
            s = fmt.format(v) if pd.notna(v) and v != "" else "--"
        except (ValueError, TypeError):
            s = str(v) if pd.notna(v) else "--"
        params.append((lab, s))

    return render_template_string(
        PAGE, ztfid=row["ZTFID"], kernel=row["kernel"], row=row, i=i, n=n,
        shapes=SHAPES, features=FEATURES, cur=cur,
        cur_features=cur["features"].split(";") if cur["features"] else [],
        params=params, flags=FLAG_PARAMS,
        tagged=bool(cur["shape"] or cur["features"] or cur["notes"]))

@app.route("/plot/<ztfid>")
def plot(ztfid):
    from flask import send_file
    df = load_events()
    r = df[df["ZTFID"] == ztfid]
    if r.empty:
        return "unknown event", 404
    r = r.iloc[0]
    try:
        path = make_png(ztfid, r["kernel"],
                        r.get("g_filter", ""), r.get("r_filter", ""))
    except Exception as e:
        # render the error as an image so the page still works
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, f"plot failed:\n{e}", ha="center", va="center",
                fontsize=9, wrap=True); ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100)
        plt.close(fig); buf.seek(0)
        return send_file(buf, mimetype="image/png")
    return send_file(path, mimetype="image/png")

@app.route("/save", methods=["POST"])
def save():
    ztfid = request.form["ztfid"]
    idx = int(request.form["idx"])
    go = request.form.get("go", "Save & Next")
    if go != "next":                      # 'Skip' doesn't write
        save_tag(ztfid, request.form.get("shape", ""),
                 request.form.getlist("features"),
                 request.form.get("notes", ""))
    if go == "prev":
        return redirect(url_for("event", i=max(0, idx - 1)))
    if go == "stay":
        return redirect(url_for("event", i=idx))
    return redirect(url_for("event", i=idx + 1))

@app.route("/goto")
def goto():
    q = (request.args.get("q") or "").strip()
    df = load_events()
    if q.isdigit():
        return redirect(url_for("event", i=max(0, int(q) - 1)))
    hit = df.index[df["ZTFID"].astype(str).str.lower() == q.lower()]
    if len(hit):
        return redirect(url_for("event", i=int(hit[0])))
    return redirect(url_for("event", i=0))

@app.route("/summary")
def summary():
    tags = load_tags()
    df = load_events()
    lines = [f"tagged {len(tags)} / {len(df)} events", ""]
    from collections import Counter
    shp = Counter(v["shape"] for v in tags.values() if v["shape"])
    lines.append("primary shape:")
    for k, c in shp.most_common():
        lines.append(f"  {k:12s} {c}")
    feat = Counter(f for v in tags.values() for f in v["features"].split(";") if f)
    lines.append("\nfeatures:")
    for k, c in feat.most_common():
        lines.append(f"  {k:12s} {c}")
    return "<pre>" + "\n".join(lines) + "</pre>"


if __name__ == "__main__":
    print(f"morphology review -> http://localhost:{PORT}   (summary at /summary)")
    app.run(host="0.0.0.0", port=PORT, debug=False)