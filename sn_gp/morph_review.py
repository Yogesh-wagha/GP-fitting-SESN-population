"""
morph_review.py  --  web app for visually classifying light-curve MORPHOLOGY.

Separate from review_fits.py (which judges fit quality). This one is about the
SHAPE of the light curve.

  * two panels, g/r/i only: LEFT magnitude space (y inverted), RIGHT flux space
  * magnitude panel draws a reference line at (r-band peak mag + 1.5)
  * auto-detects whether real data exist below that line on the FADING side;
    you can override the flag
  * "very bad" flag marks an event for exclusion from all population plots
  * stats sit BELOW the plots; controls on the right

    python morph_review.py            # http://localhost:5004
    ssh -L 5004:localhost:5004 ariywagh@prospero...

Writes fit/morphology_review.csv
  (ZTFID, shape, features, below15, very_bad, notes)
"""

import os
import io
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, url_for, render_template_string, send_file

import config
import reconstruct
import measure_population as mp

PORT = 5002
MEAS = os.path.join(config.FIT_ROOT, "population_measurements.csv")
OUT = os.path.join(config.FIT_ROOT, "morphology_review.csv")
PNG_CACHE = os.path.join(config.FIT_ROOT, "morph_png")
os.makedirs(PNG_CACHE, exist_ok=True)

F0_mJy = 3631e3
REF_DROP_MAG = 1.5          # reference line: peak + this many mag (fainter)

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

BAND_COLOR = {"g": "#1b9e2f", "r": "#d62728", "i": "#8c564b"}
GRI = ("g", "r", "i")

SHOW_PARAMS = [
    ("type", "Type", "{}"), ("z", "Redshift", "{:.4f}"),
    ("kernel", "Kernel", "{}"),
    ("g_filter", "g filter", "{}"), ("r_filter", "r filter", "{}"),
    ("fwhm_r", "FWHM r (rest)", "{:.1f} d"),
    ("fwhm_g", "FWHM g (rest)", "{:.1f} d"),
    ("rise_r", "rise r (rest)", "{:.1f} d"),
    ("fade_r", "fade r (rest)", "{:.1f} d"),
    ("rise_frac_r", "rise / FWHM (r)", "{:.2f}"),
    ("M_rest_g", "M rest-g (K)", "{:.2f}"),
    ("M_rest_g_mw", "M rest-g (K+MW)", "{:.2f}"),
    ("M_r", "M_r (obs)", "{:.2f}"), ("M_g", "M_g (obs)", "{:.2f}"),
    ("color_at_rpeak", "g-r at r-peak", "{:.3f}"),
    ("color_10d", "g-r at +10d", "{:.3f}"),
    ("dt_peak_g_minus_r", "dt_gr", "{:.2f} d"),
    ("dm15_r", "dm15 r", "{:.2f}"),
    ("tail_slope_r", "tail slope r", "{:.4f} mag/d"),
    ("fwhm_ratio_gr", "FWHM g/r", "{:.2f}"),
    ("color_rate", "d(g-r)/dt", "{:.4f} mag/d"),
]
FLAG_PARAMS = [
    ("r_is_limit", "r FWHM is a LOWER LIMIT"),
    ("g_is_limit", "g FWHM is a LOWER LIMIT"),
    ("kcorr_wave_extrap", "M_rest_g from wavelength EXTRAPOLATION"),
]

COLS = ["ZTFID", "shape", "features", "below15", "very_bad", "notes"]
app = Flask(__name__)


# ---- data ---------------------------------------------------------------
def load_events():
    return pd.read_csv(MEAS)

def load_tags():
    tags = {}
    if os.path.exists(OUT):
        d = pd.read_csv(OUT, dtype=str).fillna("")
        for _, r in d.iterrows():
            tags[r["ZTFID"]] = {c: r.get(c, "") for c in COLS if c != "ZTFID"}
    return tags

def save_tag(ztfid, rec):
    tags = load_tags()
    tags[ztfid] = rec
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS)
        for k, v in sorted(tags.items()):
            w.writerow([k] + [v.get(c, "") for c in COLS[1:]])


# ---- helpers ------------------------------------------------------------
def _letter(b):
    n = str(b).split("::")[-1]
    for p in ("sdss", "ztf", "atlas", "ps1"):
        if n.startswith(p):
            n = n[len(p):]
    return n[:1].lower() if n else "?"

def _flux_to_mag(f):
    f = np.asarray(f, float)
    out = np.full_like(f, np.nan)
    ok = f > 0
    out[ok] = -2.5 * np.log10(f[ok] / F0_mJy)
    return out

def _gri_bands(fit, g_filter, r_filter):
    allb = list(fit.obj["bands"])
    sel = [b for b in (g_filter, r_filter) if b and b in allb]
    sel += [b for b in allb if _letter(b) == "i" and b not in sel]
    if not sel:
        sel = [b for b in allb if _letter(b) in GRI]
    return sel

def _r_peak_and_coverage(fit, rband):
    """Return (peak_mag, peak_phase, peak_flux, has_data_below_ref) for r."""
    if rband is None:
        return np.nan, np.nan, np.nan, None
    ph, fl, _ = fit.band_data(rband)
    if ph.size == 0:
        return np.nan, np.nan, np.nan, None
    tg, mu, _ = fit.predict_band(rband, n=1500)
    m = _flux_to_mag(mu)
    if not np.any(np.isfinite(m)):
        return np.nan, np.nan, np.nan, None
    ipk = int(np.nanargmin(m))
    mpk, tpk, fpk = float(m[ipk]), float(tg[ipk]), float(mu[ipk])
    ref = mpk + REF_DROP_MAG
    dm = _flux_to_mag(fl)
    faded = (ph > tpk) & np.isfinite(dm) & (dm >= ref)
    return mpk, tpk, fpk, bool(np.any(faded))


# ---- plotting -----------------------------------------------------------
def _panel_mag(ax, fit, bands, ref_mag, tpk):
    yvals = []
    for b in bands:
        col = BAND_COLOR.get(_letter(b), "grey")
        tg, mu, sd = fit.predict_band(b, n=600)
        m = _flux_to_mag(mu)
        m_lo, m_hi = _flux_to_mag(mu + sd), _flux_to_mag(mu - sd)
        good = np.isfinite(m)
        if good.any():
            ax.plot(tg[good], m[good], color=col, lw=1.6, label=str(b))
            yvals.append(m[good])
            bo = good & np.isfinite(m_lo) & np.isfinite(m_hi)
            if bo.any():
                ax.fill_between(tg[bo], m_lo[bo], m_hi[bo], color=col,
                                alpha=0.18, lw=0)
        ph, fl, fe = fit.band_data(b)
        mm = _flux_to_mag(fl)
        with np.errstate(invalid="ignore", divide="ignore"):
            me = (2.5 / np.log(10)) * (np.asarray(fe, float) / np.asarray(fl, float))
        ok = np.isfinite(mm)
        if ok.any():
            ax.errorbar(np.asarray(ph)[ok], mm[ok], yerr=np.abs(me)[ok], fmt="o",
                        ms=4, color=col, mfc="white", mec=col, elinewidth=0.8,
                        capsize=2, zorder=3)
            yvals.append(mm[ok])
    if np.isfinite(ref_mag):
        ax.axhline(ref_mag, color="k", ls="--", lw=1.4, alpha=0.8)
        ax.text(0.01, ref_mag, f" r-peak +{REF_DROP_MAG} mag", va="bottom",
                ha="left", fontsize=9, transform=ax.get_yaxis_transform())
        yvals.append(np.array([ref_mag]))
    if np.isfinite(tpk):
        ax.axvline(tpk, color="k", ls=":", lw=1.0, alpha=0.5)
    if yvals:
        v = np.concatenate(yvals); v = v[np.isfinite(v)]
        lo, hi = float(v.min()), float(v.max())
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(hi + pad, lo - pad)
    ax.set_xlabel("phase [days]"); ax.set_ylabel("apparent magnitude (AB)")
    ax.set_title("magnitude space (g/r/i)", fontsize=12)
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.2)

def _panel_flux(ax, fit, bands, ref_flux=np.nan, tpk=np.nan):
    for b in bands:
        col = BAND_COLOR.get(_letter(b), "grey")
        tg, mu, sd = fit.predict_band(b, n=600)
        ax.plot(tg, mu, color=col, lw=1.6, label=str(b))
        ax.fill_between(tg, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ph, fl, fe = fit.band_data(b)
        ax.errorbar(ph, fl, yerr=fe, fmt="o", ms=4, color=col, mfc="white",
                    mec=col, elinewidth=0.8, capsize=2, zorder=3)
    if np.isfinite(ref_flux):
        ax.axhline(ref_flux, color="k", ls="--", lw=1.4, alpha=0.8)
        ax.text(0.01, ref_flux, f" r-peak -{REF_DROP_MAG} mag", va="bottom",
                ha="left", fontsize=9, transform=ax.get_yaxis_transform())
    if np.isfinite(tpk):
        ax.axvline(tpk, color="k", ls=":", lw=1.0, alpha=0.5)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("phase [days]"); ax.set_ylabel("flux [mJy]")
    ax.set_title("flux space (g/r/i)", fontsize=12)
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.2)

def make_png(ztfid, kernel, g_filter, r_filter):
    path = os.path.join(PNG_CACHE, f"{ztfid}_{kernel}.png")
    if os.path.exists(path):
        return path
    fit = reconstruct.load_fit(
        os.path.join(config.JSON_DIR, f"{ztfid}_{kernel}_constant.json"))
    bands = _gri_bands(fit, g_filter, r_filter)
    rb = r_filter if (r_filter and r_filter in fit.obj["bands"]) else \
         next((b for b in bands if _letter(b) == "r"), None)
    mpk, tpk, fpk, _ = _r_peak_and_coverage(fit, rb)
    ref = mpk + REF_DROP_MAG if np.isfinite(mpk) else np.nan
    ref_flux = fpk * 10 ** (-REF_DROP_MAG / 2.5) if np.isfinite(fpk) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    _panel_mag(axes[0], fit, bands, ref, tpk)
    _panel_flux(axes[1], fit, bands, ref_flux, tpk)
    fig.suptitle(f"{ztfid}   kernel = {kernel}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


# ---- template -----------------------------------------------------------
PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Morphology review</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:14px;background:#fafafa}
 .top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:8px}
 .name{font-size:21px;font-weight:700}
 .chip{background:#eee;border-radius:10px;padding:2px 9px;font-size:13px}
 .bad{background:#c62828;color:#fff}
 .wrap{display:flex;gap:16px;align-items:flex-start}
 .left{flex:3;min-width:0}
 .left img{width:100%;border:1px solid #ddd;background:#fff}
 .side{flex:1;min-width:300px}
 .stats{margin-top:12px;background:#fff;border:1px solid #ddd;padding:10px}
 .stats h3{margin:0 0 8px 0;font-size:14px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:2px 16px}
 .kv{display:flex;justify-content:space-between;border-bottom:1px solid #f0f0f0;
     padding:2px 0;font-size:13px}
 .kv .k{color:#555} .kv .v{font-weight:600}
 .flag{background:#ffe9e9;color:#a00;border-radius:6px;padding:3px 7px;
       font-size:12px;display:inline-block;margin:2px 4px 2px 0}
 fieldset{border:1px solid #ddd;background:#fff;margin:10px 0;padding:8px}
 legend{font-weight:700;font-size:14px}
 label{display:block;font-size:14px;padding:3px 0;cursor:pointer}
 .feat label{display:inline-block;width:48%}
 .inline label{display:inline-block;margin-right:14px}
 .auto{font-size:12px;color:#666;margin-left:4px}
 .nav{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
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
  {% if tagged %}<span class="done">tagged &#10003;</span>{% endif %}
  {% if cur.very_bad == 'yes' %}<span class="chip bad">VERY BAD - excluded</span>{% endif %}
  <form method="get" action="{{ url_for('goto') }}" style="margin-left:auto">
    <input type="text" name="q" placeholder="ZTFID or number" size="16">
    <input type="submit" value="go">
  </form>
</div>

<div class="wrap">
  <div class="left">
    <img src="{{ url_for('plot', ztfid=ztfid) }}?k={{ kernel }}">

    <div class="stats">
      <h3>Measured parameters</h3>
      {% for col,lab in flags %}{% if row[col] %}
        <div class="flag">{{ lab }}</div>
      {% endif %}{% endfor %}
      <div class="grid">
        {% for lab,val in params %}
        <div class="kv"><span class="k">{{ lab }}</span><span class="v">{{ val }}</span></div>
        {% endfor %}
      </div>
    </div>
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

      <fieldset class="inline">
        <legend>Data below r-peak +{{ ref_drop }} mag (fading side)?</legend>
        <label><input type="radio" name="below15" value="yes"
          {% if below15=='yes' %}checked{% endif %}> yes</label>
        <label><input type="radio" name="below15" value="no"
          {% if below15=='no' %}checked{% endif %}> no</label>
        <span class="auto">(auto: {{ auto_below }})</span>
      </fieldset>

      <fieldset class="inline"><legend>Very bad event? (excluded from all plots)</legend>
        <label><input type="radio" name="very_bad" value="no"
          {% if cur.very_bad != 'yes' %}checked{% endif %}> no</label>
        <label><input type="radio" name="very_bad" value="yes"
          {% if cur.very_bad == 'yes' %}checked{% endif %}> yes</label>
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
    cur = tags.get(row["ZTFID"],
                   dict(shape="", features="", below15="", very_bad="", notes=""))

    # auto coverage check (cheap: reuses the cached fit if already loaded)
    auto_below = "?"
    try:
        fit = reconstruct.load_fit(os.path.join(
            config.JSON_DIR, f"{row['ZTFID']}_{row['kernel']}_constant.json"))
        rb = row.get("r_filter", "") or None
        if rb not in fit.obj["bands"]:
            rb = next((b for b in fit.obj["bands"] if _letter(b) == "r"), None)
            _, _, _, has = _r_peak_and_coverage(fit, rb)
        auto_below = "yes" if has else ("no" if has is False else "?")
    except Exception:
        pass
    below15 = cur.get("below15") or auto_below      # saved value wins

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
        cur_features=(cur.get("features") or "").split(";"),
        params=params, flags=FLAG_PARAMS, ref_drop=REF_DROP_MAG,
        below15=below15, auto_below=auto_below,
        tagged=bool(cur.get("shape") or cur.get("features") or cur.get("notes")))

@app.route("/plot/<ztfid>")
def plot(ztfid):
    df = load_events()
    r = df[df["ZTFID"] == ztfid]
    if r.empty:
        return "unknown event", 404
    r = r.iloc[0]
    try:
        path = make_png(ztfid, r["kernel"], r.get("g_filter", ""), r.get("r_filter", ""))
    except Exception as e:
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
    if go != "next":
        save_tag(ztfid, dict(
            shape=request.form.get("shape", ""),
            features=";".join(request.form.getlist("features")),
            below15=request.form.get("below15", ""),
            very_bad=request.form.get("very_bad", "no"),
            notes=request.form.get("notes", "")))
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
    hit = df.index[df["ZTFID"].astype(str).str.lower().str.contains(q.lower())]
    return redirect(url_for("event", i=int(hit[0]) if len(hit) else 0))

@app.route("/summary")
def summary():
    from collections import Counter
    tags = load_tags(); df = load_events()
    lines = [f"tagged {len(tags)} / {len(df)} events", ""]
    for field, title in (("shape", "primary shape"), ("below15", "data below +1.5 mag"),
                         ("very_bad", "very bad (excluded)")):
        c = Counter(v.get(field, "") for v in tags.values() if v.get(field))
        lines.append(f"{title}:")
        for k, n in c.most_common():
            lines.append(f"  {k:12s} {n}")
        lines.append("")
    feat = Counter(f for v in tags.values()
                   for f in (v.get("features") or "").split(";") if f)
    lines.append("features:")
    for k, n in feat.most_common():
        lines.append(f"  {k:12s} {n}")
    bad = [k for k, v in tags.items() if v.get("very_bad") == "yes"]
    if bad:
        lines += ["", f"very-bad events ({len(bad)}):"] + [f"  {z}" for z in sorted(bad)]
    return "<pre>" + "\n".join(lines) + "</pre>"


if __name__ == "__main__":
    print(f"morphology review -> http://localhost:{PORT}   (summary at /summary)")
    app.run(host="0.0.0.0", port=PORT, debug=False)