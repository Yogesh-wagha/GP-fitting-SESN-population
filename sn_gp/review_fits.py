"""
review_fits_app.py  --  scroll the gri fits, grouped by method, leave remarks.

Lives in sn_gp/ next to run.py / batch_fit.py.

Run:
    python3 review_fits_app.py
    open http://127.0.0.1:5002

- Six method groups across the top bar; click to switch (keeps your page).
- Within a group, events appear in the FROZEN order from selected_events.csv
  (same order in every group, so switching group lines events up).
- AIC/BIC are on each plot AND printed beside it.
- A remarks box under every fit, saved to fit_results/fit_review.csv
  keyed by (group, ZTFID). Switching group or page saves first.
- /summary shows the comparison plots from summarize.py.
"""

import os
import glob
import pandas as pd
from flask import (Flask, request, redirect, url_for,
                   send_file, render_template_string, abort)

HERE     = os.path.dirname(os.path.abspath(__file__))
SELECTED = os.path.join(RESULTS, "selected_events.csv")
METRICS  = os.path.join(RESULTS, "fit_metrics.csv")
SUMDIR   = os.path.join(RESULTS, "summary")
REVIEW   = os.path.join(RESULTS, "fit_review.csv")
PER_PAGE = 20

FIGS   = config.FIG_DIR
JSONd  = config.JSON_DIR
RESULTS = config.FIT_ROOT
REVIEW  = os.path.join(RESULTS, "fit_review.csv")     # good/bad/keep record
BEST    = os.path.join(RESULTS, "best_fit.csv")
CONFIGS = [("constant","gibbs"), ("constant","changepoint"),
           ("constant","changepoint_1"), ("constant","matern32")]

# group key matches the PNG/JSON naming: {kernel}_{mean}
GROUPS = [dict(key=f"{k}_{m}", label=f"{m} + {k}", mean=m, kernel=k) for m, k in CONFIGS]
GKEYS  = [g["key"] for g in GROUPS]
GBYKEY = {g["key"]: g for g in GROUPS}

app = Flask(__name__)


def events_order():
    return list(pd.read_csv(SELECTED, dtype=str)["ZTFID"])

def load_metrics():
    if not os.path.exists(METRICS):
        return {}
    d = pd.read_csv(METRICS)
    return {(str(r["group"]), str(r["ZTFID"])): r for _, r in d.iterrows()}

def load_review():
    """remark text keyed by (group, ZTFID)."""
    if os.path.exists(REVIEW):
        d = pd.read_csv(REVIEW, dtype=str).fillna("")
        return {(r["group"], r["ZTFID"]): r.get("remark","") for _, r in d.iterrows()}
    return {}

def load_review_verdict():
    """good/bad/keep keyed by (group, ZTFID)."""
    if os.path.exists(REVIEW):
        d = pd.read_csv(REVIEW, dtype=str).fillna("")
        return {(r["group"], r["ZTFID"]): r.get("verdict","") for _, r in d.iterrows()}
    return {}

def save_review(updates):
    rev = load_review(); rev.update(updates)
    rows = [dict(group=g, ZTFID=z, remark=t) for (g, z), t in sorted(rev.items())]
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows, columns=["group", "ZTFID", "remark"]).to_csv(REVIEW, index=False)

def load_best():
    if not os.path.exists(BEST): return {}
    d = pd.read_csv(BEST, dtype=str)
    return {r["ZTFID"]: r["best_kernel"] for _, r in d.iterrows()}

def load_types():
    b = pd.read_csv(os.path.expanduser("~/GP_SN/BTS.csv"))
    col = "ZTFID" if "ZTFID" in b.columns else b.columns[0]
    return {str(r[col]): str(r.get("type","")) for _, r in b.iterrows()}

@app.route("/")
def index():
    return redirect(url_for("view", gkey=GKEYS[0], i=0))

@app.route("/group/<gkey>/event/<int:i>")
def view(gkey, i):
    if gkey not in GKEYS:
        abort(404)
    g = GBYKEY[gkey]
    ev = events_order()
    n = len(ev)
    i = max(0, min(i, n - 1))                  # clamp to range
    z = ev[i]

    metrics, rev = load_metrics(), load_review()
    png = f"{z}_{g['kernel']}_{g['mean']}.png"
    mrow = metrics.get((gkey, z))
    aicbic = ""
    if mrow is not None and str(mrow.get("status")) == "ok":
        aicbic = (f"AIC={float(mrow['AIC']):.1f}   BIC={float(mrow['BIC']):.1f}"
                  f"   k={int(mrow['k'])}")
    item = dict(num=i + 1, zid=z, png=png,
                exists=os.path.exists(os.path.join(FIGS, png)),
                aicbic=aicbic, remark=rev.get((gkey, z), ""))
    
    best_kernel = load_best().get(z, None)
    best_gkey   = f"{best_kernel}_constant" if best_kernel else None
    ev_type     = load_types().get(z, "")
    verdict     = load_review_verdict().get((gkey, z), "")

    return render_template_string(
        TEMPLATE, item=item, gkey=gkey, idx=i, total=n, groups=GROUPS,
        prev_i=max(0, i - 1), next_i=min(n - 1, i + 1),
        best_gkey=best_gkey, ev_type=ev_type, verdict=verdict,   # <-- add these three
    )

@app.route("/plot/<fname>")
def plot(fname):
    p = os.path.join(FIGS, fname)
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="image/png")

@app.route("/summary")
def summary():
    imgs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(SUMDIR, "*.png")))
    return render_template_string(SUMMARY_TEMPLATE, imgs=imgs, groups=GROUPS)

@app.route("/summary/<fname>")
def summary_img(fname):
    p = os.path.join(SUMDIR, fname)
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="image/png")

@app.route("/goto")
def goto():
    """Search box: jump to an event by ZTFID, staying in the current group."""
    gkey = request.args.get("gkey", GKEYS[0])
    query = request.args.get("q", "").strip()
    ev = events_order()
    # exact match first, else first ZTFID that contains the query (case-insensitive)
    target = None
    if query in ev:
        target = ev.index(query)
    else:
        for j, z in enumerate(ev):
            if query.lower() in z.lower():
                target = j
                break
    if target is None:
        target = 0
    return redirect(url_for("view", gkey=gkey, i=target))

@app.route("/save", methods=["POST"])
def save():
    cur = request.form["cur_gkey"]; z = request.form["zid"]
    nav = request.form.get("nav", f"{cur}:0")
    save_review(cur, z, request.form.get("remark","").strip(),
                        request.form.get("verdict",""))
    tgt_gkey, tgt_i = nav.split(":")
    return redirect(url_for("view", gkey=tgt_gkey, i=int(tgt_i)))

def save_review(group, zid, remark, verdict):
    # load existing, update this (group,zid), rewrite with 4 columns
    rows = {}
    if os.path.exists(REVIEW):
        d = pd.read_csv(REVIEW, dtype=str).fillna("")
        for _, r in d.iterrows():
            rows[(r["group"], r["ZTFID"])] = (r.get("remark",""), r.get("verdict",""))
    rows[(group, zid)] = (remark, verdict)
    out = [dict(group=g, ZTFID=z, remark=rm, verdict=vd)
           for (g, z), (rm, vd) in sorted(rows.items())]
    os.makedirs(os.path.dirname(REVIEW), exist_ok=True)
    pd.DataFrame(out, columns=["group","ZTFID","remark","verdict"]).to_csv(REVIEW, index=False)


TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8"><title>GP fit review</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f4f4f6;color:#222;}
 .bar{position:sticky;top:0;z-index:50;background:#fff;border-bottom:1px solid #ccc;
      padding:8px 14px;display:flex;gap:5px;align-items:center;flex-wrap:wrap;
      box-shadow:0 1px 5px rgba(0,0,0,.08);}
 .bar .sp{flex:1;} .bar .info{font-size:13px;color:#555;}
 button,a.btn{cursor:pointer;padding:5px 9px;border:1px solid #bbb;background:#fafafa;
        border-radius:4px;font-size:12px;text-decoration:none;color:#222;}
 button.cur{background:#2d6cdf;color:#fff;border-color:#2d6cdf;font-weight:bold;}
 button.save{background:#1a9c4c;color:#fff;border-color:#1a9c4c;font-weight:bold;}
 input.q{padding:5px;border:1px solid #bbb;border-radius:4px;font-size:12px;width:150px;}
 .item{background:#fff;margin:14px auto;padding:12px 16px;max-width:1100px;
       border:1px solid #ddd;border-radius:6px;}
 .item h3{margin:0 0 4px;font-size:15px;font-family:monospace;}
 .item .mx{font-size:13px;color:#444;margin-bottom:8px;font-family:monospace;}
 .item img{display:block;width:100%;max-width:1040px;height:auto;border:1px solid #eee;}
 .miss{color:#b00;font-size:14px;padding:30px;border:1px dashed #b00;}
 .remark{width:100%;margin-top:10px;padding:8px;border:1px solid #bbb;border-radius:4px;font-size:14px;box-sizing:border-box;}
</style></head><body>
<form method="post" action="/save">
<input type="hidden" name="cur_gkey" value="{{gkey}}">
<input type="hidden" name="zid" value="{{item.zid}}">

<div class="bar">
  {% for g in groups %}
    <button type="submit" name="nav" value="{{g.key}}:{{idx}}"
            class="{{ 'cur' if g.key==gkey else '' }}">
      {{g.label}}{% if g.key==best_gkey %} ★best{% endif %}
    </button>
  {% endfor %}
  <span class="sp"></span>
  <a class="btn" href="/summary">📊 summary</a>
</div>

<div class="bar">
  <span class="info">event {{item.num}} / {{total}}</span>
  <button type="submit" name="nav" value="{{gkey}}:{{prev_i}}">◀ Back</button>
  <button type="submit" name="nav" value="{{gkey}}:{{next_i}}">Next ▶</button>
  <button type="submit" name="nav" value="{{gkey}}:{{idx}}" class="save">💾 Save</button>
  <span class="sp"></span>
  <!-- search is its own GET form; it does NOT save the current remark -->
  <span>
    <input class="q" form="searchform" name="q" placeholder="ZTFID…">
    <button form="searchform" type="submit">Go</button>
  </span>
</div>

<div class="item">
  <input type="hidden" name="zid" value="{{item.zid}}">
  <h3>#{{item.num}} · {{item.zid}} · <span style="color:#2d6cdf">{{ev_type}}</span>
      {% if gkey==best_gkey %}<span style="color:#1a9c4c"> — BEST FIT (lowest BIC)</span>{% endif %}
  </h3>
  <div class="mx">{{item.aicbic}}</div>
  {% if item.exists %}
    <img src="/plot/{{item.png}}" alt="{{item.zid}}">
  {% else %}
    <div class="miss">no fit (failed or not run): {{item.png}}</div>
  {% endif %}
  <input class="remark" type="text" name="remark"
         placeholder="remarks…" value="{{item.remark}}">

  <!-- >>> SNIPPET 2 GOES HERE (right after the remark input, still inside the form) <<< -->
  <div style="margin-top:8px">
    {% for opt in ["good","bad","keep"] %}
      <label style="margin-right:14px">
        <input type="radio" name="verdict" value="{{opt}}"
               {{ 'checked' if verdict==opt else '' }}> {{opt}}
      </label>
    {% endfor %}
  </div>
  <!-- >>> END SNIPPET 2 <<< -->
</div>
</form>



<!-- separate form for search so it can be a GET without touching /save -->
<form id="searchform" method="get" action="/goto">
  <input type="hidden" name="gkey" value="{{gkey}}">
</form>
</body></html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)