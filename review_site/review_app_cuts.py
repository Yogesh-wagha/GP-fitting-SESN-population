"""
review_cuts_app.py  --  manually mark the phase range to keep for each light curve.

Run:
    pip install flask
    python review_cuts_app.py
    open http://127.0.0.1:5001

- Shows the PNGs already in review_site/plots/ (does NOT generate any plots).
- Per object: OK / CHECK radio (default OK), a remarks box, and two boxes
  'lower cut' and 'upper cut' (phase in days from peak) for the left-most and
  right-most data point you want to keep.
- Saves to review_site/manual_cuts.csv  (columns: ZTFID, flag, remarks,
  lower_cut, upper_cut). The old review_flags.csv is only READ (to pre-fill
  remarks); it is never modified.
- Clicking any page button or Save persists the current page first.
"""

import os
import glob
import pandas as pd
from flask import (Flask, request, redirect, url_for,
                   send_file, render_template_string, abort)

# ---- paths ----
BASE     = "/home/yogesh1729/myWork/air_phd/gaussian_process_BTS"
SITE_DIR = os.path.join(BASE, "review_site")
PLOT_DIR = os.path.join(SITE_DIR, "plots")
OLD_CSV  = os.path.join(SITE_DIR, "review_flags.csv")   # previous review (READ ONLY)
CUTS_CSV = os.path.join(SITE_DIR, "manual_cuts.csv")    # new output
PER_PAGE = 50

app = Flask(__name__)

# targets = every PNG in the plots folder (no plotting done here)
TARGETS = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(PLOT_DIR, "*.png")))]


# ---------- CSV helpers ----------
def load_old_remarks():
    """Previous remarks, used only to seed objects not yet in the new CSV."""
    if os.path.exists(OLD_CSV):
        d = pd.read_csv(OLD_CSV, dtype=str).fillna("")
        return {r["ZTFID"]: r.get("remarks", "") for _, r in d.iterrows()}
    return {}

def load_cuts():
    if os.path.exists(CUTS_CSV):
        d = pd.read_csv(CUTS_CSV, dtype=str).fillna("")
        return {r["ZTFID"]: dict(flag=r.get("flag", "ok"),
                                 remarks=r.get("remarks", ""),
                                 lower_cut=r.get("lower_cut", ""),
                                 upper_cut=r.get("upper_cut", ""))
                for _, r in d.iterrows()}
    return {}

def resolve(target, cuts, old):
    """Values to show: the new CSV if present, else seed (ok + old remark)."""
    if target in cuts:
        return cuts[target]
    return dict(flag="ok", remarks=old.get(target, ""), lower_cut="", upper_cut="")

def save_cuts(updates):
    """Upsert the posted objects into manual_cuts.csv (every posted object)."""
    cuts = load_cuts()
    cuts.update(updates)
    rows = [dict(ZTFID=z, **v) for z, v in sorted(cuts.items())]
    os.makedirs(SITE_DIR, exist_ok=True)
    pd.DataFrame(rows, columns=["ZTFID", "flag", "remarks",
                                "lower_cut", "upper_cut"]).to_csv(CUTS_CSV, index=False)


# ---------- routes ----------
@app.route("/")
def index():
    return redirect(url_for("page", n=1))

@app.route("/page/<int:n>")
def page(n):
    n_pages = max(1, (len(TARGETS) + PER_PAGE - 1) // PER_PAGE)
    n = min(max(n, 1), n_pages)
    start = (n - 1) * PER_PAGE
    chunk = TARGETS[start:start + PER_PAGE]
    cuts, old = load_cuts(), load_old_remarks()
    items = [dict(zid=z, num=start + i + 1, **resolve(z, cuts, old))
             for i, z in enumerate(chunk)]
    n_check = sum(1 for v in cuts.values() if v["flag"] == "check")
    return render_template_string(TEMPLATE, items=items, page=n,
                                  n_pages=n_pages, total=len(TARGETS), n_check=n_check)

@app.route("/plot/<target>.png")
def plot(target):
    if target not in TARGETS:
        abort(404)
    return send_file(os.path.join(PLOT_DIR, f"{target}.png"), mimetype="image/png")

@app.route("/save", methods=["POST"])
def save():
    goto = int(request.form.get("goto", request.form.get("page", 1)))
    updates = {}
    for z in request.form.getlist("zid"):
        updates[z] = dict(flag=request.form.get(f"flag_{z}", "ok"),
                          remarks=request.form.get(f"remark_{z}", "").strip(),
                          lower_cut=request.form.get(f"lower_{z}", "").strip(),
                          upper_cut=request.form.get(f"upper_{z}", "").strip())
    save_cuts(updates)
    return redirect(url_for("page", n=goto))


# ---------- template ----------
TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8"><title>LC cuts</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f4f4f6;color:#222;}
 .bar{position:sticky;top:0;z-index:50;background:#fff;border-bottom:1px solid #ccc;
      padding:8px 14px;display:flex;gap:5px;align-items:center;flex-wrap:wrap;
      box-shadow:0 1px 5px rgba(0,0,0,.08);}
 .bar .sp{flex:1;} .bar .info{font-size:13px;color:#555;margin-right:8px;}
 button{cursor:pointer;padding:5px 10px;border:1px solid #bbb;background:#fafafa;border-radius:4px;font-size:13px;}
 button.cur{background:#2d6cdf;color:#fff;border-color:#2d6cdf;font-weight:bold;}
 button.save{background:#1a9c4c;color:#fff;border-color:#1a9c4c;font-weight:bold;}
 .item{background:#fff;margin:14px auto;padding:10px 14px;max-width:1000px;border:1px solid #ddd;border-radius:6px;}
 .item.flagged{border-left:6px solid #e23b2e;}
 .item h3{margin:0 0 6px;font-size:14px;font-family:monospace;}
 .item img{display:block;width:100%;max-width:940px;height:auto;border:1px solid #eee;}
 .controls{margin-top:8px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;}
 .controls label{font-size:14px;}
 .cut{width:90px;padding:5px;border:1px solid #bbb;border-radius:4px;font-size:14px;}
 .remark{flex:1;min-width:220px;padding:6px;border:1px solid #bbb;border-radius:4px;font-size:14px;}
</style></head><body>
<form method="post" action="/save">
<input type="hidden" name="page" value="{{page}}">

<div class="bar">
  <span class="info"><b>LC cuts</b> — {{total}} objects · page {{page}}/{{n_pages}} · <b>{{n_check}}</b> CHECK</span>
  <button type="submit" name="goto" value="{{ page-1 if page>1 else 1 }}">◀ Prev</button>
  {% for p in range(1, n_pages+1) %}
    <button type="submit" name="goto" value="{{p}}" class="{{ 'cur' if p==page else '' }}">{{p}}</button>
  {% endfor %}
  <button type="submit" name="goto" value="{{ page+1 if page<n_pages else n_pages }}">Next ▶</button>
  <span class="sp"></span>
  <button type="submit" name="goto" value="{{page}}" class="save">💾 Save</button>
</div>

{% for it in items %}
<div class="item {{ 'flagged' if it.flag=='check' else '' }}">
  <input type="hidden" name="zid" value="{{it.zid}}">
  <h3>#{{it.num}} &nbsp;·&nbsp; {{it.zid}}</h3>
  <img loading="lazy" src="/plot/{{it.zid}}.png" alt="{{it.zid}}">
  <div class="controls">
    <span>
      <label><input type="radio" name="flag_{{it.zid}}" value="ok"    {{ 'checked' if it.flag!='check' }}> OK</label>
      <label><input type="radio" name="flag_{{it.zid}}" value="check" {{ 'checked' if it.flag=='check' }}> CHECK</label>
    </span>
    <label>lower cut <input class="cut" type="number" step="any" name="lower_{{it.zid}}" value="{{it.lower_cut}}"></label>
    <label>upper cut <input class="cut" type="number" step="any" name="upper_{{it.zid}}" value="{{it.upper_cut}}"></label>
    <input class="remark" type="text" name="remark_{{it.zid}}" placeholder="remarks…" value="{{it.remarks}}">
  </div>
</div>
{% endfor %}

<div class="bar">
  <button type="submit" name="goto" value="{{ page-1 if page>1 else 1 }}">◀ Prev</button>
  <button type="submit" name="goto" value="{{ page+1 if page<n_pages else n_pages }}">Next ▶</button>
  <span class="sp"></span>
  <button type="submit" name="goto" value="{{page}}" class="save">💾 Save</button>
</div>
</form></body></html>
"""

if __name__ == "__main__":
    print(f"{len(TARGETS)} plots · cuts -> {CUTS_CSV}")
    app.run(host="127.0.0.1", port=5001, debug=False)