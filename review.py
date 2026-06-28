"""
review_app.py  --  local web tool to eyeball every light curve and flag issues.

Run:
    pip install flask
    python review_app.py
    open http://127.0.0.1:5000  in your browser

- 50 plots per page, one per row.
- Pagination + Save buttons fixed at the top (and bottom).
- Per object: radio (none / ok / check) + a remarks text box.
- Clicking a page button or Save writes the current page to a CSV, so you
  never lose work when navigating.
- Only objects with a flag or a remark are written to the CSV.

Output CSV: <BASE>/review_site/review_flags.csv   (columns: ZTFID, flag, remarks)
Plot cache: <BASE>/review_site/plots/*.png        (generated lazily on first view)
"""

import os
import glob
import threading
import pandas as pd
from flask import (Flask, request, redirect, url_for,
                   send_file, render_template_string, abort)

import plot_detections as plotter     # reuses CSV_DIR + the plotting function

# ---- paths ----
BASE       = "/users/ariywagh/GP_SN"
CSV_DIR    = os.path.join(BASE, "BTS_csv")
SITE_DIR   = os.path.join(BASE, "review_site")
PLOT_CACHE = os.path.join(SITE_DIR, "plots")
FLAG_CSV   = os.path.join(SITE_DIR, "review_flags.csv")
PER_PAGE   = 50

os.makedirs(PLOT_CACHE, exist_ok=True)

app = Flask(__name__)
_plot_lock = threading.Lock()         # matplotlib is not thread-safe

# all targets = every CSV in BTS_csv, alphabetical
TARGETS = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))]


# ---------- flag CSV helpers ----------
def load_flags():
    if os.path.exists(FLAG_CSV):
        d = pd.read_csv(FLAG_CSV, dtype=str).fillna("")
        return {row["ZTFID"]: (row.get("flag", ""), row.get("remarks", ""))
                for _, row in d.iterrows()}
    return {}

def save_flags(updates):
    """updates: {ZTFID: (flag, remark)}. Merge into existing; drop empty rows."""
    flags = load_flags()
    for zid, (fl, rm) in updates.items():
        if fl or rm:
            flags[zid] = (fl, rm)
        else:
            flags.pop(zid, None)          # cleared -> remove from CSV
    rows = [{"ZTFID": z, "flag": f, "remarks": r}
            for z, (f, r) in sorted(flags.items())]
    os.makedirs(SITE_DIR, exist_ok=True)
    pd.DataFrame(rows, columns=["ZTFID", "flag", "remarks"]).to_csv(FLAG_CSV, index=False)


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
    flags = load_flags()
    items = [{"zid": z, "num": start + i + 1,
              "flag": flags.get(z, ("", ""))[0],
              "remark": flags.get(z, ("", ""))[1]}
             for i, z in enumerate(chunk)]
    n_flagged = sum(1 for v in flags.values() if v[0] == "check")
    return render_template_string(TEMPLATE, items=items, page=n,
                                  n_pages=n_pages, total=len(TARGETS),
                                  n_flagged=n_flagged)

@app.route("/plot/<target>.png")
def plot(target):
    if target not in TARGETS:
        abort(404)
    png = os.path.join(PLOT_CACHE, f"{target}.png")
    if not os.path.exists(png):
        with _plot_lock:
            if not os.path.exists(png):       # re-check inside the lock
                plotter.plot_detections(target, savepath=png)
    return send_file(png, mimetype="image/png")

@app.route("/save", methods=["POST"])
def save():
    goto = int(request.form.get("goto", request.form.get("page", 1)))
    updates = {}
    for z in request.form.getlist("zid"):     # only the objects on this page
        fl = request.form.get(f"flag_{z}", "")
        rm = request.form.get(f"remark_{z}", "").strip()
        updates[z] = (fl, rm)
    save_flags(updates)
    return redirect(url_for("page", n=goto))


# ---------- template ----------
TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8"><title>LC review</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f4f4f6;color:#222;}
 .bar{position:sticky;top:0;z-index:50;background:#fff;border-bottom:1px solid #ccc;
      padding:8px 14px;display:flex;gap:5px;align-items:center;flex-wrap:wrap;
      box-shadow:0 1px 5px rgba(0,0,0,.08);}
 .bar .sp{flex:1;}
 .bar .info{font-size:13px;color:#555;margin-right:8px;}
 button{cursor:pointer;padding:5px 10px;border:1px solid #bbb;background:#fafafa;
        border-radius:4px;font-size:13px;}
 button.cur{background:#2d6cdf;color:#fff;border-color:#2d6cdf;font-weight:bold;}
 button.save{background:#1a9c4c;color:#fff;border-color:#1a9c4c;font-weight:bold;}
 .item{background:#fff;margin:14px auto;padding:10px 14px;max-width:1000px;
       border:1px solid #ddd;border-radius:6px;}
 .item.flagged{border-left:6px solid #e23b2e;}
 .item h3{margin:0 0 6px;font-size:14px;font-family:monospace;}
 .item img{display:block;width:100%;max-width:940px;height:auto;border:1px solid #eee;}
 .controls{margin-top:8px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;}
 .controls label{margin-right:6px;font-size:14px;}
 .remark{flex:1;min-width:260px;padding:6px;border:1px solid #bbb;border-radius:4px;font-size:14px;}
</style></head><body>
<form method="post" action="/save">
<input type="hidden" name="page" value="{{page}}">

<div class="bar">
  <span class="info"><b>LC review</b> — {{total}} objects · page {{page}}/{{n_pages}}
        · <b>{{n_flagged}}</b> flagged "check"</span>
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
      <label><input type="radio" name="flag_{{it.zid}}" value=""      {{ 'checked' if it.flag=='' }}> none</label>
      <label><input type="radio" name="flag_{{it.zid}}" value="ok"    {{ 'checked' if it.flag=='ok' }}> ok</label>
      <label><input type="radio" name="flag_{{it.zid}}" value="check" {{ 'checked' if it.flag=='check' }}> check</label>
    </span>
    <input class="remark" type="text" name="remark_{{it.zid}}"
           placeholder="remarks…" value="{{it.remark}}">
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
    print(f"{len(TARGETS)} objects · CSV -> {FLAG_CSV}")
    app.run(host="127.0.0.1", port=5000, debug=False)