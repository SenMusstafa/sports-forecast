"""Self-contained HTML report for upcoming fixtures."""
from __future__ import annotations

import html

import pandas as pd

from .predict import ForecastResult

CSS = """
:root { --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --card:#fff; --line:#e7e5e4;
        --h:#2563eb; --d:#a8a29e; --a:#dc2626; --pos:#15803d; --neg:#b91c1c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0c0a09; --fg:#f5f5f4; --muted:#a8a29e; --card:#1c1917; --line:#292524;
          --h:#60a5fa; --d:#78716c; --a:#f87171; --pos:#4ade80; --neg:#f87171; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
main { max-width:960px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:15px; margin:28px 0 8px; color:var(--muted); }
.sub, .note { color:var(--muted); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:10px 0; }
.teams { display:flex; justify-content:space-between; gap:8px; font-weight:600; }
.when { color:var(--muted); font-weight:400; font-size:13px; white-space:nowrap; }
.bar { display:flex; height:10px; border-radius:5px; overflow:hidden; margin:8px 0 4px; }
.bar span:nth-child(1) { background:var(--h); } .bar span:nth-child(2) { background:var(--d); }
.bar span:nth-child(3) { background:var(--a); }
table { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:3px 6px; } th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:500; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.meta { display:flex; flex-wrap:wrap; gap:4px 16px; margin-top:8px; font-size:13px; color:var(--muted); }
.warn { color:var(--neg); font-size:13px; margin-top:6px; }
"""


def _pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def _odds(p: float) -> str:
    return "–" if pd.isna(p) or p <= 0 else f"{1 / p:.2f}"


def _diff(x: float) -> str:
    if pd.isna(x):
        return "–"
    cls = "pos" if x >= 0.03 else "neg" if x <= -0.03 else ""
    return f'<span class="{cls}">{x:+.1%}</span>'


def _card(r: pd.Series) -> str:
    has_mkt = "mkt_H" in r and pd.notna(r.get("mkt_H"))
    ref = [r[f"mkt_{o}"] for o in "HDA"] if has_mkt else [r[f"model_{o}"] for o in "HDA"]
    bar = "".join(f'<span style="width:{p * 100:.1f}%"></span>' for p in ref)
    rows = [("Market (de-vigged)", [r.get(f"mkt_{o}") for o in "HDA"], _pct)] if has_mkt else []
    rows += [("Model", [r[f"model_{o}"] for o in "HDA"], _pct),
             ("Model fair odds", [r[f"model_{o}"] for o in "HDA"], _odds)]
    body = "".join(f"<tr><td>{label}</td>" + "".join(f"<td>{fmt(v)}</td>" for v in vals) + "</tr>"
                   for label, vals, fmt in rows)
    if has_mkt:
        body += "<tr><td>Model − market</td>" + "".join(f"<td>{_diff(r[f'diff_{o}'])}</td>" for o in "HDA") + "</tr>"
    if "max_h" in r and pd.notna(r.get("max_h")):
        body += "<tr><td>Best price</td>" + "".join(f"<td>{r[f'max_{s}']:.2f}</td>" for s in "hda") + "</tr>"
    over = f"Over 2.5: {_pct(r['over25'])}"
    if pd.notna(r.get("mkt_over25", float("nan"))):
        over += f" (market {_pct(r['mkt_over25'])})"
    warn = (f'<div class="warn">No history for {html.escape(r["new_team"])}: rated as a weak newcomer</div>'
            if r["new_team"] else "")
    return f"""
<div class="card">
  <div class="teams"><span>{html.escape(r['home'])} – {html.escape(r['away'])}</span>
    <span class="when">{r['date']:%a %d %b} {html.escape(str(r['time']))}</span></div>
  <div class="bar">{bar}</div>
  <table><tr><th></th><th>Home</th><th>Draw</th><th>Away</th></tr>{body}</table>
  <div class="meta"><span>xG {r['xg_home']:.2f} – {r['xg_away']:.2f}</span><span>{over}</span>
    <span>BTTS: {_pct(r['btts'])}</span><span>Likely scores: {html.escape(r['top_scores'])}</span></div>
  {warn}
</div>"""


def render(result: ForecastResult, league: str) -> str:
    t = result.table
    v = result.validation
    weights = ", ".join(f"{k} {w:.2f}" for k, w in result.blend_weights.items())
    context = ""
    if {"blend", "mkt"} <= set(v.index):
        context = (f"Walk-forward validation on {int(v.loc['mkt', 'n'])} past matches: model RPS "
                   f"{v.loc['blend', 'rps']:.4f}, market RPS {v.loc['mkt', 'rps']:.4f} (lower is better). "
                   "The market is the more accurate 1X2 estimate; large model − market gaps are more often "
                   "model error than opportunity.")
    beta = f" Lineup effect β = {result.lineup_beta:.2f} (not applied: lineups unknown before kick-off)." \
        if result.lineup_beta is not None else ""
    cards = "".join(_card(r) for _, r in t.iterrows())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(league)} Forecast</title><style>{CSS}</style></head>
<body><main>
<h1>{html.escape(league)} forecast</h1>
<div class="sub">{len(t)} fixtures from {t['date'].min():%d %b %Y} · models fitted on data before {result.as_of:%d %b %Y}</div>
<p class="note">{context} Blend weights: {weights}.{beta} Research output, not betting advice.</p>
<h2>Fixtures</h2>
{cards}
</main></body></html>"""
