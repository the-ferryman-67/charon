#!/usr/bin/env python3
"""
render_proof.py — turn a Charon manifest into a single self-contained proof page.

    python render_proof.py manifest.json --repo owner/name --out proof.html

No external assets, no tracking, no fonts phoned home. One file you can open
anywhere or host on any static box. The page IS the demo.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone

VERDICT_STYLE = {
    "ABOARD": ("#7ee0b8", "&#9972;"),
    "OBOL-NEEDED": ("#e8c873", "&#129689;"),
    "SHORE": ("#8aa0c0", "&#8987;"),
}


def keeper_title(m: dict, number: int) -> str:
    for row in m["ledger"]:
        if row["pr"] == number:
            return row["title"]
    return ""


def _tide(c: dict, m: dict, repo: str, big: int) -> str:
    pct = max(6, round(100 * c["size"] / big))
    kt = html.escape(keeper_title(m, c["keeper"])[:90])
    url = f"https://github.com/{repo}/pull/{c['keeper']}"
    return f"""
      <div class="tide">
        <div class="bar" style="width:{pct}%"></div>
        <div class="tidemeta">
          <span class="count">{c['size']}&times;</span>
          <span class="label">{html.escape(c['label'])}</span>
          <span class="wait">{c['size'] - 1} can wait &middot; keeper
            <a href="{url}" target="_blank" rel="noopener">#{c['keeper']}</a></span>
        </div>
        <div class="keepertitle">{kt}</div>
      </div>"""


def tide_blocks(m: dict, repo: str) -> str:
    big = max((c["size"] for c in m["crowds"]), default=1)
    return "".join(_tide(c, m, repo, big) for c in m["crowds"][:14])


def _pm(r: dict) -> str:
    pm = r.get("p_merge")
    return f"{round(pm * 100)}%" if pm is not None else "&mdash;"


def _row(r: dict, repo: str) -> str:
    colour, glyph = VERDICT_STYLE[r["verdict"]]
    url = f"https://github.com/{repo}/pull/{r['pr']}"
    return f"""
        <tr>
          <td class="pr"><a href="{url}" target="_blank" rel="noopener">#{r['pr']}</a></td>
          <td><span class="badge" style="color:{colour};border-color:{colour}">{glyph} {r['verdict']}</span></td>
          <td class="obol">{r['obols']:.0f}</td>
          <td class="pm">{_pm(r)}</td>
          <td class="title">{html.escape(r['title'][:96])}</td>
          <td class="reason">{html.escape(r['reason'][:80])}</td>
        </tr>"""


def ledger_rows(m: dict, repo: str, limit: int = 24) -> str:
    cand = [r for r in m["ledger"] if r["verdict"] != "SHORE"]
    cand.sort(key=lambda r: (r.get("p_merge") or 0.0), reverse=True)
    return "".join(_row(r, repo) for r in cand[:limit])


def model_panel(m: dict) -> str:
    md = m.get("model") or {}
    if not md.get("lr_auc"):
        return ""
    turned = md["n"] - md["merged"]
    return f"""
  <h2>What Charon learned from your own history</h2>
  <div class="note"><b>He studied {md['n']} of your closed PRs</b> ({md['merged']} merged,
  {turned} turned back) and trained two classifiers &mdash; logistic regression and complement
  naive Bayes &mdash; to predict what you tend to accept. Honest score: <b>{md['lr_auc']} /
  {md['nb_auc']}</b> cross-validated AUC (0.5 is a coin-flip, 1.0 is perfect; ~0.70 is a real,
  modest signal, not a promise). Every soul below carries his learned merge-likelihood.
  Trained locally; the history never left the box.</div>"""


def render(m: dict, repo: str, stamp: str) -> str:
    v = m["verdicts"]
    distinct_pct = round(100 * m["clusters"] / max(1, m["souls"]))
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Charon &mdash; the ferryman of {html.escape(repo)}</title>
<style>
  :root {{ --ink:#dfe8f5; --dim:#8aa0c0; --gold:#e8c873; --green:#7ee0b8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:radial-gradient(120% 80% at 50% -10%,#0e1830 0%,#070b14 60%);
         color:var(--ink); font:16px/1.6 ui-monospace,"SF Mono",Menlo,Consolas,monospace; }}
  .wrap {{ max-width:860px; margin:0 auto; padding:48px 22px 80px; }}
  h1 {{ font-size:46px; letter-spacing:.18em; margin:0 0 2px; font-weight:600; }}
  .sub {{ color:var(--dim); letter-spacing:.06em; margin:0 0 4px; }}
  .styx {{ color:#3a4a66; letter-spacing:.5em; font-size:12px; margin:18px 0 28px; }}
  .lede {{ font-size:20px; line-height:1.55; margin:0 0 30px; }}
  .lede b {{ color:var(--gold); }}
  .hero {{ display:flex; gap:14px; flex-wrap:wrap; margin:0 0 8px; }}
  .stat {{ flex:1 1 150px; background:#0c1322; border:1px solid #1b2740; border-radius:12px; padding:16px 18px; }}
  .stat .n {{ font-size:30px; font-weight:600; }}
  .stat .k {{ color:var(--dim); font-size:13px; letter-spacing:.04em; }}
  .green {{ color:var(--green); }} .gold {{ color:var(--gold); }} .blue {{ color:#8aa0c0; }}
  h2 {{ font-size:14px; letter-spacing:.18em; color:var(--dim); text-transform:uppercase;
        margin:42px 0 14px; border-bottom:1px solid #1b2740; padding-bottom:8px; }}
  .tide {{ margin:0 0 16px; }}
  .bar {{ height:8px; border-radius:6px; background:linear-gradient(90deg,#2b6cb0,#7ee0b8); }}
  .tidemeta {{ display:flex; gap:12px; align-items:baseline; margin-top:6px; flex-wrap:wrap; }}
  .count {{ color:var(--gold); font-weight:600; font-size:18px; }}
  .label {{ letter-spacing:.04em; }}
  .wait {{ color:var(--dim); font-size:13px; }}
  .keepertitle {{ color:#9fb2cc; font-size:13px; margin-top:2px; }}
  a {{ color:var(--green); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  td {{ padding:7px 8px; border-bottom:1px solid #141d31; vertical-align:top; }}
  .pr a {{ color:var(--dim); }} .obol {{ color:var(--gold); text-align:center; }}
  .pm {{ color:var(--green); text-align:center; }}
  th {{ text-align:left; color:var(--dim); font-size:11px; letter-spacing:.08em; text-transform:uppercase; padding:6px 8px; border-bottom:1px solid #1b2740; }}
  .badge {{ border:1px solid; border-radius:20px; padding:1px 9px; font-size:11px; white-space:nowrap; }}
  .reason {{ color:var(--dim); }}
  .note {{ background:#0c1322; border:1px solid #1b2740; border-left:3px solid var(--gold);
           border-radius:8px; padding:14px 18px; color:#a9bad4; font-size:13px; margin:26px 0; }}
  footer {{ color:#5a6c8a; font-size:12px; margin-top:40px; line-height:1.7; }}
</style></head>
<body><div class="wrap">
  <h1>CHARON</h1>
  <p class="sub">the ferryman of the Styx &middot; {html.escape(repo)}</p>
  <p class="styx">&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552; S T Y X &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;</p>

  <p class="lede">{m['souls']} souls wait on the shore. But many carry the same offering &mdash;
  <b>they are really {m['clusters']} distinct decisions</b>, not {m['souls']}.
  {m['souls_in_crowds']} of them arrive in {m['duplicate_crowds']} kindred tides.
  The computer is reviewing the computer, on its own box, with nothing leaving the river.</p>

  <div class="hero">
    <div class="stat"><div class="n">{m['souls']}</div><div class="k">souls on the shore</div></div>
    <div class="stat"><div class="n green">{m['clusters']}</div><div class="k">truly-distinct ({distinct_pct}%)</div></div>
    <div class="stat"><div class="n gold">{m['duplicate_crowds']}</div><div class="k">kindred tides</div></div>
  </div>
  <div class="hero">
    <div class="stat"><div class="n green">{v.get('ABOARD', 0)}</div><div class="k">&#9972; ABOARD</div></div>
    <div class="stat"><div class="n gold">{v.get('OBOL-NEEDED', 0)}</div><div class="k">&#129689; OBOL-NEEDED</div></div>
    <div class="stat"><div class="n blue">{v.get('SHORE', 0)}</div><div class="k">&#8987; SHORE (can wait)</div></div>
  </div>

  <h2>The largest tides &mdash; the same soul, many times over</h2>
  {tide_blocks(m, repo)}

  {model_panel(m)}
  <h2>Souls worth the maintainer's eyes first &mdash; ranked by what you'd merge</h2>
  <table>
    <thead><tr><th>soul</th><th>verdict</th><th>obol</th><th>merge?</th><th>offering</th><th>why</th></tr></thead>
    <tbody>{ledger_rows(m, repo)}</tbody></table>

  <div class="note"><b>How Charon judges.</b> He runs entirely on your own machine.
  No cloud, no tracking, no data leaves the box. Kindred souls are found by weighing each
  PR two ways &mdash; by its words (TF-IDF) and by its concepts across the contributors who
  raised them (CF-IOF) &mdash; then he weighs each offering for the coin: tests, a real
  account, a named kind. Every verdict can be signed with a local key, replayable and
  tamper-evident. Your repository stays yours.</div>

  <footer>
  A draft triage over a public snapshot of {html.escape(repo)} &mdash; honest, not binding.
  Charon ran on a clone; point him at your own repo for the live feed.<br>
  &#127754; Odysseus sailed to the underworld in Book 11 to find his way home. Charon is the
  ferryman who decides who crosses. {m['souls']} souls is a lot for one maintainer to ferry alone.<br>
  weighed {html.escape(stamp)} &middot; Charon does not judge the dead. He just checks if they brought the coin.
  </footer>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="proof.html")
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()
    with open(args.manifest) as f:
        m = json.load(f)
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(args.out, "w") as f:
        f.write(render(m, args.repo, stamp))
    print(f"proof page written to {args.out} ({m['souls']} souls, {m['duplicate_crowds']} tides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
