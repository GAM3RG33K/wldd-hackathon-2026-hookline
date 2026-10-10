"""Pitch deck as a web page (/pitch), rendered from pitch/deck.md on every request.

deck.md: a front-matter block (`key: value` lines) then slides separated by a line `---`.
Each slide is light Markdown (# / ## headings, - and 1. lists, | tables |, **bold**, *italic*, `code`,
paragraphs). Lines starting with `Note:` are speaker notes (press n). Live blocks pull real reports:
  {{scores:KEY}} {{report:KEY}} {{why:KEY}} {{emotion:KEY}} {{suggestions:KEY}}   KEY = a front-matter
  key (e.g. showcase) or a reel name; the newest full report of that video is used.
  {{architecture}} the system diagram · {{validation}} live numbers from the sample reels.
Edit deck.md and refresh: no restart needed.
"""
import csv
import html
import json
import re
from pathlib import Path

DECK = Path(__file__).resolve().parent.parent / "pitch" / "deck.md"
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


# ------------------------------------------------------------------ markdown (small, enough for slides)
def _inline(t):
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<i>\1</i>", t)
    return t


def _markdown(lines):
    out, i = [], 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("## "):
            out.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body) +
                       "</table>")
            continue
        elif re.match(r"^(- |\d+\. )", ln):
            ordered = bool(re.match(r"^\d+\. ", ln))
            items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i].rstrip()))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        else:
            para = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|- |\d+\. |\||\{\{)", lines[i + 1]):
                i += 1
                para.append(lines[i].rstrip())
            out.append(f"<p>{_inline(' '.join(para))}</p>")
        i += 1
    return "\n".join(out)


# ------------------------------------------------------------------ data
def _latest_reports(A):
    """name(lower) -> (job, report) for the newest report of each video (full TRIBE preferred)."""
    best = {}
    for p in A.OUT.glob("*/report.json"):
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        name = A.report_name(r, p.parent.name)
        if not name:
            continue
        full = r.get("preview") is False or "full" in str(r.get("neural_source", ""))
        key = (full, A.analysed_at(p))
        if name.lower() not in best or key > best[name.lower()][0]:
            best[name.lower()] = (key, p.parent.name, r)
    return {k: (job, r) for k, (_, job, r) in best.items()}


def _corpus():
    rows = {}
    for s in ("face", "faceless", "outliers"):
        meta = SAMPLES / s / "metadata.csv"
        if meta.exists():
            for r in csv.DictReader(meta.open()):
                rows[r["video_id"].lower()] = {"stratum": s, "views": float(r.get("views") or 0)}
    return rows


def _validation(reports):
    from scipy.stats import spearmanr
    corpus = _corpus()
    pts = [(corpus[k]["stratum"], corpus[k]["views"], r["scores"]["virality"])
           for k, (_, r) in reports.items() if k in corpus and corpus[k]["views"] > 0
           and (r.get("preview") is False or "full" in str(r.get("neural_source", "")))]
    if len(pts) < 5:
        return "<p class=muted>Not enough analysed sample reels yet.</p>"
    rows = []
    for name in ("face", "faceless", "outliers", "all"):
        sub = [p for p in pts if name == "all" or p[0] == name]
        if not sub:
            continue
        views = sorted(x[1] for x in sub)
        vir = [x[2] for x in sub]
        rho = spearmanr([x[1] for x in sub], vir).statistic if len(sub) >= 5 else float("nan")
        rows.append(f"<tr><td>{'<b>all</b>' if name == 'all' else name}</td><td>{len(sub)}</td>"
                    f"<td>{views[len(views) // 2]:,.0f}</td><td>{sum(vir) / len(vir):.0f}</td>"
                    f"<td>{'–' if rho != rho else f'{rho:+.2f}'}</td></tr>")
    return ("<table><tr><th>Reels (full GPU run)</th><th>n</th><th>median views</th><th>mean viral potential</th>"
            "<th>rank correlation with views</th></tr>" + "".join(rows) + "</table>"
            "<p class=muted>Live from the reports on this machine. Views are validation only, never an input. "
            "Weights are hand-set (uncalibrated); a correlation near 0 means the ranking is not yet predictive.</p>")


ARCH = """<div class=arch>
 <div class=box><b>TUI</b><span>pick reel · <code>o</code> file · caption · queue · live ETA · <code>S</code> Space</span></div>
 <div class=arrow>video + caption →</div>
 <div class="box gpu"><b>GPU Space · A100</b><span>speech gate · Hinglish ASR · song match<br>TRIBE v2 brain model
   (V-JEPA2 + Wav2Vec-BERT + Llama)<br>scenes · faces · music · voice · words · caption signals</span></div>
 <div class=arrow>← raw data</div>
 <div class=box><b>Local backend</b><span>attention · hook · hold · emotion fusion<br>caption check · viral potential +
   why · editing suggestions</span></div>
 <div class=arrow>→</div>
 <div class=box><b>Report</b><span>TUI · web (<code>v</code>) · all reports · md / png / json</span></div>
</div>"""


def _block(kind, key, meta, reports, A):
    name = (meta.get(key, key) if key else "").lower()
    if kind == "architecture":
        return ARCH
    if kind == "validation":
        return _validation(reports)
    if name not in reports:
        return f"<p class=muted>[{kind}: no report yet for {html.escape(name)}]</p>"
    job, r = reports[name]
    name = A.report_name(r, job) or name  # display the reel's own name, not the lookup key
    s = r["scores"]
    if kind == "report":
        return (f'<div class=frame><iframe src="/report/{job}/view" loading=lazy title="report"></iframe></div>'
                f'<p class=muted><a href="/report/{job}/view" target=_blank>open {html.escape(name)} in a new tab</a></p>')
    if kind == "scores":
        tiles = [("Hook", f"{s['hook_pct']:.0f}", s.get("hook_band", "")),
                 ("Hold", f"{s['hold_rate_pct']:.0f}%", "stay to the end"),
                 ("Finish", f"{s['finish_pct']:.0f}%", "watch-through"),
                 ("Viral potential", f"{s['virality']:.0f}", s.get("virality_tag", ""))]
        return (f"<p class=muted>Showcase: <b>{html.escape(name)}</b> · {r.get('duration_s', '?')}s</p><div class=tiles>" +
                "".join(f"<div class=tile><div class=k>{k}</div><div class=v>{v}</div><div class=s>{html.escape(str(b))}</div></div>"
                        for k, v, b in tiles) + "</div>")
    if kind == "why":
        return A._virality_html(r)
    if kind == "emotion":
        return f"<p class=muted><b>{html.escape(name)}</b></p>" + A._emotion_html(r)
    if kind == "suggestions":
        return A._suggestions_html(r, job)
    return ""


def render(A):
    text = DECK.read_text(encoding="utf-8")
    head, _, body = text.partition("\n---\n")
    meta = dict(re.findall(r"^(\w+):\s*(.+)$", head, flags=re.M))
    reports = _latest_reports(A)
    slides = []
    for raw in body.split("\n---\n"):
        notes, lines = [], []
        for ln in raw.strip("\n").splitlines():
            (notes.append(ln[5:].strip()) if ln.startswith("Note:") else lines.append(ln))
        parts, buf = [], []
        for ln in lines:
            m = re.fullmatch(r"\{\{(\w+)(?::([\w.-]+))?\}\}", ln.strip())
            if m:
                parts.append(_markdown(buf))
                buf = []
                parts.append(_block(m.group(1), m.group(2), meta, reports, A))
            else:
                buf.append(ln)
        parts.append(_markdown(buf))
        slides.append(f'<section class=slide><div class=inner>{"".join(parts)}</div>'
                      f'<aside class=notes>{_inline(" ".join(notes))}</aside></section>')
    return PAGE.replace("%TITLE%", html.escape(meta.get("title", "Pitch"))).replace("%SLIDES%", "\n".join(slides))


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>%TITLE%</title>
<style>
:root{--bg:#fff;--fg:#111;--muted:#666;--line:#e3e3e3;--accent:#2563eb;--hook:#16a34a;--drop:#dc2626;--strong:#7c3aed;--att:#2563eb;--surv:#9ca3af}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#eee;--muted:#9aa0a6;--line:#2a2d33;--accent:#60a5fa;--att:#60a5fa;--surv:#8b8f97}}
*{box-sizing:border-box} html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font-family:system-ui}
.slide{display:none;position:fixed;inset:0;overflow:auto;padding:4vh 6vw 8vh}
.slide.on{display:block}
.inner{max-width:1200px;margin:0 auto}
h1{font-size:clamp(2rem,4.5vw,3.4rem);margin:.2em 0 .3em} h2{font-size:clamp(1.3rem,2.6vw,2rem);font-weight:500;color:var(--muted)}
p,li,td,th{font-size:clamp(1rem,1.6vw,1.35rem);line-height:1.45} li{margin:.35em 0}
code{background:color-mix(in srgb,var(--line) 60%,transparent);padding:1px 6px;border-radius:5px;font-size:.9em}
b{color:var(--fg)} a{color:var(--accent)} .muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:.6em 0} th,td{text-align:left;padding:.45em .6em;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;font-size:clamp(.85rem,1.2vw,1rem)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:.6em 0 1em}
.tile{border:1px solid var(--line);border-radius:14px;padding:12px;text-align:center}
.tile .k{color:var(--muted)} .tile .v{font-size:clamp(1.8rem,3.5vw,2.8rem);font-weight:700} .tile .s{color:var(--muted)}
.frame{height:68vh;border:1px solid var(--line);border-radius:14px;overflow:hidden}
.frame iframe{width:100%;height:100%;border:0;background:var(--bg)}
.arch{display:flex;flex-wrap:wrap;align-items:stretch;gap:10px;margin:1em 0}
.arch .box{flex:1 1 180px;border:1px solid var(--line);border-radius:14px;padding:14px;display:flex;flex-direction:column;gap:6px}
.arch .box span{color:var(--muted);font-size:clamp(.85rem,1.2vw,1.05rem)} .arch .gpu{border-color:var(--strong)}
.arch .arrow{align-self:center;color:var(--muted);font-size:.95rem}
.notes{display:none;position:fixed;left:0;right:0;bottom:2.4rem;margin:0 6vw;padding:10px 14px;border-radius:10px;
  background:color-mix(in srgb,var(--fg) 88%,transparent);color:var(--bg);font-size:1rem}
body.shownotes .slide.on .notes{display:block}
.bar{position:fixed;left:0;bottom:0;height:4px;background:var(--accent);transition:width .2s}
.hud{position:fixed;right:14px;bottom:10px;color:var(--muted);font-size:.85rem}
.hud kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px}
section.why h2,.caption h2{font-size:1.2rem;color:var(--fg)} .why .wrap{overflow-x:auto}
.vbar{position:relative;display:block;width:140px;height:10px;background:var(--line);border-radius:5px}
.vbar::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--muted)}
.vbar i{position:absolute;top:0;bottom:0;border-radius:5px} .vbar i.up{background:var(--hook)} .vbar i.down{background:var(--drop)}
.pos{color:var(--hook)} .neg{color:var(--drop)} .num{text-align:right;white-space:nowrap}
.emo .band{display:flex;height:22px;border-radius:5px;overflow:hidden;margin:.4em 0} .emo .band i{display:block}
.emo figcaption{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:.85rem}
.sw{display:inline-block;width:14px;height:10px;margin-right:5px;border-radius:2px;vertical-align:middle}
ol.sugs li{margin:.6em 0} .badge{display:inline-block;font-size:.75rem;border:1px solid var(--line);border-radius:10px;padding:1px 8px;margin-left:4px}
form,button{display:none}
@media (max-width:700px){.tiles{grid-template-columns:repeat(2,1fr)} .slide{padding:3vh 16px 9vh}}
</style>
%SLIDES%
<div class=bar id=bar></div>
<div class=hud><span id=count></span> · <kbd>←</kbd><kbd>→</kbd> · <kbd>n</kbd> notes · <kbd>f</kbd> full screen</div>
<script>
const S=[...document.querySelectorAll('.slide')];let i=Math.max(0,Math.min(S.length-1,(parseInt(location.hash.slice(1))||1)-1));
function show(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('on',k===i));
  document.getElementById('count').textContent=(i+1)+' / '+S.length;document.getElementById('bar').style.width=(100*(i+1)/S.length)+'%';
  history.replaceState(null,'','#'+(i+1));}
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(i+1)}
  else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(i-1)}else if(e.key==='Home')show(0);else if(e.key==='End')show(S.length-1);
  else if(e.key==='n')document.body.classList.toggle('shownotes');else if(e.key==='f'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}});
let x0=null;document.addEventListener('touchstart',e=>x0=e.touches[0].clientX);document.addEventListener('touchend',e=>{if(x0===null)return;const d=e.changedTouches[0].clientX-x0;if(Math.abs(d)>50)show(i+(d<0?1:-1));x0=null});
show(i);
</script>"""
