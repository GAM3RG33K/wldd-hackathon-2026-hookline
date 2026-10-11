"""Pitch deck as a web page (/pitch), rendered from pitch/deck.md on every request.

deck.md: front matter (`key: value` lines), then slides separated by a line `---`.
Per slide:
  @cover / @section Name          title slide / eyebrow label
  # / ## headings, - lists, | tables |, **bold**, `code`, short paragraphs
  ::: icons    icon | title | caption        icon row (icons: see ICONS)
  ::: flow     problem → answer              compact problem → answer list
  ::: bars T   label | value | shown          horizontal bars (T may add "| 1.6 counts | 3.0 full" markers)
  ::: stats    value | label                 big numbers
  ::: matrix   layer | items                 layered rows
  {{curve|survival|indicators|emotionband|whybars|edits|scores|report:KEY}}  charts from real reports
  {{architecture}} {{scatter}}             system diagram · reels vs views
  Note: ...                                speaker notes (press n)
Edit deck.md and refresh: no restart needed.
"""
import csv
import html
import json
import math
import re
from pathlib import Path

from backend import theme

DECK = Path(__file__).resolve().parent.parent / "pitch" / "deck.md"
SAMPLES = Path(__file__).resolve().parent.parent / "samples"
E = html.escape

# 24px line icons (stroke = currentColor), drawn for this deck
ICONS = {
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "hook": '<path d="M12 3v10a4 4 0 1 1-8 0"/><path d="M12 3l3 3"/><circle cx="12" cy="3" r="1"/>',
    "drop": '<path d="M3 5h4l3 4 4 6 3 2h4"/><path d="M17 17l4 0"/><path d="M3 19h18" opacity=".35"/>',
    "heart": '<path d="M12 20s-7-4.4-9-9a5 5 0 0 1 9-3 5 5 0 0 1 9 3c-2 4.6-9 9-9 9z"/>',
    "gauge": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/><circle cx="12" cy="17" r="1.2"/>',
    "cut": '<circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M7.8 16.2L18 4M16.2 16.2L6 4"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "ear": '<path d="M7 9a5 5 0 0 1 10 0c0 3-3 4-3 7a3 3 0 0 1-6 0"/><path d="M10 9a2 2 0 0 1 4 0"/>',
    "brain": '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V5a2 2 0 0 0-3-1z"/>'
             '<path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1"/>',
    "wave": '<path d="M2 14c2 0 2-6 4-6s2 10 4 10 2-14 4-14 2 10 4 10 2-4 4-4"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
    "laptop": '<rect x="4" y="5" width="16" height="11" rx="1.5"/><path d="M2 19h20"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "report": '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 15l2.5-3 2 2 3.5-5"/>',
    "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18"/>',
}


def icon(name, size=28):
    return (f'<svg class=ic width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(name, ICONS["spark"])}</svg>')


def fmt_t(t):
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


# ------------------------------------------------------------------ text
def _inline(t):
    t = E(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    return t


def _markdown(lines):
    out = []
    for ln in lines:
        ln = ln.rstrip()
        if not ln.strip():
            continue
        if ln.startswith("## "):
            out.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("- "):
            out.append(f"<p class=bullet>{_inline(ln[2:])}</p>")
        else:
            out.append(f"<p class=lead>{_inline(ln)}</p>")
    return "\n".join(out)


def _rows(lines):
    return [[c.strip() for c in ln.split("|")] for ln in lines if ln.strip()]


def grid_attrs(n):
    """Few cards get bigger; more cards wrap into rows instead of crowding one row."""
    cols = n if n <= 3 else 2 if n == 4 else 3 if n in (5, 6, 9) else 4
    size = "lg" if n <= 3 else "md" if n <= 6 else "sm"
    return f'data-size={size} style="--cols:{cols}"'


def _icons(lines):
    rows = _rows(lines)
    def links(cell):  # "Name=https://..., Name=https://..." -> small outbound link chips
        out = []
        for part in (cell or "").split(", "):
            if "=" in part:
                name, url = part.split("=", 1)
                out.append(f'<a href="{E(url.strip())}" target=_blank rel=noopener>{E(name.strip())} ↗</a>')
        return f'<span class=links>{"".join(out)}</span>' if out else ""
    cells = "".join(f'<li>{icon(r[0])}<b>{_inline(r[1])}</b><span>{_inline(r[2] if len(r) > 2 else "")}</span>'
                    f'{links(r[3] if len(r) > 3 else "")}</li>' for r in rows)
    return f'<ul class="icons grid" {grid_attrs(len(rows))}>{cells}</ul>'


def _flow(lines):
    items = []
    for ln in lines:
        if "→" in ln:
            p, s = (x.strip() for x in ln.split("→", 1))
            items.append(f'<li><span class=p>{_inline(p)}</span><span class=arr>→</span><span class=s>{_inline(s)}</span></li>')
    return f'<ol class=flow>{"".join(items)}</ol>'


def _stats(lines):
    return f'<div class="stats grid" {grid_attrs(len(_rows(lines)))}>' + "".join(f'<div><b>{_inline(r[0])}</b><span>{_inline(r[1])}</span></div>'
                                         for r in _rows(lines)) + '</div>'


def _examples(title, lines):
    """Caption do / don't for one reel: good | caption | evidence."""
    out = []
    for r in _rows(lines):
        good = r[0].lower().startswith("good")
        out.append(f'<li class="{"good" if good else "bad"}"><span class=mark>{"✓" if good else "✕"}</span>'
                   f'<q>{E(r[1])}</q><span class=ev>{E(r[2] if len(r) > 2 else "")}</span></li>')
    goods = "".join(x for x in out if 'class="good"' in x)
    bads = "".join(x for x in out if 'class="bad"' in x)
    return (f'<figure class=examples><figcaption class=eyebrow>{E(title)}</figcaption><div class=excols>'
            f'<div><span class="exhead good">Should be: describes what is seen or said</span><ul>{goods}</ul></div>'
            f'<div><span class="exhead bad">Shouldn\'t be: vague, filler or off-topic</span><ul>{bads}</ul></div></div></figure>')


def _matrix(lines):
    rows = "".join(f'<div class=mrow><b>{_inline(r[0])}</b><span>' +
                   "".join(f"<i>{_inline(x.strip())}</i>" for x in r[1].split("·")) + "</span></div>" for r in _rows(lines))
    return f'<div class=matrix>{rows}</div>'


def _bars(title, lines):
    parts = [x.strip() for x in title.split("|")]
    caption, marks = parts[0], []
    for m in parts[1:]:
        mm = re.match(r"([\d.]+)\s*(.*)", m)
        if mm:
            marks.append((float(mm.group(1)), mm.group(2)))
    rows = _rows(lines)
    starred = any(r[0].startswith("*") for r in rows)  # "*label" = the bar to highlight (default: the first)
    vmax = max([float(r[1]) for r in rows] + [m[0] for m in marks] + [1e-9]) * 1.08
    out = []
    for k, r in enumerate(rows):
        v = float(r[1])
        cls = "hi" if (r[0].startswith("*") if starred else k == 0) else "lo"
        r[0] = r[0].lstrip("*").strip()
        out.append(f'<div class=brow><span class=bl>{_inline(r[0])}</span><span class=btrack>'
                   f'<i class={cls} style="width:{100 * v / vmax:.1f}%"></i>' +
                   "".join(f'<em style="left:{100 * m / vmax:.1f}%"></em>' for m, _ in marks) +
                   f'</span><span class=bv>{_inline(r[2] if len(r) > 2 else r[1])}</span></div>')
    legend = "".join(f'<span><em></em>{E(t)} at {m:g}</span>' for m, t in marks)
    return (f'<figure class=bars><figcaption class=eyebrow>{E(caption)}</figcaption>{"".join(out)}'
            f'{f"<div class=blegend>{legend}</div>" if marks else ""}</figure>')


# ------------------------------------------------------------------ data
def _is_full(r):
    return r.get("preview") is False or "full" in str(r.get("neural_source", ""))


def _latest_reports(A):
    best = {}
    for p in A.OUT.glob("*/report.json"):
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        name = A.report_name(r, p.parent.name)
        if not name:
            continue
        key = (_is_full(r), A.analysed_at(p))
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


# ------------------------------------------------------------------ charts (SVG, drawn to scale)
W, H = 1000, 300


def _xy(n, top=24, bottom=36, left=36, right=12, h=H):
    x = lambda t: left + (W - left - right) * t / max(n - 1, 1)
    y = lambda v: top + (h - top - bottom) * (1 - v / 100)
    return x, y


def _curve(r, annotate=False):
    s = r.get("series") or {}
    a = s.get("attention") or []
    n = len(a)
    if n < 3:
        return ""
    x, y = _xy(n)
    m = r.get("moments") or {}
    g = []
    for v in (0, 50, 100):
        g.append(f'<line x1="36" x2="{W - 12}" y1="{y(v):.1f}" y2="{y(v):.1f}" class=grid />'
                 f'<text x="28" y="{y(v) + 4:.1f}" class="tk end">{v}</text>')
    g.append(f'<rect x="{x(0):.1f}" y="{y(100):.1f}" width="{x(min(3, n - 1)) - x(0):.1f}" height="{y(0) - y(100):.1f}" class=hookz />')
    for a0, b0 in m.get("drop_zones_s", []):
        g.append(f'<rect x="{x(a0):.1f}" y="{y(100):.1f}" width="{max(3, x(min(b0, n - 1)) - x(a0)):.1f}" height="{y(0) - y(100):.1f}" class=dz />')
    sw = m.get("strongest_window_s")
    if sw:
        g.append(f'<rect x="{x(sw[0]):.1f}" y="{y(100):.1f}" width="{x(min(sw[1], n - 1)) - x(sw[0]):.1f}" height="{y(0) - y(100):.1f}" class=sz />')
    if annotate:
        for c in s.get("cuts_s") or []:
            if c < n - 1:
                g.append(f'<line x1="{x(c):.1f}" x2="{x(c):.1f}" y1="{y(0):.1f}" y2="{y(0) - 8:.1f}" class=cut />')
        surv = s.get("survival_pct") or []
        if surv:
            g.append('<polyline class=surv points="' + " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in enumerate(surv)) + '" />')
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in enumerate(a))
    g.append(f'<polygon points="{x(0):.1f},{y(0):.1f} {pts} {x(n - 1):.1f},{y(0):.1f}" class=area />'
             f'<polyline points="{pts}" class=line />')
    pk = m.get("peak_attention_at_s")
    if pk is not None and pk < n:
        g.append(f'<circle cx="{x(pk):.1f}" cy="{y(a[pk]):.1f}" r="5" class=peak />')
    step = 10 if n > 40 else 5
    g += [f'<text x="{x(t):.1f}" y="{H - 14}" class=tk>{fmt_t(t)}</text>' for t in range(0, n, step)]
    labels = []
    if annotate:  # callouts, all at the top band
        used = []

        def lab(t, txt, cls=""):
            if any(abs(x(t) - u) < 90 for u in used):  # never let two callouts collide
                return
            used.append(x(t))
            labels.append(f'<text x="{x(t):.1f}" y="16" class="cl {cls}">{E(txt)}</text>')
        lab(1.5, "Hook 0–3 s", "hk")
        for a0, b0 in m.get("drop_zones_s", [])[:3]:
            lab((a0 + min(b0, n - 1)) / 2, "drop-off", "dr")
        if sw:
            lab((sw[0] + sw[1]) / 2, "strongest", "st")
    return (f'<svg class=chart viewBox="0 0 {W} {H}" role="img" aria-label="attention per second">{"".join(g)}{"".join(labels)}</svg>')


def _legend(items, big=False):
    return f'<div class="legend{" big" if big else ""}">' + "".join(f'<span><i class="{c}"></i>{E(t)}</span>' for c, t in items) + '</div>'


def _survival(r):
    s = r.get("series") or {}
    surv = s.get("survival_pct") or []
    n = len(surv)
    if n < 3:
        return ""
    x, y = _xy(n, h=260)
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in enumerate(surv))
    g = [f'<line x1="36" x2="{W - 12}" y1="{y(v):.1f}" y2="{y(v):.1f}" class=grid /><text x="28" y="{y(v) + 4:.1f}" class="tk end">{v}%</text>'
         for v in (0, 50, 100)]
    g.append(f'<polygon points="{x(0):.1f},{y(0):.1f} {pts} {x(n - 1):.1f},{y(0):.1f}" class=lossarea />'
             f'<polyline points="{pts}" class=lossline />')
    step = 10 if n > 40 else 5
    g += [f'<text x="{x(t):.1f}" y="{260 - 14}" class=tk>{fmt_t(t)}</text>' for t in range(0, n, step)]
    hook3 = surv[min(2, n - 1)]
    sc = r["scores"]
    return (f'<div class=split><div class=big><b>{sc["finish_pct"]:.0f}%</b><span>of viewers predicted to reach the end</span>'
            f'<b class=sm>{100 - hook3:.0f}%</b><span>gone in the first 3 seconds</span></div>'
            f'<svg class=chart viewBox="0 0 {W} 260" role="img" aria-label="viewers remaining">{"".join(g)}</svg></div>'
            f'<p class=caption>Predicted viewers remaining, second by second, for a real hackathon reel ({fmt_t(n)} long).</p>')


def _emotionband(r, colors):
    em = r.get("emotion") or {}
    dom = (em.get("series") or {}).get("dominant") or []
    n = len(dom)
    if not n:
        return ""
    bw = (W - 24) / n
    cells = "".join(f'<rect x="{12 + t * bw:.1f}" y="40" width="{bw + .4:.1f}" height="60" fill="{colors.get(d, "var(--sunken)")}" '
                    f'{"" if d != "calm" else "class=calm"} />' for t, d in enumerate(dom))
    marks = []
    for mm in (em.get("strong") or {}).get("moments", []):
        cx = 12 + (mm["start"] + mm["end"]) / 2 * bw
        marks.append(f'<path d="M{cx:.1f} 34 l-5 -9 h10 z" class=mk /><text x="{cx:.1f}" y="18" class="cl">{E(mm["emotion"])}</text>')
    ticks = "".join(f'<text x="{12 + t * bw:.1f}" y="124" class=tk>{fmt_t(t)}</text>' for t in range(0, n, 10 if n > 40 else 5))
    shown = [d for d in dict.fromkeys(dom) if d != "calm"][:6]
    sm = em.get("summary") or {}
    legend = "".join(f'<span><i style="background:{colors.get(d, "#888")}"></i>{E(d.replace("_", " "))}</span>' for d in shown)
    legend += '<span><i class=calmsw></i>no clear emotion</span>'
    return (f'<svg class=chart viewBox="0 0 {W} 132" role="img" aria-label="emotion per second">{cells}{"".join(marks[:8])}{ticks}</svg>'
            f'<div class=legend>{legend}</div>'
            f'<p class=caption>{E(r.get("source_id") or "")}: rescued elephant, no speech. Sorrow found from music and scenes · '
            f'no clear emotion {sm.get("no_clear_share", 0):.0%} of the reel.</p>')


def _whybars(r, meta=None):
    s = r["scores"]
    b = s.get("virality_breakdown") or []
    if not b:
        return ""
    span = max(10.0, max(abs(x["vs_typical"]) for x in b))
    rows = []
    for x in b:
        d = x["vs_typical"]
        w = 50 * abs(d) / span
        rows.append(f'<div class=wrow><span class=wl>{E(x["label"])}</span><span class=wtrack>'
                    f'<i class="{"up" if d >= 0 else "down"}" style="{"left:50%" if d >= 0 else f"left:{50 - w:.1f}%"};width:{w:.1f}%"></i></span>'
                    f'<span class="wv {"pos" if d >= 0 else "neg"}">{d:+.1f}</span></div>')
    base = s.get("virality_base", s["virality"])
    lift = float((meta or {}).get("caption_lift") or 0)
    if lift and not r.get("caption"):  # optional: what a matching caption would add
        w = 50 * lift / span
        rows.append(f'<div class="wrow opt"><span class=wl>Caption context <em>if its real caption is given</em></span><span class=wtrack>'
                    f'<i class=up style="left:50%;width:{w:.1f}%"></i></span><span class="wv pos">+{lift:.1f}</span></div>')
    with_cap = (f'<span class=withcap>with its real caption: <b>{s["virality"] + lift:.0f}</b></span>' if lift and not r.get("caption") else "")
    return (f'<div class=split><div class=big><b>{s["virality"]:.0f}</b><span>viral potential · {E(s.get("virality_tag", ""))}</span>'
            f'<span class=muted>{base:.1f} from attention, plus lifts</span>{with_cap}</div>'
            f'<div class=why2><div class=waxis><span>drags</span><span>typical reel</span><span>boosts</span></div>{"".join(rows)}</div></div>'
            f'<p class=caption>Each factor\'s pull against a typical reel (that factor at 50/100). Real reel: '
            f'{E(r.get("source_id") or "")}.</p>')


def _edits(r):
    sug = r.get("suggestions") or []
    n = len((r.get("series") or {}).get("attention") or [])
    if not sug or not n:
        return ""
    x, _ = _xy(n)
    curve = _curve(r)
    marks = []
    for i, sg in enumerate(sug[:4], 1):
        a0, b0 = float(sg.get("start", 0)), float(sg.get("end", 0))
        marks.append(f'<rect x="{x(a0):.1f}" y="226" width="{max(6, x(min(b0, n - 1)) - x(a0)):.1f}" height="16" rx="4" class=em />'
                     f'<text x="{(x(a0) + x(min(b0, n - 1))) / 2:.1f}" y="238" class="emn">{i}</text>')
    svg = curve.replace("</svg>", "".join(marks) + "</svg>")
    items = "".join(f'<li><b class=mono>{E(sg["at"])}</b><span>{E(sg["suggestion"][:74] + ("…" if len(sg["suggestion"]) > 74 else ""))}</span>'
                    f'<em>{(sg.get("impact") or {}).get("viewers_lost_pct", 0):.1f}% viewers lost here</em></li>' for sg in sug[:4])
    n_items = min(4, len(sug))
    return f'{svg}<ol class="edits grid" {grid_attrs(n_items)}>{items}</ol>'


def _scatter(reports):
    corpus = _corpus()
    pts = [(corpus[k]["stratum"], corpus[k]["views"], r["scores"]["virality"])
           for k, (_, r) in reports.items() if k in corpus and corpus[k]["views"] > 0 and _is_full(r)]
    if len(pts) < 5:
        return "<p class=muted>Not enough analysed sample reels yet.</p>"
    from scipy.stats import spearmanr
    lo, hi = 3, 8  # 1k .. 100M views
    X = lambda v: 60 + (W - 80) * (min(max(math.log10(v), lo), hi) - lo) / (hi - lo)
    Y = lambda p: 20 + 240 * (1 - p / 100)
    g = []
    for p in (0, 50, 100):
        g.append(f'<line x1="60" x2="{W - 20}" y1="{Y(p):.1f}" y2="{Y(p):.1f}" class=grid /><text x="52" y="{Y(p) + 4:.1f}" class="tk end">{p}</text>')
    for e in range(lo, hi + 1):
        g.append(f'<text x="{X(10 ** e):.1f}" y="284" class=tk>{["1k", "10k", "100k", "1M", "10M", "100M"][e - 3]}</text>')
    for st, v, p in pts:
        g.append(f'<circle cx="{X(v):.1f}" cy="{Y(p):.1f}" r="9" class="dot {st}" />')
    g.append(f'<text x="{W - 20}" y="300" class="tk end">views (log scale) →</text>')
    chips = []
    for name, lab in (("all", "All reels"), ("face", "Face creators"), ("faceless", "Faceless pages"), ("outliers", "Extreme viral")):
        sub = [q for q in pts if name == "all" or q[0] == name]
        if len(sub) >= 5:
            rho = spearmanr([q[1] for q in sub], [q[2] for q in sub]).statistic
            chips.append(f'<span class="chip {"good" if rho >= .3 else "bad" if rho < 0 else ""}">{lab}: ρ {rho:+.2f} · n {len(sub)}</span>')
    return (f'<svg class=chart viewBox="0 0 {W} 304" role="img" aria-label="viral potential against real views">{"".join(g)}</svg>'
            + _legend([("dot face", "face creators"), ("dot faceless", "faceless pages"), ("dot outliers", "extreme viral")], big=True)
            + f'<div class=chips>{"".join(chips)}</div>'
            '<p class=caption>Each dot is a sample reel with a full analysis: its viral potential against its real views. '
            'Views are only used to check, never as an input. Weights are hand-set; fitting them is next.</p>')


def _donut(value, label, sub="", pct=True):
    """Teal ring chart (WLDD style): value 0-100."""
    v = max(0.0, min(100.0, float(value)))
    c = 2 * math.pi * 42
    return (f'<figure class=donut><svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="42" class=track />'
            f'<circle cx="50" cy="50" r="42" class=arc stroke-dasharray="{c * v / 100:.1f} {c:.1f}" transform="rotate(-90 50 50)" /></svg>'
            f'<figcaption><b>{v:.0f}{"%" if pct else ""}</b><span>{E(label)}</span>{f"<em>{E(sub)}</em>" if sub else ""}</figcaption></figure>')


def _phone(job, r):
    return (f'<div class=phone><div class=notch></div><video src="/report/{job}/video#t=1" poster="/report/{job}/poster.jpg" muted loop playsinline '
            f'preload=metadata aria-label="the showcase reel"></video>'
            f'<div class=phonecap><b>{r["scores"]["virality"]:.0f}</b><span>viral potential</span></div></div>')


ARCH = """<div class=arch>
  <div class=node>""" + "{laptop}" + """<span class=eyebrow>Where the creator works</span><h3>The user interface</h3><p>pick a reel · caption · queue · live progress</p></div>
  <div class=wire><span>video + caption</span><i></i></div>
  <div class="node gpu">""" + "{brain}" + """<span class=eyebrow>Video analysis model</span><h3>The virality predictor</h3><p>speech gate · Hinglish speech · brain-response model · scenes · faces · music · voice · caption</p></div>
  <div class=wire><span>evidence, per second</span><i></i></div>
  <div class=node>""" + "{gauge}" + """<span class=eyebrow>Scores and explains</span><h3>The server</h3><p>attention · emotion · caption check · viral potential · edits</p></div>
  <div class=wire><span>report</span><i></i></div>
  <div class=node>""" + "{report}" + """<span class=eyebrow>For the creator</span><h3>Report</h3><p>user interface · web · JSON / MD / PNG</p></div>
</div>
<div class=archnote><span class="chip accent">The predictor produces evidence, never scores</span><span class=chip>the server re-scores every reel in seconds</span><span class=chip>one per-second timeline</span></div>"""


MODEL = """<div class=model>
  <section class=mcol><span class=eyebrow>What it is</span><h3>Meta TRIBE v2</h3>
    <p>A brain-encoding foundation model from Meta FAIR (March 2026): it predicts how the human brain responds to video, audio and text, without an fMRI scan.</p>
    <div class=mstats><div><b>451.6 h</b><span>fMRI, 25 people (training)</span></div><div><b>720</b><span>people in evaluation</span></div>
    <div><b>20,484</b><span>cortex points predicted per step</span></div><div><b>3</b><span>encoders: V-JEPA2 (video), Llama 3.2 (text), Wav2Vec-BERT 2.0 (audio)</span></div></div>
    <a class=src href="https://pasqualepillitteri.it/en/news/2474/meta-tribe-v2-ai-model-predicts-brain-activity-fmri" target=_blank rel=noopener>Read: Meta TRIBE v2 ↗</a></section>
  <section class="mcol mid"><span class=eyebrow>How we use it</span><h3>Our Virality Predictor</h3>
    <ol class=muse><li>Run TRIBE v2 on the reel on a GPU, in 10-second chunks</li><li>Average the predicted response of the visual, auditory, language and attention networks, minus the mind-wandering network</li>
    <li>Blend it 60 / 40 with edit signals into attention per second</li><li>Derive hook, hold, drop-offs and viral potential from that curve</li></ol></section>
  <section class=mcol><span class=eyebrow>The idea we followed</span><h3>Hook score, hold rate, viral potential</h3>
    <p>Commercial "virality predictors" score short clips on hook, hold and brain activation. We built the same kind of scores on an open research model, with every number explained.</p>
    <a class=src href="https://pasqualepillitteri.it/en/news/2273/higgsfield-virality-predictor-hook-score-hold-rate-2026" target=_blank rel=noopener>Read: Higgsfield Virality Predictor ↗</a>
    <span class="chip warn">TRIBE v2 licence: CC BY-NC, research and prototypes only</span></section>
</div>"""

LAUNCH = """<div class=launch><button type=button id=launchbtn class=cta>Launch the user interface</button>
<span id=launchmsg class=muted>Opens the Hookline app in a new Terminal window on this computer.</span></div>"""


def _block(kind, key, meta, reports, A):
    if kind == "model":
        return MODEL
    if kind == "launch":
        return LAUNCH
    if kind == "architecture":
        return ARCH.format(laptop=icon("laptop", 30), brain=icon("brain", 30), gauge=icon("gauge", 30), report=icon("report", 30))
    if kind == "scatter":
        return _scatter(reports)
    name = (meta.get(key, key) if key else "").lower()
    if name not in reports:
        return f"<p class=muted>[{kind}: no report yet for {E(name)}]</p>"
    job, r = reports[name]
    s = r["scores"]
    if kind == "curve":
        return (_curve(r) + _legend([("l-att", "attention per second"), ("l-hook", "hook window"), ("l-dz", "drop-off zone"),
                                     ("l-sz", "strongest window")]))
    if kind == "indicators":
        return (_curve(r, annotate=True) +
                _legend([("l-att", "attention"), ("l-surv", "viewers remaining"), ("l-hook", "hook 0–3 s"),
                         ("l-dz", "drop-off zone"), ("l-sz", "strongest window"), ("l-pk", "peak"), ("l-cut", "frame change (cut)")]) +
                f'<p class=caption>Real reel {E(A.report_name(r, job) or name)}: hook {s["hook_pct"]:.0f}/100, drop-offs at '
                f'{", ".join(fmt_t(a) + "–" + fmt_t(b) for a, b in (r.get("moments") or {}).get("drop_zones_s", [])[:3])}.</p>')
    if kind == "survival":
        return _survival(r)
    if kind == "emotionband":
        return _emotionband(r, A.EMO_COLORS)
    if kind == "whybars":
        return _whybars(r, meta)
    if kind == "edits":
        return _edits(r)
    donuts = (_donut(s["hook_pct"], "Hook", s.get("hook_band", ""), pct=False) +
              _donut(s["hold_rate_pct"], "Hold", "stay to the end") + _donut(s["finish_pct"], "Finish", "watch-through") +
              _donut(s["virality"], "Viral potential", s.get("virality_tag", ""), pct=False))
    if kind == "scores":
        return f'<div class=donuts>{donuts}</div>'
    if kind == "phone":
        return _phone(job, r)
    if kind == "seeit":
        return (f'<div class=seeit><div class="donuts two">{donuts}</div>'
                f'<div class=seephone>{_phone(job, r)}</div></div>')
    return ""


def render(A):
    text = DECK.read_text(encoding="utf-8")
    head, _, body = text.partition("\n---\n")
    meta = dict(re.findall(r"^(\w+):\s*(.+)$", head, flags=re.M))
    reports = _latest_reports(A)
    raw_slides = [s for s in body.split("\n---\n") if s.strip()]
    slides = []
    for idx, raw in enumerate(raw_slides, 1):
        notes, lines, section, cover, closing, titled = [], [], "", False, False, False
        report_job = None
        for ln in raw.strip("\n").splitlines():
            if ln.startswith("Note:"):
                notes.append(ln[5:].strip())
            elif ln.strip() == "@cover":
                cover = True
            elif ln.strip() == "@closing":
                closing = cover = True
            elif ln.strip() == "@title":
                titled = cover = True
            elif ln.startswith("@report"):  # attach a reel's full report to a slide that shows its data in text
                nm = meta.get(ln[7:].strip(), ln[7:].strip()).lower()
                report_job = reports[nm][0] if nm in reports else report_job
            elif ln.startswith("@section"):
                section = ln[len("@section"):].strip()
            else:
                lines.append(ln)
        parts, buf, block, block_arg, block_lines = [], [], None, "", []
        for ln in lines:
            if block:
                if ln.strip() == ":::":
                    parts.append({"icons": lambda: _icons(block_lines), "flow": lambda: _flow(block_lines),
                                  "stats": lambda: _stats(block_lines), "matrix": lambda: _matrix(block_lines),
                                  "bars": lambda: _bars(block_arg, block_lines),
                                  "examples": lambda: _examples(block_arg, block_lines)}[block]())
                    block, block_lines = None, []
                else:
                    block_lines.append(ln)
                continue
            m = re.fullmatch(r":::\s*(icons|flow|bars|stats|matrix|examples)\s*(.*)", ln.strip())
            if m:
                parts.append(_markdown(buf))
                buf, block, block_arg = [], m.group(1), m.group(2)
                continue
            m = re.fullmatch(r"\{\{(\w+)(?::([\w.-]+))?\}\}", ln.strip())
            if m:
                parts.append(_markdown(buf))
                buf = []
                parts.append(f'<div class=visual>{_block(m.group(1), m.group(2), meta, reports, A)}</div>')
                nm = (meta.get(m.group(2), m.group(2)) if m.group(2) else "").lower()
                if report_job is None and nm in reports:  # this slide shows a reel's data: link its full report
                    report_job = reports[nm][0]
            else:
                buf.append(ln)
        parts.append(_markdown(buf))
        content = "".join(parts)
        if cover and not closing and not titled:
            sc = meta.get("showcase", "").lower()
            report_job = reports[sc][0] if sc in reports else report_job
        rbtn = (f'<a class=rbtn href="/report/{report_job}/view" target=_blank rel=noopener>Open full report ↗</a>'
                if report_job else "")
        if titled:
            content = (f'<div class="cover titled"><div class=covertext><span class=kicker>{E(meta.get("team", ""))}</span>{content}'
                       f'<p class=author><span class=eyebrow>Author</span>{E(meta.get("author", ""))}</p></div>'
                       f'<div class=coverblock><span class=bigmark>Problem<br>01</span><span class=chev>»»</span></div></div>')
        elif closing:
            content = (f'<div class="cover closing"><div class=covertext><span class=kicker>{E(meta.get("team", ""))}</span>{content}</div>'
                       f'<div class=coverblock><span class=bigmark>{E(meta.get("title", ""))}</span><span class=chev>»»</span></div></div>')
        elif cover:
            sc = meta.get("showcase", "").lower()
            phone = _phone(*reports[sc]) if sc in reports else ""
            content = (f'<div class=cover><div class=covertext><span class=kicker>{E(meta.get("team", ""))}</span>{content}{rbtn}</div>'
                       f'<div class=coverblock>{phone}<span class=chev>»»</span></div></div>')
        top = (f'<header class=slidehead><span class=dot>{idx:02d}</span><span class=eyebrow>{E(section)}</span>'
               f'<span class=line></span>{rbtn}</header>'
               f'<div class=rail aria-hidden="true">{E(section)}</div>') if not cover else ""
        slides.append(f'<section class="slide{" is-cover" if cover else ""}" aria-label="Slide {idx}">{top}'
                      f'<div class=inner>{content}</div>'
                      f'<aside class=notes><span class=eyebrow>Speaker notes</span> {_inline(" ".join(notes)) or "—"}</aside></section>')
    return (PAGE.replace("%HEAD%", theme.page_head(E(meta.get("title", "Pitch"))))
            .replace("%TOKENS%", theme.TOKENS).replace("%BRAND%", f'{E(meta.get("title", ""))} — {E(meta.get("subtitle", ""))}'
                                                                 f'<span class=by> · {E(meta.get("author", ""))}</span>')
            .replace("%SLIDES%", "\n".join(slides)))


PAGE = """<!doctype html><html lang=en><head>%HEAD%
<style>%TOKENS%
/* WLDD deck identity (from the WLDD case-study deck): deep navy ground, royal-blue colour blocks,
   heavy white geometric type, teal for data (rings, lines), chevrons and numbered circles. Single dark look by choice. */
html:root:root{--bg:#101a3e;--surface:#172452;--sunken:#1e2c61;--ink:#ffffff;--muted:#bcc5ea;--faint:#8592c6;--line:#2b3a78;
  --accent:#5fe3df;--accent-soft:rgba(95,227,223,.13);--blue:#4f63f4;--blue-deep:#2149e6;
  --good:#5ee39a;--warn:#ffc65c;--bad:#ff7d6e;--peak:#b8a4ff;--surv:#8d98c9;
  --display:"Montserrat","Avenir Next","Segoe UI",system-ui,sans-serif;--sans:"Montserrat","Helvetica Neue","Segoe UI",system-ui,sans-serif;
  --r:4px;color-scheme:dark}
/* Deck: one idea per slide; headline, then one visual; max width 1180px. */
html,body{height:100%} body{overflow:hidden}
.slide{display:none;position:fixed;inset:0;overflow:auto;padding:clamp(18px,3vh,30px) clamp(16px,5vw,64px) 76px;background:var(--bg)}
.slide.on{display:flex;flex-direction:column}
.slidehead{width:100%;max-width:1180px;margin:0 auto 3vh;display:flex;justify-content:space-between;padding-bottom:12px;border-bottom:1px solid var(--line)}
.inner{width:100%;max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:clamp(14px,2.6vh,28px);flex:1;justify-content:center}
.inner > *{min-width:0}
h1{font-size:clamp(2rem,4.2vw,3.4rem);letter-spacing:-.02em}
h2{font:500 clamp(1.15rem,2.2vw,1.7rem)/1.3 var(--display);color:var(--muted)}
h2 b{color:var(--accent);font-weight:700}
p.lead{font-size:clamp(1.05rem,1.6vw,1.35rem);color:var(--muted);margin:0}
.caption{font-size:.88rem;color:var(--muted);margin:0} .caption a{color:var(--accent)}
code{background:var(--sunken);border:1px solid var(--line);padding:0 6px;border-radius:6px;font-size:.88em}
.visual{display:flex;flex-direction:column;gap:12px}
.ic{color:var(--accent);flex:none}
/* cover */
.is-cover .inner{justify-content:center}
.cover{display:flex;flex-direction:column;gap:16px;align-items:flex-start}
.cover h1{font-size:clamp(3.2rem,9vw,7rem);letter-spacing:-.04em;line-height:.92}
.cover h2{color:var(--ink);font-size:clamp(1.3rem,2.8vw,2.2rem)}
.cover .visual{width:100%}
/* charts */
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--line)} .chart .tk{font:12px var(--mono);fill:var(--faint);text-anchor:middle} .chart .tk.end{text-anchor:end}
.chart .line{fill:none;stroke:var(--accent);stroke-width:2.6;stroke-linejoin:round}
.chart .area{fill:var(--accent);opacity:.1}
.chart .hookz{fill:var(--good);opacity:.13} .chart .dz{fill:var(--bad);opacity:.14} .chart .sz{fill:var(--peak);opacity:.14}
.chart .peak{fill:var(--peak);stroke:var(--surface);stroke-width:2}
.chart .surv{fill:none;stroke:var(--surv);stroke-width:2;stroke-dasharray:6 5}
.chart .cut{stroke:var(--faint);stroke-width:1.5}
.chart .cl{font:600 12px var(--sans);fill:var(--ink);text-anchor:middle}
.chart .cl.hk{fill:var(--good)} .chart .cl.dr{fill:var(--bad)} .chart .cl.st{fill:var(--peak)}
.chart .lossline{fill:none;stroke:var(--bad);stroke-width:2.6} .chart .lossarea{fill:var(--bad);opacity:.1}
.chart .calm{fill:var(--sunken)} .chart .mk{fill:var(--ink)}
.chart .em{fill:var(--accent)} .chart .emn{font:600 11px var(--sans);fill:var(--bg);text-anchor:middle}
.chart .dot{stroke:var(--surface);stroke-width:1.5}
.chart .dot.face{fill:var(--accent)} .chart .dot.faceless{fill:var(--warn)} .chart .dot.outliers{fill:var(--peak)}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:.84rem;color:var(--muted)}
.legend i{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle;background:var(--faint)}
.legend .l-att{background:var(--accent)} .legend .l-hook{background:var(--good);opacity:.5} .legend .l-dz{background:var(--bad);opacity:.5}
.legend .l-sz{background:var(--peak);opacity:.5} .legend .l-pk{background:var(--peak);border-radius:50%;width:10px}
.legend .l-surv{background:none;border-top:2px dashed var(--surv);height:0;width:16px} .legend .l-cut{width:2px;background:var(--faint)}
.legend .dot{border-radius:50%;width:10px}
.icons li .links{position:absolute;top:14px;right:14px;display:flex;flex-direction:column;align-items:flex-end;gap:5px;max-width:55%}
.icons li .links a{font:600 .74rem var(--sans);color:var(--accent);text-decoration:none;border:1px solid var(--line);padding:2px 8px;border-radius:999px}
.icons li .links a:hover{border-color:var(--accent)}
.legend.big{display:flex;flex-direction:row;flex-wrap:wrap;align-items:center;font:600 1.15rem var(--sans);color:var(--ink);gap:12px 32px}
.legend.big span{display:inline-flex;align-items:center}
.visual .chips .chip{font-size:.95rem;padding:5px 14px}
.legend.big i.dot{width:22px;height:22px;margin-right:10px;box-shadow:0 0 0 3px var(--surface)} .legend .dot.face{background:var(--accent)} .legend .dot.faceless{background:var(--warn)} .legend .dot.outliers{background:var(--peak)}
.legend .calmsw{background:var(--sunken);border:1px solid var(--line)}
.split{display:grid;grid-template-columns:minmax(200px,.32fr) 1fr;gap:clamp(16px,3vw,40px);align-items:center}
.big{display:flex;flex-direction:column;gap:4px}
.big b{font:700 clamp(3.4rem,8vw,6.4rem)/.95 var(--display);color:var(--ink);letter-spacing:-.03em}
.big b.sm{font-size:clamp(2rem,4vw,3rem);margin-top:14px;color:var(--bad)}
.big span{color:var(--muted);font-size:1rem}
/* icons */
.icons{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,140px),1fr));gap:var(--gap)}
.icons li{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:18px 16px;display:flex;flex-direction:column;gap:8px;position:relative}
.icons li b{font:600 1.05rem/1.25 var(--display)} .icons li span{color:var(--muted);font-size:.9rem}
/* flow */
.flow{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 28px;counter-reset:f}
.flow li{display:grid;grid-template-columns:2.1em 1fr auto 1.1fr;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--line);font-size:clamp(.92rem,1.2vw,1.04rem)}
.flow li::before{counter-increment:f;content:counter(f,decimal-leading-zero);font:500 .78rem var(--mono);color:var(--faint)}
.flow .p{color:var(--muted)} .flow .arr{color:var(--accent);font-weight:600} .flow .s{font-weight:600}
/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));gap:var(--gap)}
.stats div{border-top:2px solid var(--accent);padding-top:10px;display:flex;flex-direction:column;gap:2px}
.stats b{font:700 clamp(1.8rem,3.4vw,2.6rem)/1.1 var(--display)} .stats span{color:var(--muted)}
/* bars */
.bars{margin:0;display:flex;flex-direction:column;gap:12px}
.brow{display:grid;grid-template-columns:minmax(150px,.9fr) 2fr auto;align-items:center;gap:14px}
.bl{font-size:.98rem} .bv{font:500 .95rem var(--mono);white-space:nowrap}
.btrack{position:relative;height:22px;background:var(--sunken);border-radius:6px}
.btrack i{position:absolute;left:0;top:0;bottom:0;border-radius:6px} .btrack i.hi{background:var(--accent)} .btrack i.lo{background:var(--faint)}
.btrack em{position:absolute;top:-5px;bottom:-5px;width:2px;background:var(--ink)}
.blegend{display:flex;gap:18px;font-size:.84rem;color:var(--muted)} .blegend em{display:inline-block;width:2px;height:12px;background:var(--ink);margin-right:6px;vertical-align:middle}
/* matrix */
.matrix{display:flex;flex-direction:column;gap:10px}
.mrow{display:grid;grid-template-columns:110px 1fr;gap:16px;align-items:center}
.mrow b{font:700 1.15rem var(--display)}
.mrow span{display:flex;flex-wrap:wrap;gap:8px}
.mrow i{font-style:normal;background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:.95rem}
.mrow:nth-child(4) i{background:var(--accent-soft);border-color:transparent;color:var(--accent);font-weight:600}
/* architecture */
.arch{display:grid;grid-template-columns:1fr auto 1.25fr auto 1.1fr auto 1fr;align-items:stretch}
.node{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:18px;display:flex;flex-direction:column;gap:6px}
.node.gpu{border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}
.node h3{font-size:1.3rem} .node p{margin:0;color:var(--muted);font-size:.9rem}
.wire{display:flex;flex-direction:column;justify-content:center;align-items:center;gap:6px;padding:0 8px;min-width:104px}
.wire span{font:500 .74rem/1.2 var(--mono);text-align:center}
.wire i{display:block;width:100%;height:2px;background:var(--accent);position:relative}
.wire i::after{content:"";position:absolute;right:-1px;top:-4px;border:5px solid transparent;border-left:7px solid var(--accent)}
.archnote{display:flex;flex-wrap:wrap;gap:10px}
/* why */
.why2{display:flex;flex-direction:column;gap:10px}
.waxis{display:grid;grid-template-columns:150px 1fr 1fr 1fr 56px;font:500 .72rem var(--mono);color:var(--faint);text-transform:uppercase;letter-spacing:.06em}
.waxis span:nth-child(1){grid-column:2;text-align:left} .waxis span:nth-child(2){text-align:center} .waxis span:nth-child(3){text-align:right}
.wrow{display:grid;grid-template-columns:150px 3fr 56px;align-items:center;gap:12px}
.wl{font-size:.95rem} .wv{font:600 .95rem var(--mono);text-align:right}
.wtrack{position:relative;height:18px;background:var(--sunken);border-radius:5px}
.wtrack::after{content:"";position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:var(--faint)}
.wtrack i{position:absolute;top:0;bottom:0;border-radius:5px} .wtrack i.up{background:var(--good)} .wtrack i.down{background:var(--bad)}
.pos{color:var(--good)} .neg{color:var(--bad)}
/* edits */
.edits{list-style:none;counter-reset:e;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:var(--gap)}
.edits li{counter-increment:e;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px 14px 48px;position:relative;display:flex;flex-direction:column;gap:4px}
.edits li::before{content:counter(e);position:absolute;left:14px;top:14px;width:22px;height:22px;border-radius:50%;background:var(--accent);color:var(--bg);font:600 .8rem/22px var(--sans);text-align:center}
.edits li span{font-size:.95rem} .edits li em{font-style:normal;color:var(--muted);font-size:.84rem}
/* live */
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--gap)}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.tile b{font:700 clamp(1.7rem,3vw,2.4rem)/1.1 var(--display)} .tile .muted{font-size:.85rem}
.frame{height:56vh;border:1px solid var(--line);border-radius:var(--r);overflow:hidden;background:var(--surface)}
.frame iframe{width:100%;height:100%;border:0}
.chips{display:flex;flex-wrap:wrap;gap:8px}
/* chrome */
.notes{display:none;position:fixed;left:50%;transform:translateX(-50%);bottom:56px;width:min(92vw,900px);padding:12px 16px;
  border-radius:var(--r);background:var(--ink);color:var(--bg);font-size:.95rem;line-height:1.45;z-index:5}
.notes .eyebrow{color:var(--bg);opacity:.7;margin-right:6px}
body.shownotes .slide.on .notes{display:block}
.chrome{position:fixed;left:0;right:0;bottom:0;padding:10px clamp(16px,5vw,64px) calc(10px + env(safe-area-inset-bottom,0px));
  display:flex;justify-content:space-between;align-items:center;gap:12px;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(6px);border-top:1px solid var(--line);z-index:4}
.brand{font:700 1rem var(--display)} .brand i{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--accent);margin-right:8px;vertical-align:1px}
.ctl{display:flex;gap:6px;align-items:center}
.ctl button{font:500 .85rem var(--sans);color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer}
.ctl button:hover{border-color:var(--accent);color:var(--accent)}
.prog{position:fixed;left:0;top:0;height:3px;background:var(--accent);transition:width .25s;z-index:6}
/* WLDD components */
h1{font-weight:800;letter-spacing:-.01em}
.slide{padding-right:calc(clamp(16px,5vw,64px) + 64px)}
.rail{position:fixed;top:0;right:0;bottom:0;width:64px;background:var(--blue);color:#fff;display:flex;align-items:center;justify-content:center;
  writing-mode:vertical-rl;transform:rotate(180deg);font:800 clamp(1rem,2.2vh,1.5rem)/1 var(--display);text-transform:uppercase;letter-spacing:.06em;z-index:3}
.slidehead{border-bottom:0;justify-content:flex-start;gap:12px;align-items:center}
.slidehead .line{flex:1;height:2px;background:linear-gradient(90deg,var(--blue),transparent)}
.rbtn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;background:var(--blue);color:#fff;text-decoration:none;
  font:700 .85rem var(--sans);letter-spacing:.02em;border-radius:3px;white-space:nowrap}
.rbtn:hover{background:var(--blue-deep)} .covertext .rbtn{align-self:flex-start}
/* adaptive card grids: few cards get room, more cards wrap into rows */
.grid{grid-template-columns:repeat(var(--cols,3),minmax(0,1fr))!important}
.icons[data-size=lg] li{padding:28px 26px;gap:12px} .icons[data-size=lg] li .ic{width:72px;height:72px;padding:16px}
.icons[data-size=lg] li b{font-size:1.45rem} .icons[data-size=lg] li span{font-size:1.05rem}
.icons[data-size=md] li{padding:22px 20px} .icons[data-size=md] li b{font-size:1.2rem} .icons[data-size=md] li span{font-size:.98rem}
.icons[data-size=sm] li{padding:14px 14px;gap:6px} .icons[data-size=sm] li .ic{width:44px;height:44px;padding:9px}
.icons[data-size=sm] li b{font-size:.98rem} .icons[data-size=sm] li span{font-size:.84rem}
.stats[data-size=lg] b{font-size:clamp(2.2rem,4.4vw,3.4rem)}
.edits[data-size=lg] li{padding:20px 22px 20px 56px} .edits[data-size=lg] li span{font-size:1.08rem}
/* title screen */
.titled .covertext h1{font-size:clamp(4rem,11vw,9rem)}
.titled .covertext h2{font:800 clamp(1.6rem,3.6vw,2.8rem)/1.1 var(--display);color:var(--accent);text-transform:uppercase;letter-spacing:.01em}
.author{margin:18px 0 0;display:flex;flex-direction:column;gap:4px;font-size:1.1rem;color:var(--ink);border-left:3px solid var(--blue);padding-left:14px}
.titled .bigmark{font-size:clamp(3rem,7vw,6rem);line-height:.95;text-align:left}
/* model page */
.model{display:grid;grid-template-columns:1.15fr 1fr 1fr;gap:var(--gap)}
.mcol{background:var(--surface);padding:22px;display:flex;flex-direction:column;gap:10px;border-radius:var(--r)}
.mcol.mid{background:var(--blue)} .mcol.mid .eyebrow{color:#e3e8ff} .mcol h3{font-size:1.3rem} .mcol p{color:var(--muted);font-size:.95rem;margin:0}
.mstats{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.mstats div{display:flex;flex-direction:column;border-top:2px solid var(--accent);padding-top:6px}
.mstats b{font:800 1.5rem var(--display)} .mstats span{font-size:.8rem;color:var(--muted)}
.muse{margin:0;padding-left:1.2em;display:grid;gap:10px;color:#fff;font-size:.98rem}
.src{color:var(--accent);font-weight:700;text-decoration:none;margin-top:auto} .src:hover{text-decoration:underline}
/* caption examples */
.examples{margin:0;display:flex;flex-direction:column;gap:10px}
.excols{display:grid;grid-template-columns:1fr 1fr;gap:var(--gap)}
.excols ul{list-style:none;margin:8px 0 0;padding:0;display:grid;gap:10px}
.excols li{background:var(--surface);padding:14px 16px 14px 50px;position:relative;display:flex;flex-direction:column;gap:6px;border-radius:var(--r)}
.excols li q{font-size:1rem;quotes:"“" "”"} .excols li .ev{font:500 .82rem var(--mono);color:var(--muted)}
.excols .mark{position:absolute;left:14px;top:14px;width:24px;height:24px;border-radius:50%;font:800 .85rem/24px var(--sans);text-align:center}
.excols li.good .mark{background:var(--good);color:var(--bg)} .excols li.bad .mark{background:var(--bad);color:var(--bg)}
.excols li.good{box-shadow:inset 3px 0 0 var(--good)} .excols li.bad{box-shadow:inset 3px 0 0 var(--bad)}
.exhead{font:700 .82rem var(--sans);text-transform:uppercase;letter-spacing:.06em} .exhead.good{color:var(--good)} .exhead.bad{color:var(--bad)}
/* verdict: optional caption row */
.wrow.opt{opacity:.9} .wrow.opt .wtrack{background:transparent;outline:1px dashed var(--line)}
.wrow.opt .wtrack i{background:repeating-linear-gradient(90deg,var(--good) 0 8px,transparent 8px 12px)}
.wrow.opt .wl em{display:block;font-style:normal;font-size:.78rem;color:var(--accent)}
.withcap{margin-top:12px;padding:8px 12px;border:1px dashed var(--accent);color:var(--muted);font-size:.9rem}
.withcap b{color:var(--accent);font:800 1.4rem var(--display);margin-left:4px}
/* see it */
.seeit{display:grid;grid-template-columns:1.2fr .8fr;gap:clamp(16px,3vw,40px);align-items:center}
.donuts.two{grid-template-columns:repeat(2,1fr)} .donuts.two .donut{padding:22px} .donuts.two .donut svg{width:110px;height:110px}
.donuts.two .donut b{font-size:2.4rem}
.seephone{display:flex;justify-content:center;background:var(--blue);padding:24px 0;border-radius:var(--r)}
.seephone .phone{height:min(56vh,480px)} .seephone .phonecap{display:none}
/* launch */
.launch{display:flex;flex-wrap:wrap;align-items:center;gap:16px}
.cta{font:800 1.05rem var(--sans);color:var(--bg);background:var(--accent);border:0;padding:14px 26px;border-radius:3px;cursor:pointer}
.cta:hover{filter:brightness(1.08)} .cta:disabled{opacity:.6;cursor:default}
.brand .by{font:500 .9rem var(--sans);color:var(--muted)}
@media (max-width:900px){.model,.excols,.seeit{grid-template-columns:1fr} .grid{grid-template-columns:repeat(min(var(--cols,2),2),minmax(0,1fr))!important}
  .brand .by{display:none}}
.slidehead .dot{width:30px;height:30px;border-radius:50%;background:var(--blue);color:#fff;font:700 .78rem/30px var(--sans);text-align:center}
.slidehead .eyebrow{color:var(--accent)}
.is-cover{padding:0!important} .is-cover .inner{max-width:none;margin:0;height:100%}
.cover{display:grid;grid-template-columns:1.05fr .95fr;min-height:calc(100vh - 58px);gap:0;align-items:stretch}
.covertext{display:flex;flex-direction:column;justify-content:center;gap:18px;padding:6vh clamp(20px,5vw,72px)}
.covertext h1{font:800 clamp(3.4rem,8.5vw,7.4rem)/.9 var(--display);text-transform:uppercase;letter-spacing:-.02em}
.covertext h2{color:var(--ink);font-weight:600} .covertext h2 b{color:var(--accent)}
.covertext .visual{margin-top:8px}
.kicker{font:600 .82rem var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.coverblock{background:var(--blue);position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}
.coverblock::before{content:"";position:absolute;left:0;bottom:0;width:38%;height:34%;background:var(--blue-deep)}
.closing .covertext h1{font-size:clamp(3.6rem,10vw,8.4rem);text-transform:none;letter-spacing:-.03em}
.bigmark{font:800 clamp(2.4rem,6vw,5rem)/1 var(--display);color:#fff;text-transform:uppercase;letter-spacing:-.02em;z-index:1}
.chev{position:absolute;right:28px;bottom:16px;font:800 3.4rem/1 var(--display);color:#fff;letter-spacing:-.08em}
.phone{position:relative;height:min(64vh,560px);aspect-ratio:9/19;max-width:100%;background:#05070f;border-radius:34px;padding:10px;
  box-shadow:0 30px 60px rgba(0,0,0,.45),0 0 0 2px #2a2f45;z-index:1}
.phone video{width:100%;height:100%;object-fit:cover;border-radius:26px;display:block;background:#000}
.phone .notch{position:absolute;top:14px;left:50%;transform:translateX(-50%);width:34%;height:16px;background:#05070f;border-radius:0 0 10px 10px;z-index:2}
.phonecap{position:absolute;right:-22%;left:auto;top:18%;background:var(--bg);border:2px solid var(--accent);padding:8px 14px;display:flex;flex-direction:column;z-index:2}
.phonecap b{font:800 2rem/1 var(--display);color:var(--accent)} .phonecap span{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.donuts{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--gap)}
.donut{margin:0;display:flex;align-items:center;gap:14px;background:var(--surface);padding:14px 16px}
.donut svg{width:84px;height:84px;flex:none} .donut .track{fill:none;stroke:#3b4466;stroke-width:12}
.donut .arc{fill:none;stroke:var(--accent);stroke-width:12;stroke-linecap:round}
.donut figcaption{display:flex;flex-direction:column} .donut b{font:800 1.9rem/1 var(--display)}
.donut span{font:600 .78rem var(--sans);text-transform:uppercase;letter-spacing:.08em;color:var(--muted)} .donut em{font-style:normal;color:var(--accent);font-size:.85rem}
.icons li{border:0;background:var(--surface);align-items:flex-start}
.icons li .ic{width:54px;height:54px;padding:12px;border-radius:50%;border:3px solid var(--blue);color:var(--ink);background:var(--bg)}
.stats div{background:var(--blue);border:0;padding:44px 18px 20px;position:relative;text-align:center;align-items:center}
.stats div::before{counter-increment:st;content:counter(st);position:absolute;top:12px;left:50%;transform:translateX(-50%);
  width:24px;height:24px;border-radius:50%;background:var(--bg);color:#fff;font:700 .75rem/24px var(--sans)}
.stats{counter-reset:st} .stats span{color:#e3e8ff;font-weight:500}
.flow .arr{color:var(--accent)} .flow .arr::before{content:"»";font-weight:800} .flow .arr{font-size:0} .flow .arr::before{font-size:1.2rem}
.node{border:0} .node.gpu{background:var(--blue);box-shadow:none} .node.gpu p,.node.gpu .eyebrow{color:#e3e8ff}
.node .ic{color:var(--accent)} .node.gpu .ic{color:#fff}
.wire i{background:var(--accent)} .wire i::after{border-left-color:var(--accent)}
.btrack i.hi{background:var(--accent)} .card,.icons li,.edits li,.tile,.tablewrap{border-radius:var(--r)}
.mrow i{border-color:var(--line);background:var(--surface)} .mrow:nth-child(4) i{background:var(--blue);color:#fff}
.edits li::before{background:var(--blue);color:#fff}
.chart .line{stroke:var(--accent)} .chart .area{fill:var(--accent)}
.chart .em{fill:var(--blue)} .chart .emn{fill:#fff}
.chip.accent{background:var(--blue);color:#fff}
.chrome{background:color-mix(in srgb,var(--bg) 92%,transparent);padding-right:calc(clamp(16px,5vw,64px) + 64px)}
.prog{background:var(--accent)}
@media (max-width:900px){
  .rail{display:none} .slide,.chrome{padding-right:clamp(16px,5vw,64px)}
  .cover{grid-template-columns:1fr} .coverblock{min-height:52vh} .phonecap{right:4%} .donuts{grid-template-columns:repeat(2,1fr)}
  .arch{grid-template-columns:1fr} .wire{min-height:44px;flex-direction:row} .wire i{width:2px;height:24px}
  .wire i::after{right:-4px;top:auto;bottom:-6px;border:5px solid transparent;border-top:7px solid var(--accent)}
  .split{grid-template-columns:1fr} .flow{grid-template-columns:1fr} .tiles{grid-template-columns:repeat(2,1fr)}
  .brow{grid-template-columns:1fr auto} .brow .btrack{grid-column:1/-1;order:3}
  .wrow{grid-template-columns:110px 1fr 48px} .waxis{display:none} .mrow{grid-template-columns:1fr}
}
</style></head><body>
<div class=prog id=prog></div>
%SLIDES%
<nav class=chrome aria-label="Slides"><span class=brand><i></i>%BRAND%</span>
<span class=ctl><button id=prev aria-label="Previous slide">←</button><span class="eyebrow num" id=count></span>
<button id=next aria-label="Next slide">→</button><button id=notesbtn>Notes</button><button id=fs>Full screen</button></span></nav>
<script>
const S=[...document.querySelectorAll('.slide')];let i=Math.max(0,Math.min(S.length-1,(parseInt(location.hash.slice(1))||1)-1));
function show(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('on',k===i));
  document.getElementById('count').textContent=(i+1)+' / '+S.length;document.getElementById('prog').style.width=(100*(i+1)/S.length)+'%';
  history.replaceState(null,'','#'+(i+1));S[i].scrollTop=0;
  document.querySelectorAll('.phone video').forEach(v=>{if(S[i].contains(v)){v.play().catch(()=>{})}else{v.pause()}});}
const notes=()=>document.body.classList.toggle('shownotes');
const full=()=>{try{document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()}catch(e){}};
document.getElementById('prev').onclick=()=>show(i-1);document.getElementById('next').onclick=()=>show(i+1);
document.getElementById('notesbtn').onclick=notes;document.getElementById('fs').onclick=full;
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown'].includes(e.key)||(e.key===' '&&!e.shiftKey)){e.preventDefault();show(i+1)}
  else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(i-1)}else if(e.key==='Home')show(0);else if(e.key==='End')show(S.length-1);
  else if(e.key==='n')notes();else if(e.key==='f')full();});
let x0=null;document.addEventListener('touchstart',e=>x0=e.touches[0].clientX,{passive:true});
document.addEventListener('touchend',e=>{if(x0===null)return;const d=e.changedTouches[0].clientX-x0;if(Math.abs(d)>60)show(i+(d<0?1:-1));x0=null});
show(i);
const lb=document.getElementById('launchbtn');
if(lb){lb.onclick=async()=>{const m=document.getElementById('launchmsg');lb.disabled=true;m.textContent='Launching…';
  try{const r=await fetch('/launch-ui',{method:'POST'});const j=await r.json();m.textContent=j.message||j.error||'Done';}
  catch(e){m.textContent='Could not reach the server. Run ./run-demo.sh in a terminal instead.';}
  finally{setTimeout(()=>{lb.disabled=false},4000)}};}
</script></body></html>"""
