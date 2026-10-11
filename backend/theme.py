"""Hookline visual system, shared by /pitch, /reports and the report view.

Direction: an instrument panel for attention over time. Cool neutrals with a slight blue bias, one
cobalt accent for the attention signal, semantic green / amber / red only for good / caution / loss,
a characterful display face, a quiet sans for reading and a mono for timecodes (M:SS) and numbers.
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700'
         '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500'
         '&family=Montserrat:wght@400;500;600;700;800&display=swap">')

TOKENS = """
/* Layout: one reading column (max 1180px) with a 16px+ gutter; panels only where a thing is an object. */
:root{
  --bg:#f5f6f9; --surface:#ffffff; --sunken:#eef0f4; --ink:#141821; --muted:#5a6475; --faint:#8a93a3;
  --line:#e1e5eb; --accent:#2747b5; --accent-soft:#e6ebfa;
  --good:#17803d; --warn:#b45309; --bad:#c2410c; --peak:#6d3fc0;
  --att:var(--accent); --surv:#9aa3b2; --hook:var(--good); --drop:var(--bad); --strong:var(--peak);
  --fg:var(--ink);
  --display:"Bricolage Grotesque","Avenir Next","Segoe UI",system-ui,sans-serif;
  --sans:"IBM Plex Sans","Helvetica Neue","Segoe UI",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
  --r:12px; --gap:clamp(12px,2vw,20px);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0d1016; --surface:#141922; --sunken:#1a202b; --ink:#e7eaf0; --muted:#9aa4b5; --faint:#6d7787;
  --line:#262d39; --accent:#7f9cff; --accent-soft:#1c2540; --good:#4ade80; --warn:#f59e0b; --bad:#fb7a50;
  --peak:#b394ff; --surv:#6f7889; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0d1016; --surface:#141922; --sunken:#1a202b; --ink:#e7eaf0; --muted:#9aa4b5; --faint:#6d7787;
  --line:#262d39; --accent:#7f9cff; --accent-soft:#1c2540; --good:#4ade80; --warn:#f59e0b; --bad:#fb7a50;
  --peak:#b394ff; --surv:#6f7889; color-scheme:dark}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.55 var(--sans)}
h1,h2,h3{font-family:var(--display);font-weight:700;letter-spacing:-.01em;text-wrap:balance;line-height:1.15;margin:0}
a{color:var(--accent);text-underline-offset:2px} a:focus-visible,button:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
code,.mono,.tc{font-family:var(--mono);font-variant-numeric:tabular-nums}
.muted{color:var(--muted)} .num{font-variant-numeric:tabular-nums}
.eyebrow{font:500 .74rem/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.chip{display:inline-flex;align-items:center;gap:6px;padding:2px 10px;border-radius:999px;font:500 .8rem/1.6 var(--sans);
  background:var(--sunken);color:var(--muted);border:1px solid var(--line)}
.chip.good{color:var(--good)} .chip.warn{color:var(--warn)} .chip.bad{color:var(--bad)} .chip.accent{color:var(--accent);background:var(--accent-soft);border-color:transparent}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


def page_head(title):
    return f'<meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>{title}</title>{FONTS}'


def tier_class(tag):
    """Semantic chip class for a viral-potential tag."""
    return {"Viral potential": "good", "Popular": "good", "Above average": "accent",
            "Average": "", "Low reach": "bad"}.get(tag or "", "")
