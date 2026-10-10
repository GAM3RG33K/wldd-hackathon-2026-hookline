#!/usr/bin/env python
"""Hookline terminal UI — pick a reel, analyse it, read the report.

Layout (fixed, never moves):
    header   app name + backend / Space status
    left     reels tree: demo, sample strata, your files, saved reports
    right    report: score tiles, attention strip, moments, timestamped fixes
    footer   key hints (full list behind ?)

Talks to the local backend (localhost:8000, auto-started) for previews and
straight to the HF Space for full TRIBE runs. Run:  python3 tui.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Footer, Input, Static, Tree

ROOT = Path(__file__).parent.resolve()
SAMPLES = ROOT / "samples"
OUT = ROOT / "backend" / "out"
BACKEND = "http://localhost:8000"
SPACE = os.environ.get("HOOKLINE_SPACE_URL", "https://wlddhq-hookline-tribe.hf.space")
STRATA = ("face", "faceless", "outliers")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm")
BAND_STYLE = {"Bad": "red", "Weak": "yellow", "Average": "cyan", "Strong": "green", "Viral-grade": "green"}
BLOCKS = " ▁▂▃▄▅▆▇█"
POLL_S = 1.0  # Space progress poll interval; /jobs is an in-memory lookup, cheap to hit


# ---------------------------------------------------------------- data / http

def http_json(url, token="", timeout=30, data=None, headers=None):
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=h, method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_bytes(url, token="", timeout=60):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def post_video(url, path, token="", timeout=600):
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
            f"filename=\"{Path(path).name}\"\r\nContent-Type: video/mp4\r\n\r\n").encode()
    body = head + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return http_json(url, token, timeout, data=body,
                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})


def hf_token():
    tok = os.environ.get("HF_TOKEN", "")
    if tok or not shutil.which("hf"):
        return tok
    try:
        out = subprocess.run(["hf", "auth", "token"], capture_output=True, text=True, timeout=10).stdout
        return out.strip().splitlines()[-1] if out.strip() else ""
    except Exception:
        return ""


def load_corpus():
    corpus = {}
    for s in STRATA:
        meta = SAMPLES / s / "metadata.csv"
        rows = list(csv.DictReader(meta.open())) if meta.exists() else []
        vids = {p.stem: p for p in (SAMPLES / s / "videos").glob("*.mp4")}
        corpus[s] = [{"id": r["video_id"], "path": str(vids[r["video_id"]]), "views": r.get("views", ""),
                      "stratum": r.get("page_type") or s} for r in rows if r["video_id"] in vids]
    return corpus


def saved_reports():
    jobs = [p for p in OUT.glob("*/report.json") if p.stat().st_size > 100]
    return sorted(jobs, key=lambda p: p.stat().st_mtime, reverse=True)


def short_path(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(Path(p)).replace(str(Path.home()), "~", 1)


def fmt_t(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def fmt_dur(s):
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def eta_text(p):
    """'~4m10s left (video chunk 2/3)' or 'ETA unknown, 5m over estimate' — never a fake number."""
    if p.get("eta_s") is not None:
        basis = p.get("eta_basis")
        return f"~{fmt_dur(p['eta_s'])} left" + (f" ({basis})" if basis and basis != "plan" else "")
    if p.get("overrun_s") is not None:
        return f"ETA unknown, {fmt_dur(p['overrun_s'])} over estimate"
    return "ETA unknown"


def fmt_views(v):
    try:
        n = int(v)
    except (TypeError, ValueError):
        return ""
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(n)


# -------------------------------------------------------------------- widgets

class AttentionStrip(Widget):
    """Per-second attention chart: 30% of the screen tall, full pane width.

    Bars = attention (0-100, avg second ~ 50), dots = estimated viewers remaining %,
    marker row: h hook window, ! drop zone, * peak, | scene cut. Text markers mean it
    still reads without colour."""

    DEFAULT_CSS = "AttentionStrip { height: 30vh; min-height: 10; width: 1fr; }"
    GUTTER = 5  # "100 ┤"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.report = None

    def show(self, report):
        self.report = report
        self.refresh()

    def render(self):
        r = self.report
        series = (r or {}).get("series", {})
        att = series.get("attention") or []
        if not att:
            return Text("no per-second series in this report", "dim")
        surv = series.get("survival_pct") or []
        cuts = {int(c) for c in series.get("cuts_s") or []}
        m = r.get("moments", {})
        drops = {t for a, b in m.get("drop_zones_s", []) for t in range(a, b + 1)}
        peak = m.get("peak_attention_at_s", -1)
        n = len(att)
        w = max(10, self.size.width - self.GUTTER)
        rows = max(3, self.size.height - 3)  # minus marker, axis and legend rows
        sec = [min(n - 1, int(i * n / w)) for i in range(w)]
        level = [max(0.0, min(100.0, att[s])) / 100 * rows * 8 for s in sec]
        surv_row = [rows - 1 - min(rows - 1, int(max(0.0, min(100.0, surv[s])) / 100 * rows)) if surv else -1
                    for s in sec]
        mid = rows - 1 - int(rows / 2)
        out = Text()
        for row in range(rows):
            label = "100" if row == 0 else " 50" if row == mid else "  0" if row == rows - 1 else "   "
            out.append(f"{label} ┤" if label.strip() else "    │", "dim")
            base = (rows - 1 - row) * 8
            for i, s in enumerate(sec):
                fill = int(max(0, min(8, level[i] - base)))
                bar = "red" if s in drops else "green" if s < 3 else "magenta" if s == peak else "cyan"
                if surv_row[i] == row and fill in (0, 8):  # survival dot, drawn over full bar cells too
                    out.append("•", f"bold yellow on {bar}" if fill else "bold yellow")
                elif fill:
                    out.append(BLOCKS[fill], bar)
                elif surv_row[i] == row:
                    out.append("•", "yellow")
                elif row == mid:
                    out.append("┈", "dim")
                else:
                    out.append(" ")
            out.append("\n")
        out.append(" " * (self.GUTTER - 1) + "└", "dim")
        prev = None
        for s in sec:
            first = s != prev
            ch = "*" if s == peak and first else "|" if s in cuts and first else "!" if s in drops else \
                "h" if s < 3 else "─"
            out.append(ch, "bold magenta" if ch == "*" else "bold red" if ch == "!" else "bold green" if ch == "h"
                       else "dim" if ch == "─" else "")
            prev = s
        out.append("\n")
        axis = [" "] * w
        step = max(1, round(n / max(1, w // 9) / 5) * 5) if n > 20 else 2
        for t in range(0, n + 1, step):
            lbl = fmt_t(t)
            x = min(w - len(lbl), int(t / n * w))
            if all(c == " " for c in axis[max(0, x - 1):x + len(lbl) + 1]):
                axis[x:x + len(lbl)] = list(lbl)
        out.append(" " * self.GUTTER + "".join(axis) + "\n", "dim")
        out.append(" " * self.GUTTER)
        for glyph, style, name in (("█", "cyan", "attention (avg ≈ 50)  "), ("•", "yellow", "viewers remaining %  "),
                                   ("h", "bold green", "hook 0-3s  "), ("!", "bold red", "drop  "),
                                   ("*", "bold magenta", f"peak {fmt_t(peak) if peak >= 0 else '?'}  "),
                                   ("|", "", "cut")):
            out.append(glyph, style)
            out.append(f" {name}", "dim")
        return out


class PathPrompt(ModalScreen):
    """Ask for a local video path (drag-and-drop into the terminal works)."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]
    DEFAULT_CSS = """
    PathPrompt { align: center middle; }
    PathPrompt > Vertical { width: 70; height: auto; border: round $accent; padding: 1 2; background: $panel; }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Analyse your own video[/b]\nPaste or drag a file path, then press Enter.")
            yield Input(placeholder="~/Movies/my-reel.mp4")

    @on(Input.Submitted)
    def done(self, ev: Input.Submitted):
        self.dismiss(ev.value.strip().strip("'\""))


HELP = """[b]Hookline[/b] — preview a reel's attention before posting.

[b]Try it in 3 steps[/b]
  1. Pick a reel on the left (↑/↓; Enter opens a folder)
  2. Press [b]Enter[/b] or [b]p[/b] for a preview report (seconds, runs locally)
  3. Read [b]What to change[/b]: every fix has a timestamp

[b]Keys[/b]
  p        preview analysis (local, real audio/visual/speech signals)
  f        full TRIBE brain model on the GPU Space (minutes)
  o        analyse your own video file
  r        jump to a random sample reel
  w        open the selected report in the browser (chart + PNG)
  v a t    toggle video / audio / transcript for full TRIBE runs
           (defaults: video + audio on, transcript off; off = model never loads)
  x        stop the running Space job (exits after the current item)
  ctrl+r   restart the running Space job (no re-upload, model stays loaded)
  tab      switch between reels and report
  ?        this help        q  quit

[b]Reading the scores[/b]
  Scores are within-video, uncalibrated heuristics: the average second ≈ 50.
  They are not measured retention. Views/follower bands are never used as
  inputs; views are shown only as validation context.

[b]Attention strip[/b]
  h = hook window (first 3s)   ! = drop zone   * = peak attention

Press Esc to close."""


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,q,question_mark", "dismiss", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > VerticalScroll { width: 80; max-height: 90%; border: round $accent; padding: 1 2; background: $panel; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(HELP)


# ------------------------------------------------------------------------ app

class Hookline(App):
    TITLE = "Hookline"
    CSS = """
    Screen { layout: vertical; }
    #top { height: 1; background: $panel; padding: 0 1; }
    #main { height: 1fr; }
    #reels { width: 34; border: round $panel-lighten-2; scrollbar-size-vertical: 1; }
    #reels:focus-within { border: round $accent; }
    #right { width: 1fr; border: round $panel-lighten-2; padding: 0 1; scrollbar-size-vertical: 1; }
    #right:focus-within { border: round $accent; }
    #tiles { height: 5; }
    .tile { width: 1fr; height: 5; border: round $panel-lighten-2; content-align: center middle; text-align: center; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    .section { margin-top: 1; text-style: bold; }
    """
    BINDINGS = [
        Binding("p", "preview", "Preview"),
        Binding("f", "full", "Full TRIBE"),
        Binding("o", "own_file", "Your video"),
        Binding("r", "random", "Random reel"),
        Binding("w", "web", "Web view"),
        Binding("x", "stop", "Stop job"),
        Binding("v", "toggle('video')", "Video", show=False),
        Binding("a", "toggle('audio')", "Audio", show=False),
        Binding("t", "toggle('text')", "Transcript", show=False),
        Binding("ctrl+r", "restart", "Restart job", show=False),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.corpus = load_corpus()
        self.current = None  # tree node data of the selected item
        self.by_path = {}    # video path -> job id analysed this session
        self.backend_ok = False
        self.token = ""
        self.space_run = None  # in-flight Space job: {d, sid, t0, last_ok, p}
        # Modalities for full TRIBE runs; disabled ones never load their model on the Space.
        self.mods = {"video": True, "audio": True, "text": False}

    # ---- layout
    def compose(self) -> ComposeResult:
        yield Static(id="top")
        with Horizontal(id="main"):
            yield Tree("Reels", id="reels")
            with VerticalScroll(id="right"):
                yield Static(id="title")
                with Horizontal(id="tiles"):
                    for k in ("hook", "hold", "finish", "virality"):
                        yield Static(classes="tile", id=f"t-{k}")
                yield Static("Attention per second · viewers remaining", classes="section", id="strip-h")
                yield AttentionStrip(id="strip")
                yield Static(id="moments")
                yield Static("What to change", classes="section", id="recs-h")
                yield Static(id="recs")
                yield Static(id="method")
        yield Static(id="status")
        yield Footer()

    def on_mount(self):
        self.build_tree()
        self.query_one("#reels", Tree).border_title = "1 Reels"
        self.query_one("#right").border_title = "2 Report"
        self.set_header()
        demo = OUT / "demo" / "report.json"
        if demo.exists():
            self.show_report(json.loads(demo.read_text()), "demo", "Demo · sneaker ad (precomputed)")
        self.say("Start: pick a reel on the left and press Enter.  ? for help.")
        self.query_one("#reels", Tree).focus()
        self.boot()

    def build_tree(self):
        tree = self.query_one("#reels", Tree)
        tree.clear()
        tree.root.expand()
        tree.show_root = False
        tree.root.add_leaf("★ Demo report", data={"kind": "saved", "job": "demo"})
        for s in STRATA:
            items = self.corpus[s]
            node = tree.root.add(f"{s} ({len(items)})", data={"kind": "folder"}, expand=(s == "face"))
            for v in items:
                label = Text(f"{v['id']}  ")
                label.append(f"{fmt_views(v['views'])} views", "dim")
                node.add_leaf(label, data={"kind": "reel", **v})
        if not any(self.corpus.values()):
            tree.root.add_leaf(Text("no samples/ folder found", "dim"), data={"kind": "folder"})
        self.own_node = tree.root.add("your files", data={"kind": "folder"}, expand=True)
        self.saved_node = tree.root.add("saved reports", data={"kind": "folder"})
        self.refresh_saved()
        self.call_after_refresh(setattr, tree, "cursor_line", 0)

    def refresh_saved(self):
        self.saved_node.remove_children()
        for p in saved_reports()[:30]:
            if p.parent.name == "demo":
                continue
            try:
                r = json.loads(p.read_text())
                label = Text(f"{p.parent.name}  ")
                label.append(f"hook {r['scores']['hook_pct']:.0f}", "dim")
            except Exception:
                continue
            self.saved_node.add_leaf(label, data={"kind": "saved", "job": p.parent.name})

    # ---- status line + header
    def set_header(self):
        b = Text("● up", "green") if self.backend_ok else Text("○ down", "red")
        s = Text("● token", "green") if self.token else Text("○ no token", "yellow")
        t = Text.assemble(("HOOKLINE", "bold"), ("  video in → attention preview before posting", "dim"))
        mods = Text("   full run: ")
        for key, name in (("video", "v video"), ("audio", "a audio"), ("text", "t transcript")):
            on = self.mods[key]
            mods.append(f"[{'x' if on else ' '}] {name}  ", "bold green" if on else "dim")
        right = Text.assemble(mods, "backend ", b, "   space ", s)
        if len(t) + len(right) + 3 > self.size.width:  # narrow terminal: drop the tagline
            t = Text("HOOKLINE", "bold")
        pad = max(1, self.size.width - len(t) - len(right) - 2)
        self.query_one("#top", Static).update(Text.assemble(t, " " * pad, right))

    def on_resize(self):
        self.set_header()

    def say(self, msg, style=""):
        self.query_one("#status", Static).update(Text(msg, style))

    @work(thread=True)
    def boot(self):
        """Find the token and make sure the local backend is up (starts it if not)."""
        self.token = hf_token()
        ok = self.backend_alive()
        if not ok:
            self.call_from_thread(self.say, "Starting local backend on :8000 …")
            log = open("/tmp/hookline_backend.log", "a")
            subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--port", "8000"],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            for _ in range(30):
                time.sleep(1)
                if self.backend_alive():
                    ok = True
                    break
        self.backend_ok = ok
        self.call_from_thread(self.set_header)
        if not ok:
            self.call_from_thread(self.say, "Backend failed to start; see /tmp/hookline_backend.log", "red")

    def backend_alive(self):
        try:
            return bool(http_json(BACKEND + "/health", timeout=3).get("ok"))
        except Exception:
            return False

    # ---- report pane
    def show_report(self, r, job, title):
        self.current_job = job
        s = r["scores"]
        mode = "preview (synthetic brain, real signals)" if r.get("preview", True) and "full" not in str(
            r.get("neural_source", "")) else "full TRIBE"
        self.query_one("#title", Static).update(Text.assemble((title, "bold"), ("   " + mode, "dim")))

        def tile(id_, name, value, band=""):
            t = Text.assemble((name + "\n", "dim"), (value, "bold"))
            if band:
                t.append("\n" + band, BAND_STYLE.get(band, ""))
            self.query_one(f"#t-{id_}", Static).update(t)

        tile("hook", "HOOK", f"{s['hook_pct']:.0f}", s.get("hook_band", ""))
        tile("hold", "HOLD", f"{s['hold_rate_pct']:.0f}%")
        tile("finish", "FINISH", f"{s['finish_pct']:.0f}%")
        tile("virality", "VIRALITY", f"{s['virality']:.0f}", s.get("virality_band", ""))
        self.query_one("#strip", AttentionStrip).show(r)
        m = r["moments"]
        sw, ww = m["strongest_window_s"], m["weakest_window_s"]
        drops = ", ".join(f"{fmt_t(a)}-{fmt_t(b)}" for a, b in m.get("drop_zones_s", [])) or "none"
        self.query_one("#moments", Static).update(Text.assemble(
            ("peak ", "dim"), fmt_t(m["peak_attention_at_s"]),
            ("   strongest ", "dim"), f"{fmt_t(sw[0])}-{fmt_t(sw[1])}",
            ("   weakest ", "dim"), f"{fmt_t(ww[0])}-{fmt_t(ww[1])}",
            ("   drops ", "dim"), drops))
        tbl = Table(expand=True, box=None, show_header=True, header_style="dim", pad_edge=False)
        tbl.add_column("AT", style="bold green", no_wrap=True, width=11)
        tbl.add_column("ISSUE", ratio=2)
        tbl.add_column("FIX", ratio=3)
        for x in r["recommendations"]:
            tbl.add_row(x["at"], x["issue"], x["fix"])
        self.query_one("#recs", Static).update(tbl)
        self.query_one("#method", Static).update(Text(
            f"\n{r.get('method', '')} · neural: {r.get('neural_source', '?')}", "dim italic"))
        for w in ("#tiles", "#strip-h", "#strip", "#moments", "#recs-h", "#recs", "#method"):
            self.query_one(w).display = True

    def show_reel_card(self, d):
        self.current_job = None
        for w in ("#tiles", "#strip-h", "#strip", "#moments", "#recs-h", "#recs", "#method"):
            self.query_one(w).display = False
        views = f"{fmt_views(d.get('views'))} views · " if d.get("views") else ""
        self.query_one("#title", Static).update(Text.assemble(
            (d["id"], "bold"), (f"   {d.get('stratum', '')}\n\n", "dim"),
            (f"{views}{short_path(d['path'])}\n", "dim"),
            ("Views are validation context only; scoring judges the video, never the account.\n\n", "dim italic"),
            ("Enter / p", "bold"), "  preview report (seconds)\n",
            ("f", "bold"), "          full TRIBE brain model on GPU Space (minutes)\n",
            ("v a t", "bold"), "      toggle video / audio / transcript for the full run"))

    @on(Tree.NodeHighlighted)
    def highlighted(self, ev: Tree.NodeHighlighted):
        d = ev.node.data or {}
        self.current = d if d.get("kind") in ("reel", "file", "saved") else self.current
        if d.get("kind") == "saved":
            p = OUT / d["job"] / "report.json"
            if p.exists():
                title = "Demo · sneaker ad (precomputed)" if d["job"] == "demo" else f"Saved report {d['job']}"
                self.show_report(json.loads(p.read_text()), d["job"], title)
        elif d.get("kind") in ("reel", "file"):
            run = getattr(self, "space_run", None)
            if run and run["d"]["path"] == d["path"]:
                self.show_progress(run, time.time() - run["last_ok"])
                return
            job = self.by_path.get(d["path"])
            if job and (OUT / job / "report.json").exists():
                self.show_report(json.loads((OUT / job / "report.json").read_text()), job, d["id"])
            else:
                self.show_reel_card(d)

    @on(Tree.NodeSelected)
    def selected(self, ev: Tree.NodeSelected):
        if (ev.node.data or {}).get("kind") in ("reel", "file"):
            self.action_preview()

    # ---- actions
    def target(self):
        d = self.current
        if not d or d.get("kind") not in ("reel", "file"):
            self.say("Select a reel or your own file first (o adds one).", "yellow")
            return None
        return d

    def action_preview(self):
        d = self.target()
        if d:
            self.run_preview(d)

    def action_full(self):
        d = self.target()
        if not d:
            return
        if not self.token:
            self.say("No HF token: run `hf auth login` or export HF_TOKEN, then restart.", "yellow")
            return
        if self.space_run:
            self.say(f"Space is busy with {self.space_run['d']['id']}; one full run at a time.", "yellow")
            return
        self.run_space(d)

    @work(thread=True, exclusive=True, group="analyse")
    def run_preview(self, d):
        if not self.backend_ok and not self.backend_alive():
            self.call_from_thread(self.say, "Backend not up yet, try again in a moment.", "yellow")
            return
        self.call_from_thread(self.say, f"Analysing {d['id']} (preview) …")
        t0 = time.time()
        try:
            res = post_video(BACKEND + "/analyze?mode=preview", d["path"])
        except urllib.error.HTTPError as e:
            self.call_from_thread(self.say, f"Preview failed: {e.read().decode()[:200]}", "red")
            return
        except Exception as e:
            self.call_from_thread(self.say, f"Preview failed: {e}", "red")
            return
        job = res["job_id"]
        self.by_path[d["path"]] = job
        self.call_from_thread(self.show_report, res["report"], job, d["id"])
        self.call_from_thread(self.refresh_saved)
        self.call_from_thread(self.say, f"Preview ready in {time.time() - t0:.0f}s · job {job} · w opens web view",
                              "green")

    @work(thread=True, group="space")
    def run_space(self, d):
        """Upload to the Space, then poll its progress API every POLL_S until done/failed."""
        tok = self.token
        mb = Path(d["path"]).stat().st_size / 1e6
        self.call_from_thread(self.say, f"Uploading {d['id']} ({mb:.1f}MB) to GPU Space …")
        try:
            q = "&".join(f"{k}={int(v)}" for k, v in self.mods.items())
            res = post_video(f"{SPACE}/analyze?{q}", d["path"], token=tok)
        except Exception as e:
            self.call_from_thread(self.say, f"Space upload failed: {e}", "red")
            return
        sid = res["job_id"]
        run = {"d": d, "sid": sid, "t0": time.time(), "last_ok": time.time(), "mods": dict(self.mods), "p": {
            "status": "queued", "stage_label": "Waiting for the GPU", "step": 0, "steps": 4,
            "eta_s": res.get("estimate_s"), "video_s": res.get("video_s")}}
        self.space_run = run
        self.call_from_thread(self.refresh_progress)
        while True:
            time.sleep(POLL_S)
            sid = run["sid"]  # ctrl+r swaps in the restarted job's id
            try:
                r = http_json(f"{SPACE}/report/{sid}", tok, timeout=10)
                run["last_ok"] = time.time()
                if sid != run["sid"]:
                    continue  # restarted while this poll was in flight
            except Exception:
                self.call_from_thread(self.refresh_progress)
                if time.time() - run["t0"] > 3 * 3600:
                    self.call_from_thread(self.say, f"Gave up after 3h; Space job {sid}", "red")
                    return
                continue
            if "scores" in r:
                job = f"s-{sid}"
                jd = OUT / job
                jd.mkdir(parents=True, exist_ok=True)
                r["preview"] = False
                (jd / "report.json").write_text(json.dumps(r, indent=1))
                for ext in ("png", "md"):
                    try:
                        (jd / f"report.{ext}").write_bytes(http_bytes(f"{SPACE}/report/{sid}.{ext}", tok))
                    except Exception:
                        pass
                self.by_path[d["path"]] = job
                self.space_run = None
                el = time.time() - run["t0"]
                self.call_from_thread(self.show_report, r, job, d["id"])
                self.call_from_thread(self.refresh_saved)
                self.call_from_thread(self.say, f"Full TRIBE report ready in {fmt_dur(el)} · saved as {job}", "green")
                return
            run["p"] = r
            self.call_from_thread(self.refresh_progress)
            if r.get("status") == "cancelled":
                self.space_run = None
                self.call_from_thread(self.say, f"Stopped Space job {sid} in {r.get('stage')}. "
                                                "f starts a new run.", "yellow")
                return
            if r.get("status") == "failed":
                self.space_run = None
                self.call_from_thread(self.say, f"Space job failed in {r.get('stage')}: {r.get('error', '?')[:140]}",
                                      "red")
                return

    def refresh_progress(self):
        """Status bar always; the report pane too while the running item is selected."""
        run = getattr(self, "space_run", None)
        if not run:
            return
        p, d = run["p"], run["d"]
        silent = time.time() - run["last_ok"]
        el = p.get("elapsed_s", time.time() - run["t0"])
        sub = p.get("substep") or {}
        where = f"{sub['name']} {fmt_dur(sub.get('elapsed_s'))}" if sub else p.get("stage_label", "")
        loops = p.get("loops") or []
        if loops:
            lp = loops[-1]
            where += f" [{lp['name']} {lp.get('n', 0)}/{lp.get('total') if lp.get('total') is not None else '?'}]"
        msg = (f"Space {run['sid']} · step {p.get('step', 0)}/{p.get('steps', 4)} {where} · "
               f"{fmt_dur(el)} elapsed · {eta_text(p)} · timeout {fmt_dur(p.get('timeout_s'))}")
        style = ""
        if silent > 30:
            msg = f"Space not responding for {fmt_dur(silent)} (last: {p.get('stage_label', '?')}) · " + msg
            style = "yellow"
        if p.get("overdue"):
            style = "yellow"
            msg = "Over timeout · " + msg
        if p.get("status") == "failed":
            style, msg = "red", f"Space job failed in {p.get('stage')}: {p.get('error', '?')[:140]}"
        self.say(msg, style)
        if self.current and self.current.get("path") == d["path"]:
            self.show_progress(run, silent)

    def show_progress(self, run, silent=0.0):
        p, d = run["p"], run["d"]
        self.current_job = None
        for w in ("#tiles", "#strip-h", "#strip", "#moments", "#recs-h", "#recs", "#method"):
            self.query_one(w).display = False
        el = p.get("elapsed_s", time.time() - run["t0"])
        eta = p.get("eta_s")
        if p.get("status") == "done":
            frac = 1.0
        elif eta is None:  # no live signal: fall back to the step count, never claim 100%
            frac = min(0.95, max(0, p.get("step", 1) - 1) / max(1, p.get("steps", 4)))
        else:
            frac = el / max(1, el + eta)
        width = 40
        filled = int(frac * width)
        mods = p.get("modalities") or run.get("mods") or {}
        used = " + ".join(k for k, v in mods.items() if v) or "?"
        t = Text.assemble((d["id"], "bold"), (f"   full TRIBE on GPU Space · job {run['sid']} · {used}\n\n", "dim"))
        t.append("━" * filled, "cyan")
        t.append("━" * (width - filled), "dim")
        t.append(f"  {frac * 100:3.0f}%\n", "bold")
        t.append(f"{fmt_dur(el)} elapsed · {eta_text(p)} · timeout {fmt_dur(p.get('timeout_s'))}"
                 f" · video {p.get('video_s', '?')}s\n\n", "dim")
        cur = p.get("stage")
        for i, st in enumerate(p.get("plan") or []):
            if st.get("took_s") is not None:
                mark, style, right = "✓", "green", f"took {fmt_dur(st['took_s'])}"
            elif st["key"] == cur and p.get("status") == "running":
                mark, style = "▶", "bold cyan"
                right = f"{fmt_dur(p.get('stage_elapsed_s'))} of ~{fmt_dur(st['est_s'])}"
            elif st["key"] == cur and p.get("status") == "failed":
                mark, style, right = "✗", "bold red", "failed"
            else:
                mark, style, right = "·", "dim", f"~{fmt_dur(st['est_s'])}" if st["est_s"] else "skip (cached)"
            t.append(f" {mark} {i + 1}. {st['label']:<55}", style)
            t.append(f"{right}\n", "dim")
        if not p.get("plan"):
            t.append(f" · {p.get('stage_label', 'Waiting for the GPU')}\n", "dim")
        sub = p.get("substep")
        if sub:
            t.append(f"\n   now: {sub['name']} · {fmt_dur(sub.get('elapsed_s'))}\n", "cyan")
        for loop in p.get("loops") or []:
            n, total = loop.get("n", 0), loop.get("total")
            width = 20
            done = int(width * n / total) if total else 0
            t.append(f"     {loop['name']:<30}", "dim")
            t.append("█" * done, "cyan")
            t.append("░" * (width - done), "dim")
            t.append(f" {n}/{total if total is not None else '?'}", "bold")
            extra = f" · {loop['per_item_s']}s each" if loop.get("per_item_s") else ""
            if loop.get("eta_s") is not None:
                extra += f" · ~{fmt_dur(loop['eta_s'])} left"
            t.append(f" · {fmt_dur(loop.get('elapsed_s'))}{extra}\n", "dim")
        chunks = p.get("chunks") or []
        done = {c["i"]: c for c in p.get("chunks_done") or []}
        cur = (p.get("video_chunk") or {}).get("i")
        for kind in ("Video", "Audio"):
            spans = [c for c in chunks if c["type"] == kind]
            if not spans:
                continue
            t.append(f"\n   {kind.lower()} chunks ({len(spans)}): ", "dim")
            for i, c in enumerate(spans, 1):
                rng = f"{fmt_t(c['start'])}-{fmt_t(c['start'] + c['duration'])}"
                if kind == "Video" and i in done:
                    t.append(f"✓ {rng} {fmt_dur(done[i]['took_s'])}  ", "green")
                elif kind == "Video" and i == cur and p.get("status") == "running":
                    t.append(f"▶ {rng}  ", "bold cyan")
                else:
                    t.append(f"· {rng}  ", "dim")
        if chunks:
            t.append("\n")
        subs = p.get("substeps") or []
        if subs:
            longest = max(x["took_s"] for x in subs) or 1
            total = sum(x["took_s"] for x in subs) or 1
            t.append("\nTime per sub-step\n", "bold")
            for x in subs:
                bar = "█" * max(1, int(20 * x["took_s"] / longest))
                t.append(f"   {x['step'] + '/' + x['name']:<38}", "dim")
                t.append(f"{bar:<21}", "yellow" if x["took_s"] == longest else "cyan")
                t.append(f"{x['took_s']:>6.1f}s {100 * x['took_s'] / total:>3.0f}%\n", "dim")
        if silent > 30:
            t.append(f"\n! Space not answering for {fmt_dur(silent)}; it may be out of memory. "
                     "Still polling.\n", "yellow")
        if p.get("overdue"):
            t.append("\n! Past the estimated timeout. The Space may be stuck; consider restarting it.\n", "yellow")
        mem = p.get("mem") or {}
        if mem:
            t.append(f"\nRAM free {mem.get('ram_avail_gb', '?')}/{mem.get('ram_total_gb', '?')} GB · "
                     f"process {mem.get('rss_gb', '?')} GB · GPU {mem.get('gpu_gb', '?')} GB\n", "dim")
        log = p.get("log") or []
        if log:
            t.append("\nSpace log\n", "bold")
            for line in log[-6:]:
                t.append(line[:150] + "\n", "dim")
        self.query_one("#title", Static).update(t)

    def action_toggle(self, key):
        new = dict(self.mods, **{key: not self.mods[key]})
        if not any(new.values()):
            self.say("Keep at least one of video, audio, transcript on.", "yellow")
            return
        self.mods = new
        self.set_header()
        on = " + ".join(k for k, v in self.mods.items() if v)
        self.say(f"Full TRIBE runs will use: {on}. Models for the others are never loaded.")

    def action_stop(self):
        run = self.space_run
        if not run:
            self.say("No Space job running.", "yellow")
            return
        self.space_call("cancel", run["sid"])

    def action_restart(self):
        run = self.space_run
        if not run:
            self.say("No Space job running; press f to start one.", "yellow")
            return
        self.space_call("restart", run["sid"])

    @work(thread=True, group="space-ctl")
    def space_call(self, verb, sid):
        try:
            r = http_json(f"{SPACE}/jobs/{sid}/{verb}", self.token, timeout=20, data=b"")
        except urllib.error.HTTPError as e:
            self.call_from_thread(self.say, f"{verb} failed: {e.read().decode()[:160]}", "red")
            return
        except Exception as e:
            self.call_from_thread(self.say, f"{verb} failed: {e}", "red")
            return
        run = self.space_run
        if verb == "cancel":
            self.call_from_thread(self.say, f"Stopping {sid} after the current item …", "yellow")
        elif run:
            run.update(sid=r["job_id"], t0=time.time(), last_ok=time.time(), p={
                "status": "queued", "stage_label": f"Restarted (was {sid}); waiting for the GPU",
                "step": 0, "steps": 4, "eta_s": r.get("estimate_s"), "video_s": r.get("video_s")})
            self.call_from_thread(self.refresh_progress)
            self.call_from_thread(self.say, f"Restarted as {r['job_id']} (same upload, model stays loaded)", "green")

    def action_own_file(self):
        def add(raw):
            if not raw:
                return
            p = Path(os.path.expanduser(raw))
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
                self.say(f"Not a video file: {raw}", "red")
                return
            d = {"kind": "file", "id": p.stem, "path": str(p), "stratum": "your file"}
            node = self.own_node.add_leaf(p.name, data=d)
            self.own_node.expand()
            self.query_one("#reels", Tree).move_cursor(node)
            self.current = d
            self.show_reel_card(d)
            self.say("Added. Press p for preview or f for full TRIBE.")

        self.push_screen(PathPrompt(), add)

    def action_random(self):
        import random
        tree = self.query_one("#reels", Tree)
        leaves = [n for folder in tree.root.children for n in folder.children
                  if (n.data or {}).get("kind") == "reel"]
        if not leaves:
            self.say("No sample reels found in samples/.", "yellow")
            return
        node = random.choice(leaves)
        node.parent.expand()
        tree.move_cursor(node)
        tree.focus()

    def action_web(self):
        job = getattr(self, "current_job", None)
        if not job:
            self.say("No report selected.", "yellow")
            return
        webbrowser.open(f"{BACKEND}/report/{job}/view")
        self.say(f"Opened {BACKEND}/report/{job}/view")

    def action_help(self):
        self.push_screen(HelpScreen())


if __name__ == "__main__":
    Hookline().run()
