#!/usr/bin/env python
"""Hookline terminal UI — pick a reel, analyse it, read the report.

Layout (fixed, never moves):
    header   app name + backend / Space status
    left     reels tree: demo, sample strata, your files, saved reports
    right    report: score tiles, attention strip, moments, timestamped fixes
    footer   key hints (full list behind ?)

Talks to the local backend (localhost:8000, auto-started) for saved reports and
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
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Footer, Input, Static, TextArea, Tree

ROOT = Path(__file__).parent.resolve()
SAMPLES = ROOT / "samples"
OUT = ROOT / "backend" / "out"
BACKEND = "http://localhost:8000"
SPACE = os.environ.get("HOOKLINE_SPACE_URL", "https://wlddhq-hookline-tribe.hf.space")
STRATA = ("face", "faceless", "outliers")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm")
BAND_STYLE = {"Bad": "red", "Weak": "yellow", "Average": "cyan", "Strong": "green", "Viral-grade": "green",
              "Low reach": "red", "Above average": "green", "Popular": "green", "Viral potential": "bold green"}
BLOCKS = " ▁▂▃▄▅▆▇█"
POLL_S = 1.0  # Space progress poll interval; /jobs is an in-memory lookup, cheap to hit
SHOW_SUGGESTIONS = True  # evidence-based editing suggestions (tribe/suggest.py), not the old templates


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


def post_video(url, path, token="", timeout=600, fields=None):
    """Multipart upload: the video plus optional text form fields (e.g. caption)."""
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in (fields or {}).items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
            f"filename=\"{Path(path).name}\"\r\nContent-Type: video/mp4\r\n\r\n").encode()
    body += head + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
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


# ---------------------------------------------------------------- Space control (HF Hub REST, no extra deps)

SPACE_REPO = os.environ.get("HOOKLINE_SPACE_REPO", "wlddhq/hookline-tribe")
HF_API = f"https://huggingface.co/api/spaces/{SPACE_REPO}"
# HF runtime stage -> (glyph, word, style) for the header + Space modal
STAGE_VIEW = {"RUNNING": ("●", "live", "bold green"), "RUNNING_BUILDING": ("●", "live, rebuilding", "green"),
              "RUNNING_APP_STARTING": ("◐", "restarting", "yellow"), "APP_STARTING": ("◐", "starting", "yellow"),
              "BUILDING": ("◐", "building", "yellow"), "PAUSED": ("○", "paused", "red"),
              "SLEEPING": ("○", "sleeping", "red"), "STOPPED": ("○", "stopped", "red"),
              "BUILD_ERROR": ("✕", "build error", "bold red"), "RUNTIME_ERROR": ("✕", "runtime error", "bold red"),
              "CONFIG_ERROR": ("✕", "config error", "bold red"), "NO_APP_FILE": ("✕", "no app file", "bold red")}


def space_runtime(token):
    """{stage, hardware, requested, sleep_s} from the HF Hub (works while the Space is paused)."""
    r = http_json(f"{HF_API}/runtime", token, timeout=10)
    hw = r.get("hardware") or {}
    return {"stage": r.get("stage", "?"), "hardware": hw.get("current"), "requested": hw.get("requested"),
            "sleep_s": r.get("gcTimeout"), "error": (r.get("errorMessage") or "")[:200]}


def space_ping(token):
    """GET /health on the Space itself. Wakes a sleeping Space; returns (ok, ms or error)."""
    t0 = time.time()
    try:
        ok = bool(http_json(f"{SPACE}/health", token, timeout=15).get("ok", True))
        return ok, f"{(time.time() - t0) * 1000:.0f} ms"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, type(e).__name__


def space_control(verb, token):
    """verb: restart (also resumes a paused Space) | pause. Billing starts on restart."""
    return http_json(f"{HF_API}/{verb}", token, timeout=20, data=b"")


def load_corpus():
    corpus = {}
    for s in STRATA:
        meta = SAMPLES / s / "metadata.csv"
        rows = list(csv.DictReader(meta.open())) if meta.exists() else []
        vids = {p.stem: p for p in (SAMPLES / s / "videos").glob("*.mp4")}
        corpus[s] = [{"id": r["video_id"], "path": str(vids[r["video_id"]]), "views": r.get("views", ""),
                      "followers": r.get("follower_band", ""),
                      "stratum": r.get("page_type") or s} for r in rows if r["video_id"] in vids]
    return corpus


# Page size, shown next to views as context only (never a scoring input: judge the video).
# Plain follower ranges read without a legend; the size word is for the reel card.
PAGE_SIZE = {"10k-50k": "small creator", "50k-200k": "medium page", "200k-1M": "big creator", "1M+": "top creator"}


def followers_text(band):
    """'50k-200k' -> '50k–200k followers'; unknown -> ''."""
    band = (band or "").strip()
    return f"{band.replace('-', '–')} followers" if band else ""


def page_size(band):
    return PAGE_SIZE.get((band or "").strip(), "unknown page size")


def analysed_at(report_json):
    """When the reel was analysed: the model output's time, not report.json's (re-scoring rewrites that)."""
    d = report_json.parent
    for f in (d / "raw" / "preds.npy", d / "preds.npy", report_json):  # preds are never rewritten
        if f.exists():
            return f.stat().st_mtime
    return 0.0


def saved_reports():
    """Saved reports, latest analysis first (always, in the TUI)."""
    jobs = [p for p in OUT.glob("*/report.json") if p.stat().st_size > 100]
    return sorted(jobs, key=analysed_at, reverse=True)


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


EMO_STYLE = {'amusement': '#f5b301', 'excitement': '#f97316', 'awe': '#7c3aed', 'admiration': '#a78bfa', 'adoration': '#f9a8d4', 'satisfaction': '#2dd4bf', 'craving': '#d97706', 'surprise': '#ec4899', 'interest': '#06b6d4', 'nostalgia': '#c4b5fd', 'romance': '#e11d48', 'hope': '#84cc16', 'empathic_pain': '#fb7185', 'sadness': '#3b82f6', 'fear': '#ef4444', 'anger': '#b91c1c', 'disgust': '#65a30d', 'awkwardness': '#a3a3a3', 'moved': '#22c55e', 'laughter': '#f5b301', 'outrage': '#b91c1c', 'sorrow': '#3b82f6'}
AUDIO_TYPES = {"song": "song", "speech_over_music": "speech over music", "speech": "speech",
               "instrumental": "instrumental music", "humming": "humming / chant", "other": "no clear speech"}


def audio_text(audio, gate=None):
    """One line: audio type, matched song + trending list, reuse across reels, transcript decision."""
    audio, gate = audio or {}, gate or {}
    kind = AUDIO_TYPES.get(audio.get("type") or gate.get("audio_type"), "")
    if not kind and not gate:
        return None
    t = Text("audio: ", "dim")
    t.append(kind or "?", "bold")
    song = audio.get("song")
    if song:
        t.append(f" · ♪ {song['title']} - {song['artist']}", "bold magenta")
        t.append(f" (trending list: {song.get('list', '?')})", "magenta")
    reused = audio.get("reused_by") or audio.get("reuse") or []
    if reused:
        t.append(f" · same audio in {len(reused)} other reel(s)", "yellow")
    if gate.get("decision"):
        t.append(f" · {gate['decision']}", "green" if gate["decision"].endswith("on") else "yellow")
    return t


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
        # emotion dot on top of each bar: colour = dominant emotion (non-neutral, some arousal)
        em = (r.get("emotion") or {}).get("series") or {}
        dom, aro = em.get("dominant") or [], em.get("arousal") or []
        top_row = [rows - 1 - min(rows - 1, int(level[i] // 8)) for i in range(w)]
        first_col = {}
        for i, s in enumerate(sec):
            first_col.setdefault(s, i)
        dot_at = {}
        for s, i in first_col.items():
            if s < len(dom) and dom[s] not in ("neutral", "calm") and (aro[s] if s < len(aro) else 0) > 0.25:
                dot_at[i] = EMO_STYLE.get(dom[s], "white")
        out = Text()
        for row in range(rows):
            label = "100" if row == 0 else " 50" if row == mid else "  0" if row == rows - 1 else "   "
            out.append(f"{label} ┤" if label.strip() else "    │", "dim")
            base = (rows - 1 - row) * 8
            for i, s in enumerate(sec):
                fill = int(max(0, min(8, level[i] - base)))
                bar = "red" if s in drops else "green" if s < 3 else "magenta" if s == peak else "cyan"
                if i in dot_at and row == top_row[i]:
                    out.append("●", f"bold {dot_at[i]}")
                elif surv_row[i] == row and fill in (0, 8):  # survival dot, drawn over full bar cells too
                    out.append("•", f"bold white on {bar}" if fill else "grey70")
                elif fill:
                    out.append(BLOCKS[fill], bar)
                elif surv_row[i] == row:
                    out.append("•", "grey70")
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
        for glyph, style, name in (("█", "cyan", "attention (avg ≈ 50)  "), ("•", "grey70", "viewers remaining %  "),
                                   ("h", "bold green", "hook 0-3s  "), ("!", "bold red", "drop  "),
                                   ("*", "bold magenta", f"peak {fmt_t(peak) if peak >= 0 else '?'}  "),
                                   ("|", "", "cut  "), ("●", "bold #f5b301", "emotion (colour = type)")):
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


class SpaceScreen(ModalScreen):
    """Live status of the GPU Space + ping / start / restart / stop, without leaving the TUI.

    Stop and restart kill any running job, so both ask for a second press within 5s.
    """

    BINDINGS = [Binding("escape,S", "dismiss", "Close"), Binding("p", "do('ping')", "Ping"),
                Binding("s", "do('start')", "Start"), Binding("r", "do('restart')", "Restart"),
                Binding("x", "do('stop')", "Stop")]
    DEFAULT_CSS = """
    SpaceScreen { align: center middle; }
    SpaceScreen > Vertical { width: 76; height: auto; border: round $accent; padding: 1 2; background: $panel; }
    SpaceScreen #sp-body { height: auto; }
    SpaceScreen #sp-buttons { height: 3; margin-top: 1; }
    SpaceScreen Button { min-width: 12; margin-right: 1; }
    SpaceScreen #sp-msg { height: auto; margin-top: 1; }
    """
    CONFIRM = {"stop": "pause the Space (stops billing, kills running jobs)",
               "restart": "restart the Space (kills running jobs, reloads models ~3-5 min)"}

    def __init__(self):
        super().__init__()
        self.armed = None  # (verb, time) waiting for its confirming second press

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="sp-body")
            with Horizontal(id="sp-buttons"):
                yield Button("Ping  p", id="ping")
                yield Button("Start  s", id="start", variant="success")
                yield Button("Restart  r", id="restart", variant="warning")
                yield Button("Stop  x", id="stop", variant="error")
            yield Static(id="sp-msg")

    def on_mount(self):
        self.query_one(Vertical).border_title = f"GPU Space · {SPACE_REPO}"
        self.query_one(Vertical).border_subtitle = "esc close"
        self.render_status()
        self.app.poll_space(ping=True)
        self.set_interval(2, lambda: self.app.poll_space(ping=False))  # faster while open
        self.set_interval(1, self.render_status)

    def render_status(self):
        st = self.app.space_rt
        stage = st.get("stage")
        glyph, word, style = STAGE_VIEW.get(stage, ("?", (stage or "unknown").lower(), "dim"))
        t = Text()
        t.append(f"{glyph} {word.upper()}\n", style)
        rows = [("stage", stage or "—"), ("hardware", st.get("hardware") or f"none (requested {st.get('requested') or '?'})"),
                ("sleep after", fmt_dur(st.get("sleep_s")) + " idle" if st.get("sleep_s") else "—"),
                ("health", st.get("health") or "not pinged"),
                ("jobs", f"{st['jobs']} running/queued" if st.get("jobs") is not None else "—"),
                ("checked", f"{time.time() - st['checked']:.0f}s ago" if st.get("checked") else "—")]
        for k, v in rows:
            t.append(f"{k:<12}", "dim")
            t.append(f"{v}\n")
        if st.get("error"):
            t.append(f"\n{st['error']}\n", "red")
        if st.get("fetch_error"):
            t.append(f"\nstatus check failed: {st['fetch_error']}\n", "yellow")
        running = stage in ("RUNNING", "RUNNING_BUILDING")
        t.append("\nBilling runs while live (a100-large ≈ $2.50/hr).", "dim")
        self.query_one("#sp-body", Static).update(t)
        self.query_one("#start", Button).disabled = running or stage in ("BUILDING", "APP_STARTING")
        self.query_one("#stop", Button).disabled = stage in ("PAUSED", "STOPPED")
        if self.armed and time.time() - self.armed[1] > 5:
            self.armed = None
            self.note("")

    def note(self, msg, style=""):
        self.query_one("#sp-msg", Static).update(Text(msg, style))

    @on(Button.Pressed)
    def pressed(self, ev: Button.Pressed):
        self.action_do(ev.button.id)

    def action_do(self, verb):
        if not self.app.token:
            self.note("No HF token: run `hf auth login` or export HF_TOKEN.", "yellow")
            return
        if self.query_one(f"#{verb}", Button).disabled:
            self.note(f"{verb} not available in the current stage.", "dim")
            return
        if verb in self.CONFIRM:
            if not (self.armed and self.armed[0] == verb):
                self.armed = (verb, time.time())
                jobs = self.app.space_rt.get("jobs")
                warn = f"  {jobs} job(s) active!" if jobs else ""
                self.note(f"Press {verb} again within 5s to {self.CONFIRM[verb]}.{warn}", "bold yellow")
                return
            self.armed = None
        self.note(f"{verb} …", "cyan")
        self.app.space_action(verb, self)


class CaptionArea(TextArea):
    """Multi-line caption field with wrapping: Enter submits; pasted line breaks are kept."""

    class Submitted(Message):
        pass

    async def _on_key(self, event):
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted())
            return
        await super()._on_key(event)


class CaptionPrompt(ModalScreen):
    """Optional caption before an analysis. Enter = go (empty = no caption), Esc = cancel / just the report."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]
    DEFAULT_CSS = """
    CaptionPrompt { align: center middle; }
    CaptionPrompt > Vertical { width: 84; height: auto; border: round $accent; padding: 1 2; background: $panel; }
    CaptionPrompt CaptionArea { margin-top: 1; height: 7; }
    CaptionPrompt #cap-count { height: 1; color: $text-muted; text-align: right; }
    """
    MAX = 2200  # Instagram's caption limit (the Space truncates beyond it)

    def __init__(self, name, prefill="", esc_hint="cancels"):
        super().__init__()
        self.reel, self.prefill, self.esc_hint = name, prefill, esc_hint

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text.assemble(("Caption for ", ""), (self.reel, "bold"), (" (optional)\n", ""),
                                       ("Paste the post caption, or leave it empty. It is only used if it describes this "
                                        "video (checked against its frames and speech); an unrelated or filler caption "
                                        f"is ignored. Enter analyzes on the GPU, Esc {self.esc_hint}.", "dim")))
            yield CaptionArea(self.prefill, soft_wrap=True, show_line_numbers=False, highlight_cursor_line=False,
                              placeholder="e.g. Rs 200 ka cleaning hack jo har ghar mein hona chahiye #amazonfinds")
            yield Static(id="cap-count")

    def on_mount(self):
        self.query_one(Vertical).border_title = "Analyze"
        area = self.query_one(CaptionArea)
        area.focus()
        area.move_cursor(area.document.end)
        self.count()

    @on(TextArea.Changed)
    def count(self, _=None):
        n = len(self.query_one(CaptionArea).text)
        self.query_one("#cap-count", Static).update(
            Text(f"{n}/{self.MAX}", "bold red" if n > self.MAX else "dim"))

    @on(CaptionArea.Submitted)
    def done(self, _):
        self.dismiss(self.query_one(CaptionArea).text.strip()[:self.MAX])


class RunOptions(ModalScreen):
    """What a full TRIBE run uses. Disabled modalities never load their model on the Space."""

    BINDINGS = [Binding("escape,m,enter", "dismiss", "Close"), Binding("v", "toggle('video')", "Video"),
                Binding("a", "toggle('audio')", "Audio"), Binding("t", "toggle('text')", "Transcript"),
                Binding("l", "lang", "Language")]
    DEFAULT_CSS = """
    RunOptions { align: center middle; }
    RunOptions > Static { width: 64; height: auto; border: round $accent; padding: 1 2; background: $panel; }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="ro-body")

    def on_mount(self):
        self.query_one("#ro-body").border_title = "Run options"
        self.query_one("#ro-body").border_subtitle = "esc close"
        self.render_opts()

    def render_opts(self):
        app = self.app
        t = Text()
        for key, label, name in (("v", "video", "video"), ("a", "audio", "audio")):
            on = app.mods[name]
            t.append(f" {key} ", "bold reverse")
            t.append(f"  [{'x' if on else ' '}] {label}\n", "bold green" if on else "dim")
        tm = app.text_mode
        t.append(" t ", "bold reverse")
        t.append(f"  [{ {'auto': 'a', 'on': 'x', 'off': ' '}[tm] }] transcript: {tm}\n",
                 "bold green" if tm != "off" else "dim")
        t.append(" l ", "bold reverse")
        t.append(f"  language: {app.lang}  (auto = Hindi / English / Hinglish)\n\n", "bold")
        t.append("Transcript auto: a speech gate decides per reel; songs and humming are\n"
                 "scored on visuals + sound. Models for disabled inputs are never loaded.", "dim")
        self.query_one("#ro-body", Static).update(t)

    def action_toggle(self, key):
        self.app.action_toggle(key)
        self.render_opts()

    def action_lang(self):
        self.app.action_cycle_lang()
        self.render_opts()


def pick_video_file():
    """System file picker for a video. Returns a path, "" if cancelled, or None if no picker exists."""
    try:
        if sys.platform == "darwin":
            script = ['tell application "System Events"', "activate",
                      'set f to choose file with prompt "Pick a reel to analyze" '
                      'of type {"public.movie", "public.mpeg-4", "com.apple.quicktime-movie"}',
                      "end tell", "POSIX path of f"]
            r = subprocess.run(["osascript", *sum((["-e", line] for line in script), [])],
                               capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else ""  # non-zero = cancelled
        if shutil.which("zenity"):
            r = subprocess.run(["zenity", "--file-selection", "--title=Pick a reel to analyze",
                                "--file-filter=Videos | *.mp4 *.mov *.m4v *.webm"], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - fall back to typing the path
        return None
    return None


HELP = """[b]Hookline[/b] — preview a reel's attention before posting.

[b]Try it in 3 steps[/b]
  1. Pick a reel on the left (↑/↓; Enter opens a folder), or press [b]o[/b] to pick your own video
  2. A saved report opens as you move onto a reel. [b]Enter[/b] (or [b]a[/b]) asks for an optional caption
     and analyzes it on the GPU; Esc in the caption box just shows the saved report
  3. Read the scores, viral-potential tag and the attention chart

[b]Keys[/b]
  Enter    caption box -> analyze (re-analyze) on the GPU; Esc in the box opens the saved report
  p        open the reel's saved report (nothing is analysed)
  a        analyze: caption box, then the full TRIBE run on the GPU Space (minutes); the report
           replaces the old one. With the Space paused the run waits and starts once it is live.
           While a run is going, a (or picking a file with o) queues the video; it starts
           when the current run ends. Works on reels, your files and saved reports
  o        pick your own video (system file picker); analysis starts right away
  v        view the selected report in the browser (video + chart + transcript)
  r        jump to a random sample reel
  R        open all reports in the browser (live list, refreshes every 5s)
  g        regenerate the selected report from its raw model output (current formulas)
  V        model-check the top 3 editing suggestions (renders each edit, runs it on the GPU)
  m        run options: video / audio / transcript on or off, transcription language.
           Transcript auto: a speech gate checks how much of the reel is speech; songs
           and humming are scored on visuals + sound, not words
  S        GPU Space: live status, ping / start / restart / stop (stop + restart ask twice)
  x        stop a Space job: the one selected under "live on Space", or this session's run,
           or (press twice) the only job on the Space started elsewhere; on a queued video, unqueue it
  ctrl+r   restart the running Space job (no re-upload, model stays loaded)
  tab      switch between reels and report
  ?        this help        q  quit

[b]Views and followers[/b] (next to each sample reel)
  The page's follower range is context only, never a scoring input: Hookline judges the video.

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
    #reels { width: 46; border: round $panel-lighten-2; scrollbar-size-vertical: 1; }
    #reels:focus-within { border: round $accent; }
    #right { width: 1fr; border: round $panel-lighten-2; padding: 0 1; scrollbar-size-vertical: 1; }
    #right:focus-within { border: round $accent; }
    #tiles { height: 5; }
    .tile { width: 1fr; height: 5; border: round $panel-lighten-2; content-align: center middle; text-align: center; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    .section { margin-top: 1; text-style: bold; }
    """
    BINDINGS = [
        Binding("p", "preview", "Open report"),
        Binding("a", "full", "Analyze"),
        Binding("o", "own_file", "Pick video"),
        Binding("r", "random", "Random reel"),
        Binding("v", "web", "View report"),
        Binding("R", "all_reports", "All reports"),
        Binding("g", "regenerate", "Regenerate", show=False),
        Binding("V", "verify", "Verify edits", show=False),
        Binding("x", "stop", "Stop job"),
        Binding("m", "run_options", "Run options", show=False),
        Binding("ctrl+r", "restart", "Restart job", show=False),
        Binding("S", "space", "Space"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.corpus = load_corpus()
        self.current = None  # tree node data of the selected item
        self.by_path = {}    # video path -> job id analysed this session
        self.report_index = {}  # reel name -> newest saved report job (full TRIBE preferred)
        self.backend_ok = False
        self.token = ""
        self.space_run = None  # in-flight Space job: {d, sid, t0, last_ok, p}
        self.queue = []        # analysis requests waiting for the running one: [{d, mods, text_mode, lang}]
        # Modalities for full TRIBE runs; disabled ones never load their model on the Space.
        self.mods = {"video": True, "audio": True, "text": True}
        # transcript: auto = speech gate decides per reel (songs/humming scored on visuals + sound)
        self.text_mode = "auto"
        self.lang = "auto"  # transcription language: auto (Hindi/English/Hinglish), hi, en
        self.space_rt = {}  # Space runtime: stage, hardware, sleep_s, health, jobs, checked

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
                yield Static("Editing suggestions (ranked by viewers lost)", classes="section", id="recs-h")
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
        self.watch_live = None  # Space job id whose progress is on screen (from "live on Space")
        self.set_interval(5, self.tick_saved)   # batch/other clients save reports while we run
        self.set_interval(3, self.tick_live)    # running/queued jobs on the Space, from any client
        self.set_interval(15, lambda: self.poll_space(ping=False))  # Space stage for the header

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
                label = Text(f"{v['id']:<6} ")
                label.append(f"{fmt_views(v['views']):>5} views", "")
                if v.get("followers"):
                    label.append(f" · {followers_text(v['followers'])}", "dim")
                node.add_leaf(label, data={"kind": "reel", **v})
        if not any(self.corpus.values()):
            tree.root.add_leaf(Text("no samples/ folder found", "dim"), data={"kind": "folder"})
        self.own_node = tree.root.add("your files", data={"kind": "folder"}, expand=True)
        self.live_node = tree.root.add("live on Space", data={"kind": "folder"}, expand=True)
        self._live_sig = None
        self.saved_node = tree.root.add("saved reports", data={"kind": "folder"}, expand=True)
        self._saved_sig = None
        self.refresh_saved()
        self.call_after_refresh(setattr, tree, "cursor_line", 0)

    def refresh_saved(self, force=True):
        paths = [p for p in saved_reports() if p.parent.name != "demo"]
        sig = tuple((p.parent.name, p.stat().st_mtime) for p in paths)
        if not force and sig == self._saved_sig:
            return
        self._saved_sig = sig
        keep = self.cursor_snapshot()
        self.saved_node.remove_children()
        index = {}  # reel name (lowercase file stem) -> (is_full, job); newest first, full TRIBE wins
        for p in paths:
            try:
                r = json.loads(p.read_text())
                s = r["scores"]
                video = r.get("local_video")
                name = r.get("source_id") or (Path(video).stem if video else None)
                # list only reports whose video still exists and that are named after it
                if not video or not Path(video).is_file() or name in (None, "input", p.parent.name):
                    continue
                full = r.get("preview") is False or "full" in str(r.get("neural_source", ""))
                key = name.lower()
                if key not in index or (full and not index[key][0]):
                    index[key] = (full, p.parent.name)
                if len(self.saved_node.children) >= 40:
                    continue
                label = Text(f"{name:<7}")
                tag = s.get("virality_tag") or s.get("virality_band", "")
                label.append(f"{s['virality']:>3.0f} {tag}", BAND_STYLE.get(tag, "dim"))
            except Exception:
                continue
            self.saved_node.add_leaf(label, data={"kind": "saved", "job": p.parent.name})
        self.report_index = {k: job for k, (_, job) in index.items()}
        self.cursor_restore(keep)
        self.saved_node.set_label(f"saved reports ({len(self.saved_node.children)})")

    def tick_saved(self):
        self.refresh_saved(force=False)

    def tick_live(self):
        if self.token and self.space_rt.get("stage") in ("RUNNING", "RUNNING_BUILDING", None):
            self.fetch_live()

    @work(thread=True, exclusive=True, group="live")
    def fetch_live(self):
        try:
            jobs = http_json(f"{SPACE}/jobs", self.token, timeout=8)
        except Exception:
            return
        active = [j for j in jobs if j.get("status") in ("queued", "running")]
        self.space_rt["jobs"] = len(active)
        self.call_from_thread(self.update_live, active)
        sid = self.watch_live
        if sid:
            try:
                p = http_json(f"{SPACE}/jobs/{sid}", self.token, timeout=8)
            except Exception:
                return
            self.call_from_thread(self.show_live_progress, sid, p)

    def update_live(self, active):
        sig = tuple((j["job_id"], j.get("status"), j.get("stage")) for j in active)
        for node in self.live_node.children:  # refresh labels in place (cheap, keeps cursor)
            j = next((x for x in active if x["job_id"] == node.data.get("sid")), None)
            if j:
                node.set_label(self._live_label(j))
        ids = {j["job_id"] for j in active}
        have = {n.data.get("sid"): n for n in self.live_node.children}
        if ids != set(have):  # add / remove only what changed, and keep the cursor where it was
            keep = self.cursor_snapshot()
            for sid, node in have.items():
                if sid not in ids:
                    node.remove()
            for j in active:
                if j["job_id"] not in have:
                    self.live_node.add_leaf(self._live_label(j), data={"kind": "live", "sid": j["job_id"],
                                                                       "source": j.get("source")})
            self.cursor_restore(keep)
        self._live_sig = sig
        self.live_node.set_label(f"live on Space ({len(active)})")
        self.update_queue_label()

    def _live_label(self, j):
        src = Path(j.get("source") or j["job_id"]).stem
        t = Text(src if len(src) <= 24 else src[:23] + "…", "bold")
        t.append("  ·  ", "dim")
        if j.get("status") == "queued":
            t.append("queued", "yellow")
        else:
            t.append(f"{j.get('stage', '?')} {fmt_dur(j.get('age_s'))}", "cyan")
        return t

    def focus_node(self, node):
        """Put the cursor on a node that was just added or expanded: its line only exists after the next
        refresh (moving earlier leaves the cursor on line 0, the top of the panel)."""
        tree = self.query_one("#reels", Tree)

        def go():
            if node.line >= 0:
                tree.move_cursor(node)
                tree.scroll_to_node(node, animate=False)
        self.call_after_refresh(go)

    # ---- keep the tree cursor and scroll steady while live / saved lists change underneath it
    @staticmethod
    def _node_key(node):
        d = (node.data or {}) if node else {}
        return (d.get("kind"), d.get("path") or d.get("job") or d.get("sid") or str(node.label) if node else None)

    def cursor_snapshot(self):
        tree = self.query_one("#reels", Tree)
        return self._node_key(tree.cursor_node), tree.scroll_y

    def cursor_restore(self, snap):
        key, scroll_y = snap
        tree = self.query_one("#reels", Tree)

        def restore():
            found, stack = None, list(tree.root.children)
            while stack and not found:
                n = stack.pop(0)
                if self._node_key(n) == key:
                    found = n
                stack.extend(n.children)
            if found is not None and found.line >= 0 and found is not tree.cursor_node:
                self._restoring = True
                tree.cursor_line = found.line
                tree.call_after_refresh(setattr, self, "_restoring", False)
            tree.scroll_to(y=scroll_y, animate=False)

        self.call_after_refresh(restore)

    def show_live_progress(self, sid, p):
        if self.watch_live != sid:
            return
        if p.get("status") in ("done", "failed", "cancelled"):
            self.say(f"Space job {sid} {p['status']}; batch reports appear under saved reports when saved.",
                     "green" if p["status"] == "done" else "yellow")
            if p["status"] != "done":
                return
        run = {"d": {"id": Path(getattr(self, "watch_source", None) or sid).stem, "path": None}, "sid": sid,
               "t0": time.time() - (p.get("elapsed_s") or 0), "last_ok": time.time(), "p": p}
        self.show_progress(run)

    # ---- status line + header
    def set_header(self):
        b = Text("● up", "green") if self.backend_ok else Text("○ down", "red")
        if not self.token:
            s = Text("○ no token", "yellow")
        else:
            glyph, word, style = STAGE_VIEW.get(self.space_rt.get("stage"), ("·", "checking", "dim"))
            s = Text(f"{glyph} {word}", style)
            s.append("  S", "bold")
        t = Text.assemble(("HOOKLINE", "bold"), ("  video in → attention preview before posting", "dim"))
        mods = Text("   m run: ")
        for key, name in (("video", "video"), ("audio", "audio"), ("text", "transcript")):
            on = self.mods[key]
            box = {"auto": "a", "on": "x", "off": " "}[self.text_mode] if key == "text" else ("x" if on else " ")
            mods.append(f"[{box}] {name}  ", "bold green" if on else "dim")
        if self.text_mode != "off":
            mods.append(f"{self.lang}  ", "bold")
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
        self.call_from_thread(self.poll_space, False)
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
        tile("virality", "VIRAL POTENTIAL", f"{s['virality']:.0f}", s.get("virality_tag", s.get("virality_band", "")))
        self.query_one("#strip", AttentionStrip).show(r)
        m = r["moments"]
        sw, ww = m["strongest_window_s"], m["weakest_window_s"]
        drops = ", ".join(f"{fmt_t(a)}-{fmt_t(b)}" for a, b in m.get("drop_zones_s", [])) or "none"
        moments = Text.assemble(
            ("peak ", "dim"), fmt_t(m["peak_attention_at_s"]),
            ("   strongest ", "dim"), f"{fmt_t(sw[0])}-{fmt_t(sw[1])}",
            ("   weakest ", "dim"), f"{fmt_t(ww[0])}-{fmt_t(ww[1])}",
            ("   drops ", "dim"), drops)
        vb = r["scores"].get("virality_breakdown") or []
        if vb:  # why this viral potential: each factor's pull vs a typical reel (factor at 50)
            moments.append("\nviral potential: ", "dim")
            for i, x in enumerate(sorted(vb, key=lambda x: -abs(x["vs_typical"]))):
                d = x["vs_typical"]
                moments.append(("  " if i else "") + f"{x['label'].lower()} ", "bold" if i == 0 else "")
                moments.append(f"{d:+.1f}", "green" if d >= 0 else "red")
            if r["scores"].get("virality_summary"):
                moments.append("\n" + r["scores"]["virality_summary"], "dim italic")
        cap = r.get("caption")
        if cap:  # only when the creator gave one
            ok = cap.get("relevance", 0) > 0
            moments.append("\ncaption: ", "dim")
            moments.append(f"\"{cap['text'][:90]}{'…' if len(cap['text']) > 90 else ''}\" ", "italic")
            moments.append(cap.get("verdict", ""), "green" if ok else "yellow")
            moments.append(f" · viral +{cap.get('virality_points', 0):.1f}" if ok else " · no effect", "bold" if ok else "dim")
        line = audio_text(r.get("audio"), r.get("speech_gate"))
        if line:
            moments.append("\n")
            moments.append(line)
        mu = (r.get("music") or {}).get("summary")
        mood = (r.get("emotion") or {}).get("music_mood")
        if mu or mood:
            moments.append("\nmusic: ", "dim")
            bits = []
            if mu:
                bits.append(f"{mu['tempo_bpm']:.0f} BPM {mu['mode_label']}")
                if mu.get("buildups"):
                    bits.append("build-ups " + ", ".join(f"{fmt_t(x['start'])}-{fmt_t(x['end'])}" for x in mu["buildups"][:3]))
                if mu.get("beat_cut_sync") is not None:
                    bits.append(f"cuts on beat {mu['beat_cut_sync']:.0%}")
            if mood:
                bits.append("mood " + " → ".join(mood.get("sequence", [])[:5]))
            moments.append(" · ".join(bits), "cyan")
        em = r.get("emotion")
        if em:
            sm = em["summary"]
            dom = ", ".join(f"{x['emotion']} {x['share']:.0%}" for x in sm.get("dominant_emotions", [])) or "no clear emotion"
            if sm.get("no_clear_share") is not None:
                dom += f" · no clear {sm['no_clear_share']:.0%}"
            if sm.get("confidence") not in (None, "none"):
                dom += f" · confidence {sm['confidence']}"
            pk = ", ".join(f"{fmt_t(p['at_s'])} {p['emotion']}" for p in sm.get("peak_moments", [])) or "none"
            moments.append("\nemotion: ", "dim")
            moments.append(dom, "bold magenta")
            moments.append(f" · arc {sm.get('arc')} · high-arousal {sm.get('high_arousal_share', 0):.0%}"
                           f" · peaks {pk}", "magenta")
            st = em.get("strong") or {}
            if st:
                parts = ", ".join(f"{e} {d}s" for e, d in (st.get("durations_s") or {}).items() if d) or "none"
                moments.append("\nstrong emotions: ", "dim")
                moments.append(f"{parts} · {st.get('share', 0):.0%} of the reel · emotional strength "
                               f"{st.get('score', 0):.0f}/100", "bold")
                moments.append(" · insight only (not a score input)", "dim")
                if st.get("audible_laughter_s") is not None:
                    moments.append(f" · audible laughter {st['audible_laughter_s']}s", "dim")
        self.query_one("#moments", Static).update(moments)
        tbl = Table(expand=True, box=None, show_header=True, header_style="dim", pad_edge=False)
        tbl.add_column("AT", style="bold green", no_wrap=True, width=10)
        tbl.add_column("EDIT", ratio=3)
        tbl.add_column("WHY (vs your best part)", ratio=3, style="dim")
        tbl.add_column("LOST", justify="right", width=6)
        tbl.add_column("MODEL CHECK", ratio=1)
        for x in (r.get("suggestions") or [] if SHOW_SUGGESTIONS else []):
            v = x.get("verification") or {}
            check = (Text(f"{v['verdict']} {v.get('engagement_change_pct', 0):+.1f}%",
                          "green" if v["verdict"] == "improved" else "red" if v["verdict"] == "worse" else "dim")
                     if v.get("verdict") in ("improved", "worse", "no clear change") else Text("not checked", "dim"))
            tbl.add_row(x["at"], x["suggestion"], x["why"], f"{x['impact']['viewers_lost_pct']:.0f}%", check)
        if r.get("verify_status"):
            tbl.caption = f"verification: {r['verify_status']}"
        tr = r.get("transcript") if isinstance(r.get("transcript"), dict) else {}
        if tr.get("segments") or tr.get("status"):
            lines = Text(f"\nTranscript {'used for judging' if tr.get('used_by_model') else '(not used)'}",
                         "bold")
            lines.append(f"  {tr.get('status', '')} · {tr.get('detected') or ''}\n", "dim")
            for seg in (tr.get("segments") or [])[:40]:
                lines.append(f"{fmt_t(seg['start']):>6} ", "green")
                lines.append(seg["text"] + "\n")
            self.query_one("#recs", Static).update(Group(tbl, lines) if SHOW_SUGGESTIONS else lines)
        else:
            self.query_one("#recs", Static).update(tbl if SHOW_SUGGESTIONS else Text(""))
        self.query_one("#method", Static).update(Text(
            f"\n{r.get('method', '')} · neural: {r.get('neural_source', '?')}", "dim italic"))
        for w in ("#tiles", "#strip-h", "#strip", "#moments", "#recs-h", "#recs", "#method"):
            self.query_one(w).display = True
        self.query_one("#recs-h").display = SHOW_SUGGESTIONS

    def show_reel_card(self, d):
        self.current_job = None
        for w in ("#tiles", "#strip-h", "#strip", "#moments", "#recs-h", "#recs", "#method"):
            self.query_one(w).display = False
        views = (f"{fmt_views(d.get('views'))} views · {followers_text(d.get('followers'))} "
                 f"({page_size(d.get('followers'))}) · " if d.get("views") else "")
        self.query_one("#title", Static).update(Text.assemble(
            (d["id"], "bold"), (f"   {d.get('stratum', '')}\n\n", "dim"),
            (f"{views}{short_path(d['path'])}\n", "dim"),
            ("Views and page size are context only; scoring judges the video, never the account.\n\n", "dim italic"),
            ("No report yet for this reel.\n", "bold yellow"),
            ("Enter / a", "bold"), "  caption (optional), then analyze: full TRIBE on the GPU Space (minutes)\n",
            ("m", "bold"), "          run options: video / audio / transcript, language\n",
            ("S", "bold"), "          GPU Space status (start it here if it is paused)"))

    @on(Tree.NodeHighlighted)
    def highlighted(self, ev: Tree.NodeHighlighted):
        if getattr(self, "_restoring", False):  # cursor put back after a list refresh: nothing changed for the user
            return
        d = ev.node.data or {}
        self.watch_live = d.get("sid") if d.get("kind") == "live" else None
        self.watch_source = d.get("source")
        if d.get("kind") == "live":
            self.current_job = None
            self.say(f"Live Space job {d['sid']} ({d.get('source')}): progress updates every 3s")
            self.fetch_live()
            return
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
            job = self.report_for(d)
            if job:
                self.show_report(json.loads((OUT / job / "report.json").read_text()), job, d["id"])
            else:
                self.show_reel_card(d)
            pos = self.queue_pos(d)
            if pos:
                self.say(f"{d['id']} is queued (#{pos}) behind {self.space_run['d']['id'] if self.space_run else 'the current run'}."
                         " x removes it from the queue.", "cyan")

    def report_for(self, d):
        """Saved report job for a reel or file, or None: this session's run, else the index by file name."""
        for job in (self.by_path.get(d["path"]), self.report_index.get(Path(d["path"]).stem.lower()),
                    self.report_index.get(str(d.get("id", "")).lower())):
            if job and (OUT / job / "report.json").exists():
                return job
        return None

    @on(Tree.NodeSelected)
    def selected(self, ev: Tree.NodeSelected):
        """Enter on a reel or file: caption box -> (re-)analyze; Esc in the box opens the saved report."""
        if (ev.node.data or {}).get("kind") in ("reel", "file"):
            self.current = ev.node.data
            self.action_full(from_select=True)

    # ---- actions
    def target(self):
        d = self.current
        if d and d.get("kind") == "saved":  # a saved report: act on the video it was made from
            d = self.video_of_report(d["job"]) or d
        if not d or d.get("kind") not in ("reel", "file"):
            self.say("Select a reel, your own file or a saved report first (o picks a file).", "yellow")
            return None
        return d

    def video_of_report(self, job):
        """The reel/file a saved report belongs to, as a tree item (sample reel if it is one)."""
        try:
            r = json.loads((OUT / job / "report.json").read_text())
        except (OSError, ValueError):
            return None
        video = r.get("local_video")
        if not video or not Path(video).is_file():
            self.say(f"The video for report {job} is no longer on disk.", "yellow")
            return None
        name = r.get("source_id") or Path(video).stem
        for items in self.corpus.values():
            for v in items:
                if v["id"].lower() == name.lower():
                    return {"kind": "reel", **v}
        return {"kind": "file", "id": name, "path": video, "stratum": "your file"}

    def action_preview(self):
        """Open the reel's saved report. Nothing is analysed here: new reports come from `a`."""
        d = self.target()
        if not d:
            return
        job = self.report_for(d)
        if job:
            self.show_report(json.loads((OUT / job / "report.json").read_text()), job, d["id"])
            self.say(f"Report for {d['id']} ({job}). v views it in the browser, a re-runs the analysis.")
        else:
            self.show_reel_card(d)
            self.say(f"No report for {d['id']} yet. Press a to analyze it and build the report.", "yellow")

    def action_full(self, from_select=False):
        """a / Enter / file pick: ask for the optional caption, then analyze (queued when busy or paused).
        from_select: Esc in the caption box opens the saved report instead of just cancelling."""
        d = self.target()
        if not d:
            return
        if not self.token:
            self.say("No HF token: run `hf auth login` or export HF_TOKEN, then restart.", "yellow")
            if from_select:
                self.action_preview()
            return
        self.push_screen(CaptionPrompt(d["id"], self.last_caption(d), esc_hint="opens the saved report"
                                       if from_select else "cancels"),
                         lambda cap: self.analyze_with(d, cap, from_select))

    def space_down(self):
        stage = self.space_rt.get("stage")
        return stage in ("PAUSED", "STOPPED") or str(stage).endswith("_ERROR")

    def last_caption(self, d):
        """Prefill: the caption this video was analysed with last time, if any."""
        job = self.report_for(d)
        try:
            return ((json.loads((OUT / job / "report.json").read_text()).get("caption") or {}).get("text") or "") if job else ""
        except (OSError, ValueError):
            return ""

    def analyze_with(self, d, caption, from_select=False):
        if caption is None:
            if from_select:
                self.action_preview()  # Esc: just look at the saved report (or the "no report yet" card)
            else:
                self.say(f"Analysis of {d['id']} cancelled.", "dim")
            return
        opts = {"mods": dict(self.mods), "text_mode": self.text_mode, "lang": self.lang, "caption": caption}
        if self.space_run or self.queue or self.space_down():
            self.enqueue(d, opts)
            return
        if self.report_for(d):
            self.say(f"Re-analyzing {d['id']}; the new report replaces the current one when it finishes.")
        self.begin_run(d, opts)

    def enqueue(self, d, opts):
        """Queue an analysis behind the running one, with its run options and caption."""
        running = self.space_run["d"]["path"] if self.space_run else None
        if d["path"] == running or any(q["d"]["path"] == d["path"] for q in self.queue):
            pos = next((i + 1 for i, q in enumerate(self.queue) if q["d"]["path"] == d["path"]), 0)
            self.say(f"{d['id']} is already " + (f"queued (#{pos})." if pos else "being analyzed."), "yellow")
            return
        self.queue.append({"d": d, **opts})
        if self.space_run:
            when = f"starts after {self.space_run['d']['id']}"
        elif self.space_down():
            when = "the GPU Space is not running: press S to start it, the run starts by itself once it is live"
        else:
            when = "starts after the current run"
        self.say(f"Queued {d['id']} (#{len(self.queue)}): {when}. x on a queued item removes it.", "cyan")
        self.update_queue_label()

    def update_queue_label(self):
        n = len(self.queue)
        base = str(self.live_node.label).split(" · ")[0]
        self.live_node.set_label(base + (f" · {n} queued here" if n else ""))

    def queue_pos(self, d):
        return next((i + 1 for i, q in enumerate(self.queue) if q["d"]["path"] == (d or {}).get("path")), 0)

    def start_next(self):
        """Called when a run ends (done, failed, stopped): start the next queued request."""
        if self.space_run or not self.queue:
            return
        stage = self.space_rt.get("stage")
        if self.space_down():
            self.say(f"{len(self.queue)} queued, but the GPU Space is {str(stage).lower()}: press S to start it; "
                     "the queue resumes by itself once it is live.", "yellow")
            return
        q = self.queue.pop(0)
        self.update_queue_label()
        self.begin_run(q["d"], q)

    def begin_run(self, d, opts):
        """Mark the Space busy right away (the upload itself can take a while), then start the worker."""
        self.space_run = {"d": d, "sid": None, "t0": time.time(), "last_ok": time.time(), "mods": dict(opts["mods"]),
                          "p": {"status": "uploading", "stage_label": "Uploading to the GPU Space", "step": 0, "steps": 4}}
        self.run_space(d, opts)

    @work(thread=True, group="space")
    def run_space(self, d, opts):
        try:
            self._space_job(d, opts)
        finally:
            self.space_run = None
            self.call_from_thread(self.start_next)

    def _space_job(self, d, opts):
        """Upload to the Space, then poll its progress API every POLL_S until done/failed."""
        tok = self.token
        mb = Path(d["path"]).stat().st_size / 1e6
        self.call_from_thread(self.say, f"Uploading {d['id']} ({mb:.1f}MB) to GPU Space"
                                        f"{' with caption' if opts.get('caption') else ''} …")
        try:
            m = opts["mods"]
            q = f"video={int(m['video'])}&audio={int(m['audio'])}&text={opts['text_mode']}&lang={opts['lang']}"
            res = post_video(f"{SPACE}/analyze?{q}", d["path"], token=tok,
                             fields={"caption": opts["caption"]} if opts.get("caption") else None)
        except Exception as e:
            self.call_from_thread(self.say, f"Space upload failed: {e}", "red")
            return
        sid = res["job_id"]
        run = {"d": d, "sid": sid, "t0": time.time(), "last_ok": time.time(), "mods": dict(opts["mods"]), "p": {
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
                try:  # preferred: raw output -> report built locally by the backend
                    q = urllib.parse.urlencode({"video": str(Path(d["path"]).resolve()), "source_id": d["id"]})
                    r = http_json(f"{BACKEND}/import/{sid}?{q}", timeout=300, data=b"")["report"]
                except Exception as e:  # backend down: keep the Space-built report
                    self.call_from_thread(self.say, f"Local import failed ({e}); using the Space report", "yellow")
                    jd.mkdir(parents=True, exist_ok=True)
                    r["preview"] = False
                    r["local_video"] = str(Path(d["path"]).resolve())
                    (jd / "report.json").write_text(json.dumps(r, indent=1))
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
                                                "a starts a new run.", "yellow")
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
        msg = (f"Space {run['sid'] or '(uploading)'} · step {p.get('step', 0)}/{p.get('steps', 4)} {where} · "
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
        t = Text.assemble((d["id"], "bold"), (f"   full TRIBE on GPU Space · job {run['sid'] or 'uploading'} · {used}\n\n", "dim"))
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
        gate = p.get("speech_gate")
        if gate:
            off = gate.get("decision") == "transcript off"
            t.append(f"\n   speech gate: {gate.get('decision')} · {gate.get('reason')}\n",
                     "yellow" if off else "green")
            am = p.get("audio_match") or {}
            line = audio_text({"type": gate.get("audio_type"), "song": am.get("song"), "reuse": am.get("reuse")})
            if line:
                t.append("   ")
                t.append(line)
                t.append("\n")
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
        if key == "text":  # auto -> on -> off
            order = ["auto", "on", "off"]
            mode = order[(order.index(self.text_mode) + 1) % 3]
            if mode == "off" and not (self.mods["video"] or self.mods["audio"]):
                self.say("Keep at least one of video, audio, transcript on.", "yellow")
                return
            self.text_mode = mode
            self.mods["text"] = mode != "off"
            self.set_header()
            self.say({"auto": "Transcript: auto (speech gate decides; songs/humming scored on visuals + sound)",
                      "on": "Transcript: always on", "off": "Transcript: off"}[mode])
            return
        new = dict(self.mods, **{key: not self.mods[key]})
        if not any(new.values()):
            self.say("Keep at least one of video, audio, transcript on.", "yellow")
            return
        self.mods = new
        self.set_header()
        on = " + ".join(k for k, v in self.mods.items() if v)
        self.say(f"Full TRIBE runs will use: {on}. Models for the others are never loaded.")

    def action_cycle_lang(self):
        order = ["auto", "hi", "en"]
        self.lang = order[(order.index(self.lang) + 1) % len(order)]
        self.set_header()
        what = {"auto": "auto-detect (Hindi, English, Hinglish)", "hi": "Hindi", "en": "English"}[self.lang]
        self.say(f"Transcription language: {what}" + ("" if self.mods["text"] else " (turn transcript on in run options, m)"))

    def action_stop(self):
        """x: remove the selected video from the queue, or stop a Space job: the one selected under
        "live on Space", else this session's run, else the only job running on the Space (any client:
        batch scripts, a previous TUI session) after a second press."""
        node = self.query_one("#reels", Tree).cursor_node
        sel = (node.data or {}) if node else {}
        if sel.get("kind") == "live":  # explicit choice in the tree
            self.space_call("cancel", sel["sid"])
            return
        d = self.current
        if d and d.get("kind") == "saved":
            d = self.video_of_report(d["job"])
        pos = self.queue_pos(d)
        if pos:
            self.queue.pop(pos - 1)
            self.update_queue_label()
            self.say(f"Removed {d['id']} from the queue ({len(self.queue)} left).", "yellow")
            return
        run = self.space_run
        if run and not run["sid"]:
            self.say(f"{run['d']['id']} is still uploading; stop it once it shows progress.", "yellow")
            return
        if run:
            self.space_call("cancel", run["sid"])
            return
        live = [n.data for n in self.live_node.children if (n.data or {}).get("sid")]
        if not live:
            self.say("No Space job running.", "yellow")
            return
        if len(live) > 1:
            self.say(f"{len(live)} jobs on the Space: select one under \"live on Space\" and press x.", "yellow")
            return
        sid, src = live[0]["sid"], Path(live[0].get("source") or live[0]["sid"]).stem
        armed = getattr(self, "_stop_armed", None)
        if armed and armed[0] == sid and time.time() - armed[1] < 5:
            self._stop_armed = None
            self.space_call("cancel", sid)
        else:
            self._stop_armed = (sid, time.time())
            self.say(f"{src} ({sid}) was started outside this session. Press x again within 5s to stop it.",
                     "bold yellow")

    def action_restart(self):
        run = self.space_run
        if not run:
            self.say("No Space job running; press a to start one.", "yellow")
            return
        if not run["sid"]:
            self.say(f"{run['d']['id']} is still uploading.", "yellow")
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
            self.call_from_thread(self.say, f"Stopping {sid}: the Space stops at its next checkpoint (usually within a minute) …", "yellow")
        elif run:
            run.update(sid=r["job_id"], t0=time.time(), last_ok=time.time(), p={
                "status": "queued", "stage_label": f"Restarted (was {sid}); waiting for the GPU",
                "step": 0, "steps": 4, "eta_s": r.get("estimate_s"), "video_s": r.get("video_s")})
            self.call_from_thread(self.refresh_progress)
            self.call_from_thread(self.say, f"Restarted as {r['job_id']} (same upload, model stays loaded)", "green")

    def action_own_file(self):
        """System file picker (typed path as fallback); the picked video goes straight to analysis."""
        self.say("Opening the file picker …")
        self.pick_file()

    @work(thread=True, exclusive=True, group="picker")
    def pick_file(self):
        path = pick_video_file()
        if path is None:  # no system picker: type or drag the path instead
            self.call_from_thread(self.push_screen, PathPrompt(), self.add_own_file)
        elif not path:
            self.call_from_thread(self.say, "No file picked.", "dim")
        else:
            self.call_from_thread(self.add_own_file, path)

    def add_own_file(self, raw):
        if not raw:
            return
        p = Path(os.path.expanduser(raw.strip().strip("'\""))).resolve()
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
            self.say(f"Not a video file: {raw}", "red")
            return
        d = {"kind": "file", "id": p.stem, "path": str(p), "stratum": "your file"}
        node = next((n for n in self.own_node.children if (n.data or {}).get("path") == str(p)), None) \
            or self.own_node.add_leaf(p.name, data=d)
        self.own_node.expand()
        self.focus_node(node)
        self.current = d
        self.action_full()  # analyze right away (uploads to the GPU Space)

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
        self.focus_node(node)
        tree.focus()

    def action_web(self):
        job = getattr(self, "current_job", None)
        if not job:
            self.say("No report selected.", "yellow")
            return
        webbrowser.open(f"{BACKEND}/report/{job}/view")
        self.say(f"Opened {BACKEND}/report/{job}/view")

    def action_regenerate(self):
        job = getattr(self, "current_job", None)
        if not job or not (OUT / job / "raw" / "meta.json").exists():
            self.say("No raw model output stored for this report (only reports built locally can be regenerated).",
                     "yellow")
            return
        self.regenerate(job)

    @work(thread=True, group="regen")
    def regenerate(self, job):
        self.call_from_thread(self.say, f"Regenerating {job} from raw model output …")
        try:
            r = http_json(f"{BACKEND}/report/{job}/regenerate", timeout=300, data=b"")["report"]
        except Exception as e:
            self.call_from_thread(self.say, f"Regenerate failed: {e}", "red")
            return
        self.call_from_thread(self.show_report, r, job, r.get("source_id") or job)
        self.call_from_thread(self.refresh_saved)
        self.call_from_thread(self.say, f"Regenerated {job} with the current formulas", "green")

    def action_verify(self):
        job = getattr(self, "current_job", None)
        if not job or not (OUT / job / "raw" / "meta.json").exists():
            self.say("Verification needs a full TRIBE report built from raw output.", "yellow")
            return
        try:
            http_json(f"{BACKEND}/report/{job}/verify", timeout=20, data=b"")
            self.say(f"Verifying the top edits of {job} on the GPU (~2-3 min each). "
                     "The report updates as each finishes (g to reload).", "green")
        except Exception as e:
            self.say(f"Verify failed to start: {e}", "red")

    def action_all_reports(self):
        webbrowser.open(f"{BACKEND}/reports")
        self.say(f"Opened {BACKEND}/reports (all reports, refreshes every 5s)")

    def action_help(self):
        self.push_screen(HelpScreen())

    def action_run_options(self):
        self.push_screen(RunOptions())

    # ---- GPU Space status + control
    def action_space(self):
        self.push_screen(SpaceScreen())

    @work(thread=True, exclusive=True, group="space-rt")
    def poll_space(self, ping=False):
        if not self.token:
            return
        try:
            rt = space_runtime(self.token)
            self.space_rt.update(rt, checked=time.time(), fetch_error="")
        except Exception as e:
            self.space_rt.update(fetch_error=str(e)[:120], checked=time.time())
        if self.space_rt.get("stage") not in ("RUNNING", "RUNNING_BUILDING"):
            self.space_rt.update(jobs=None, health=None)
        elif ping:  # only ping a live Space here; pinging a sleeping one wakes it (and bills)
            ok, info = space_ping(self.token)
            self.space_rt["health"] = f"{'ok' if ok else 'failing'} · {info}"
        self.call_from_thread(self.set_header)
        if self.queue and not self.space_run and self.space_rt.get("stage") == "RUNNING":
            self.call_from_thread(self.start_next)  # Space came up: resume the waiting queue

    @work(thread=True, group="space-ctl")
    def space_action(self, verb, screen):
        tok = self.token
        stage = self.space_rt.get("stage")
        try:
            if verb == "ping":
                ok, info = space_ping(tok)
                self.space_rt["health"] = f"{'ok' if ok else 'failing'} · {info}"
                msg, style = (f"Ping {info}: Space answered." if ok else
                              f"Ping {info}: not answering (paused/starting?)."), "green" if ok else "yellow"
            elif verb == "start" and stage == "SLEEPING":
                space_ping(tok)  # a request wakes a sleeping Space
                msg, style = "Wake request sent; Space is starting (~2-4 min).", "green"
            elif verb in ("start", "restart"):
                space_control("restart", tok)  # also resumes a paused Space
                msg, style = ("Start requested; Space is booting (~3-5 min, billing from now)." if verb == "start"
                              else "Restart requested; models reload in ~3-5 min."), "green"
            elif verb == "stop":
                space_control("pause", tok)
                msg, style = "Space paused: billing stopped. Start it again from here.", "green"
            else:
                return
        except urllib.error.HTTPError as e:
            msg, style = f"{verb} failed: HTTP {e.code} {e.reason}", "red"
        except Exception as e:
            msg, style = f"{verb} failed: {e}", "red"
        self.call_from_thread(screen.note, msg, style)
        self.call_from_thread(self.say, f"Space: {msg}", style)
        self.call_from_thread(self.poll_space, False)


if __name__ == "__main__":
    Hookline().run()
