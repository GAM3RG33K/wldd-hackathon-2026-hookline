#!/usr/bin/env python
"""Merge the YouTube Shorts songs chart into the trending-audio catalog (hf_space/trending_audio.json).

Source: https://charts.youtube.com/charts/TopShortsSongs/in/weekly ("songs used most by creators
on Shorts", top 50 per country, weekly). The chart is public, but its data endpoint only answers a
real browser session (scripts get HTTP 429), so the weekly refresh is one console paste:

  1. Open https://charts.youtube.com/charts/TopShortsSongs/in/weekly in Chrome.
  2. DevTools console: paste CONSOLE_SNIPPET (printed by `python tribe/update_trending.py --snippet`).
     It copies the India + global charts to the clipboard as JSON.
  3. pbpaste | python tribe/update_trending.py -        # then redeploy hf_space (catalog is baked in)

Fingerprints need audio: each song is looked up on the iTunes Search API for its public 30s preview
(same as the hand-curated entries). No YouTube audio is downloaded. Songs without an iTunes preview
(many slowed / sped-up edits) are listed as chart-only: title-level context, not audio-matchable.
Report context only; nothing here feeds the scores.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "hf_space" / "trending_audio.json"
CHART_URL = "https://charts.youtube.com/charts/TopShortsSongs/{cc}/weekly"
ITUNES = "https://itunes.apple.com/search?{q}"
PAUSE_S = 3.2  # iTunes Search allows ~20 calls/minute

CONSOLE_SNIPPET = r"""
(async () => {
  const get = async (cc) => {
    const body = {context: {client: {clientName: "WEB_MUSIC_ANALYTICS", clientVersion: "2.0", hl: "en", gl: "IN"}},
      browseId: "FEmusic_analytics_charts_home",
      query: `perspective=CHART_DETAILS&chart_params_country_code=${cc}&chart_params_chart_type=SHORTS_TRACKS_BY_USAGE&chart_params_period_type=WEEKLY`};
    const j = await (await fetch('/youtubei/v1/browse?alt=json', {method: 'POST', credentials: 'omit',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})).json();
    const tt = j.contents.sectionListRenderer.contents[0].musicAnalyticsSectionRenderer.content.trackTypes[0];
    return {chart: "youtube_shorts_weekly", country: cc, week_end: tt.endDate,
      tracks: tt.trackViews.map(t => ({rank: t.chartEntryMetadata.currentPosition, title: t.name,
        artist: t.artists.map(a => a.name).join(', '), youtube_id: t.encryptedVideoId || '',
        weeks_on_chart: t.chartEntryMetadata.periodsOnChart}))});
  };
  copy(JSON.stringify([await get('in'), await get('global')]));
  console.log('YouTube Shorts charts copied to clipboard');
})();
"""

# Edits that change tempo/pitch: an original-version preview will not fingerprint-match them.
ALTERED = re.compile(r"\b(slowed|sped ?up|super slowed|reverb|nightcore|looped|8d)\b", re.I)


def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\((feat|ft)\.?[^)]*\)|\[[^]]*\]", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _artists(s):
    return {_norm(a) for a in re.split(r",|&| x | and ", s) if _norm(a)}


def itunes_preview(title, artist):
    """Best iTunes match with a preview: title must match and at least one artist must overlap."""
    want_t, want_a = _norm(title), _artists(artist)
    for country in ("IN", "US"):
        q = urllib.parse.urlencode({"term": f"{title} {artist.split(',')[0]}", "media": "music",
                                    "entity": "song", "country": country, "limit": 10})
        try:
            with urllib.request.urlopen(ITUNES.format(q=q), timeout=20) as r:
                results = json.load(r).get("results", [])
        except Exception as e:  # noqa: BLE001 - one failed lookup must not stop the import
            print(f"  ! iTunes {country} failed for {title}: {e}")
            results = []
        time.sleep(PAUSE_S)
        for x in results:
            got_t = _norm(x.get("trackName", ""))
            if not x.get("previewUrl") or not want_t or not (got_t == want_t or got_t.startswith(want_t + " ")):
                continue
            if want_a & _artists(x.get("artistName", "")) or any(a in _norm(x.get("artistName", "")) for a in want_a):
                return {"itunes_id": x["trackId"], "preview_url": x["previewUrl"],
                        "itunes_title": x["trackName"], "itunes_artist": x["artistName"]}
    return None


def merge(charts, catalog):
    """Rebuild the YouTube part of the catalog from `charts`; keep every other source untouched."""
    keep = [t for t in catalog.get("tracks", []) if not str(t.get("list", "")).startswith("YouTube Shorts")]
    known = {t.get("youtube_id"): t for t in catalog.get("tracks", []) + catalog.get("chart_only", [])
             if t.get("youtube_id")}
    rows = {}
    for ch in charts:
        cc = ch["country"]
        for t in ch["tracks"]:
            key = t.get("youtube_id") or _norm(t["title"] + " " + t["artist"])
            r = rows.setdefault(key, {"title": t["title"], "artist": t["artist"], "youtube_id": t.get("youtube_id", ""),
                                      "ranks": {}, "weeks_on_chart": t.get("weeks_on_chart")})
            r["ranks"][cc] = t["rank"]
    week = max(ch["week_end"] for ch in charts)
    matched, chart_only = [], []
    for i, r in enumerate(sorted(rows.values(), key=lambda r: min(r["ranks"].values())), 1):
        ranks = ", ".join(f"{'India' if cc == 'in' else cc.title()} #{n}" for cc, n in sorted(r["ranks"].items()))
        entry = {"title": r["title"], "artist": r["artist"], "youtube_id": r["youtube_id"],
                 "list": f"YouTube Shorts songs, {ranks} (week to {week})",
                 "region": "IN" if "in" in r["ranks"] else "global", "weeks_on_chart": r["weeks_on_chart"]}
        prev = known.get(r["youtube_id"])
        if prev and prev.get("preview_url"):  # looked up before: reuse, no iTunes call
            hit = {k: prev[k] for k in ("itunes_id", "preview_url", "itunes_title", "itunes_artist") if k in prev}
        elif prev and prev.get("no_preview_checked") == max(ch["week_end"] for ch in charts):
            hit = None
        else:
            hit = itunes_preview(r["title"], r["artist"])
        status = "preview" if hit else ("altered edit, no preview" if ALTERED.search(r["title"]) else "no preview")
        print(f"[{i:>3}/{len(rows)}] {status:<26} {r['title']} - {r['artist']}")
        if hit:
            matched.append({**entry, **hit})
        else:
            chart_only.append({**entry, "no_preview_checked": week})
    catalog["tracks"] = keep + matched
    catalog["chart_only"] = chart_only
    catalog["updated"] = date.today().isoformat()
    srcs = [s for s in catalog.get("sources", []) if "charts.youtube.com" not in s]
    catalog["sources"] = srcs + [CHART_URL.format(cc=ch["country"]) for ch in charts]
    catalog["youtube_shorts_week"] = week
    return catalog, len(matched), len(chart_only)


def main():
    if "--snippet" in sys.argv:
        print(CONSOLE_SNIPPET)
        return
    src = sys.argv[1] if len(sys.argv) > 1 else "-"
    charts = json.load(sys.stdin if src == "-" else open(src))
    catalog = json.loads(CATALOG.read_text())
    catalog, n_ok, n_none = merge(charts, catalog)
    CATALOG.write_text(json.dumps(catalog, indent=1, ensure_ascii=False))
    print(f"\n{n_ok} chart songs with previews (fingerprinted on the Space), {n_none} chart-only; "
          f"{len(catalog['tracks'])} tracks total -> {CATALOG}")


if __name__ == "__main__":
    main()
