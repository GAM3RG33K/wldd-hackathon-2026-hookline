"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Report = {
  job_id: string;
  video: string;
  caption: string;
  duration_s: number;
  demo: boolean;
  method: string;
  scores: { hook_pct: number; hook_band: string; hold_rate_pct: number; finish_pct: number; virality: number; virality_band: string; virality_tag: string };
  virality_rows: Array<{ key: string; label: string; value: number; points: number; vs_typical: number; explain: string; weight: number }>;
  series: { retention: number[]; attention: number[] };
  moments: { peak_attention_at_s: number; hooks_s: Array<[number, number]>; drop_zones_s: Array<[number, number]> };
  suggestions: Array<{ at: string; title: string; why: string; est_lost_pct: number }>;
  stages: string[];
};

const SAMPLES = [
  { name: "fl005", label: "Reality tale", dur: "0:08", durS: 8 },
  { name: "marriage-is-scary", label: "Marriage is scary", dur: "0:28", durS: 28 },
  { name: "konsa-colour", label: "Konsa colour bola?", dur: "0:19", durS: 19 },
];

const DEFAULT_CAPTION =
  "Wait for the ending — you won't see it coming. #storytime #reels #viral @realitytales.in";

const STAGES = [
  "Uploading video and caption",
  "Processing the video",
  "Analysing video content",
  "Analysing audio",
  "Matching script and caption to the video",
  "Building the retention graph",
  "Extracting emotional peaks",
  "Scoring against viral formulas",
];

const API = "http://localhost:8000";

function fmt(t: number) {
  t = Math.max(0, Math.floor(t));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
}
function tc(t: number) {
  // 00:00:06.000 — the PDF's millisecond timecode
  const s = Math.max(0, t);
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const sec = String(Math.floor(s % 60)).padStart(2, "0");
  const ms = String(Math.floor((s % 1) * 1000)).padStart(3, "0");
  return `${h}:${m}:${sec}.${ms}`;
}

/* ══════════════ chart: synced playhead, drag-scrub, layers ══════════════ */

const DEMO_EMOS = [
  { name: "Admiration", color: "#c9a7f5" },
  { name: "Sorrow", color: "#4b7bf5" },
  { name: "Romance", color: "#f27fb2" },
  { name: "No clear emotion", color: "#c9c9c9" },
  { name: "Excitement", color: "#f59e3b" },
];

function ChartCard({ report, playT, onScrub, layers, setLayers }: {
  report: Report; playT: number; onScrub: (t: number) => void;
  layers: Record<string, boolean>; setLayers: (l: Record<string, boolean>) => void;
}) {
  const R = report.series.retention;
  const A = report.series.attention;
  const T = R.length;
  const W = 720, H = 280, PADL = 34, PADB = 26, PADT = 14;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragging = useRef(false);
  const x = (i: number) => PADL + (i / Math.max(T - 1, 1)) * (W - PADL - 10);
  const y = (v: number) => PADT + (1 - Math.max(0, Math.min(100, v)) / 100) * (H - PADT - PADB);
  // retention is 0-1, attention is 0-100 — scale each correctly
  const pathR = (arr: number[]) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v * 100).toFixed(1)}`).join(" ");
  const pathA = (arr: number[]) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const tToX = (t: number) => PADL + (t / Math.max(T - 1, 1)) * (W - PADL - 10);

  const scrubFromEvent = (e: React.PointerEvent) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const fx = (e.clientX - rect.left) / rect.width * W;
    const t = Math.max(0, Math.min(T - 1, (fx - PADL) / (W - PADL - 10) * (T - 1)));
    onScrub(t);
  };

  const hooks = report.moments.hooks_s;
  const drops = report.moments.drop_zones_s;
  const peak = report.moments.peak_attention_at_s;
  // demo scene cuts (every few seconds) — demo build only
  const cuts = useMemo(() => {
    if (!report.demo) return [];
    const out: number[] = [];
    for (let s = 4; s < T - 1; s += 4 + (s % 3)) out.push(s);
    return out;
  }, [report.demo, T]);

  const LAYER_DEFS = [
    { key: "retention", label: "Retention" },
    { key: "zones", label: "Zones" },
    { key: "cuts", label: "Cuts" },
    { key: "emotion", label: "Emotion" },
  ];

  return (
    <div className="card chart-card a-chart">
      <svg
        ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Retention chart"
        style={{ cursor: "crosshair", touchAction: "none" }}
        onPointerDown={(e) => { dragging.current = true; (e.target as Element).setPointerCapture?.(e.pointerId); scrubFromEvent(e); }}
        onPointerMove={(e) => { if (dragging.current) scrubFromEvent(e); }}
        onPointerUp={() => { dragging.current = false; }}
        onPointerLeave={() => { dragging.current = false; }}
      >
        {/* zones */}
        {layers.zones && drops.map(([a, b], i) => (
          <g key={`d${i}`}>
            <rect x={x(a)} y={PADT} width={Math.max(x(b) - x(a), 2)} height={H - PADT - PADB} fill="var(--red-soft)" opacity={0.55} />
            {i === 0 && <text x={x(a) + 4} y={PADT + 10} fontSize={9} fontFamily="var(--font-mono)" fill="#8a4a3a">Drop zone</text>}
          </g>
        ))}
        {layers.zones && hooks.map(([a, b], i) => (
          <g key={`h${i}`}>
            <rect x={x(a)} y={PADT} width={Math.max(x(b) - x(a), 2)} height={H - PADT - PADB} fill="var(--green-soft)" opacity={0.5} />
            {i === 0 && <text x={x(a) + 4} y={PADT + 10} fontSize={9} fontFamily="var(--font-mono)" fill="#4a6b1a">Hook window</text>}
          </g>
        ))}
        {layers.zones && (
          <g>
            <rect x={x(Math.max(0, peak - 1))} y={PADT} width={Math.max(x(Math.min(T - 1, peak + 1)) - x(Math.max(0, peak - 1)), 4)} height={H - PADT - PADB} fill="var(--yellow)" opacity={0.4} />
            <text x={x(peak) - 4} y={H - PADB - 6} fontSize={9} fontFamily="var(--font-mono)" fill="#7a6410" textAnchor="end">Best window</text>
          </g>
        )}

        {/* gridlines */}
        {[0, 50, 100].map((v) => (
          <g key={v}>
            <line x1={PADL} x2={W - 10} y1={y(v)} y2={y(v)} stroke="#d8d2c4" strokeWidth={1} />
            <text x={PADL - 6} y={y(v) + 4} textAnchor="end" fontSize={10} fill="#8a8d94" fontFamily="var(--font-mono)">{v}</text>
          </g>
        ))}

        {/* scene cuts */}
        {layers.cuts && cuts.map((s, i) => (
          <line key={`c${i}`} x1={x(s)} x2={x(s)} y1={PADT} y2={H - PADB} stroke="var(--ink)" strokeWidth={1.2} strokeDasharray="4 4" opacity={0.6} />
        ))}

        {/* retention (dashed) + attention (solid) */}
        {layers.retention && <path d={pathR(R)} fill="none" stroke="#7c6f5d" strokeWidth={1.5} strokeDasharray="5 4" />}
        <path d={pathA(A)} fill="none" stroke="var(--ink)" strokeWidth={2.5} strokeLinejoin="round" />

        {/* per-second dots */}
        {A.map((v, i) => {
          const inDrop = drops.some(([a, b]) => i >= a && i < b);
          const fill = inDrop ? "#f59e3b" : "#fff";
          return <circle key={`p${i}`} cx={x(i)} cy={y(v)} r={3.4} fill={fill} stroke="var(--ink)" strokeWidth={1.6} />;
        })}

        {/* emotion dots (demo build) */}
        {layers.emotion && report.demo && A.map((v, i) => (
          <circle key={`e${i}`} cx={x(i)} cy={y(v) - 7} r={2.6} fill={DEMO_EMOS[i % 3].color} stroke="var(--ink)" strokeWidth={0.8} opacity={0.9} />
        ))}

        {/* peak chip */}
        <g>
          <circle cx={x(peak)} cy={y(A[peak] ?? 50)} r={5.5} fill="#fff" stroke="var(--ink)" strokeWidth={2} />
          <rect x={Math.max(x(peak) - 34, PADL)} y={Math.max(y(A[peak] ?? 50) - 30, PADT)} width={62} height={16} rx={3} fill="var(--yellow)" stroke="var(--ink)" strokeWidth={1.4} />
          <text x={Math.max(x(peak) - 34, PADL) + 31} y={Math.max(y(A[peak] ?? 50) - 18, PADT + 12)} fontSize={10} fontFamily="var(--font-mono)" fill="var(--ink)" textAnchor="middle">Peak {fmt(peak)}</text>
        </g>

        {/* playhead + timecode chip (synced with the video) */}
        <g>
          <line x1={tToX(playT)} x2={tToX(playT)} y1={PADT - 4} y2={H - PADB} stroke="var(--ink)" strokeWidth={2} />
          <circle cx={tToX(playT)} cy={y(A[Math.round(playT)] ?? 50)} r={6.5} fill="#fff" stroke="var(--ink)" strokeWidth={3} />
          <rect x={Math.min(Math.max(tToX(playT) - 24, PADL), W - 60)} y={0} width={48} height={17} rx={3} fill="var(--blue)" stroke="var(--ink)" strokeWidth={1.4} />
          <text x={Math.min(Math.max(tToX(playT) - 24, PADL) + 24, W - 36)} y={12.5} fontSize={10} fontFamily="var(--font-mono)" fill="#fff" textAnchor="middle">{fmt(playT)}</text>
        </g>

        {/* x labels */}
        {[0, Math.floor(T / 4), Math.floor(T / 2), Math.floor((3 * T) / 4), T - 1].map((t) => (
          <text key={t} x={x(t)} y={H - 8} textAnchor="middle" fontSize={10} fill="#8a8d94" fontFamily="var(--font-mono)">{fmt(t)}</text>
        ))}
      </svg>

      <h3 style={{ fontFamily: "var(--font-display)", fontWeight: 700, margin: "10px 0 0" }}>Chart layers</h3>
      <p style={{ fontSize: ".85rem", color: "var(--muted)", margin: "2px 0 8px" }}>Drag across the chart above to scrub the reel.</p>
      <div className="layer-row">
        {LAYER_DEFS.map((l) => (
          <button key={l.key} className={"layer-btn" + (layers[l.key] ? "" : " off")}
            onClick={() => setLayers({ ...layers, [l.key]: !layers[l.key] })}>
            {l.label}
          </button>
        ))}
      </div>
      <div className="chart-legend">
        <span>— Attention</span>
        <span>– – Viewers still watching</span>
        <span><i style={{ display: "inline-block", width: 9, height: 9, background: "var(--green-soft)", border: "1px solid var(--ink)", marginRight: 4 }} />Hook</span>
        <span><i style={{ display: "inline-block", width: 9, height: 9, background: "var(--red-soft)", border: "1px solid var(--ink)", marginRight: 4 }} />Drop zone</span>
        <span><i style={{ display: "inline-block", width: 9, height: 9, background: "var(--yellow)", border: "1px solid var(--ink)", marginRight: 4 }} />Best window</span>
        <span>¦ Scene cut</span>
      </div>
      <p style={{ fontSize: ".78rem", color: "var(--muted)", margin: "6px 0 0" }}>
        Attention is scored within this video, so the average sits near 50 by design.
      </p>
    </div>
  );
}

/* ══════════════ realtime player (PDF left column) ══════════════ */

function PlayerCard({ report, playT, onScrub, videoRef, playing, onPlayingChange }: {
  report: Report; playT: number; onScrub: (t: number) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>; playing: boolean; onPlayingChange: (p: boolean) => void;
}) {
  const dur = report.duration_s;
  const i = Math.min(Math.floor(playT), report.series.attention.length - 1);
  const att = report.series.attention[i] ?? 0;
  const still = Math.round((report.series.retention[i] ?? 0) * 100);

  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); } else { v.pause(); }
  };

  return (
    <div className="card a-player">
      {report.job_id.startsWith("job_") ? (
        <div className="player-frame">
          <video
            ref={videoRef}
            src={`${API}/api/media/${report.job_id}`}
            playsInline
            onPlay={() => onPlayingChange(true)}
            onPause={() => onPlayingChange(false)}
            onEnded={() => onPlayingChange(false)}
            onSeeked={(e) => onScrub(e.currentTarget.currentTime)}
          />
          <button className="play-overlay" onClick={toggle} aria-label={playing ? "Pause" : "Play"}
            style={{ pointerEvents: playing ? "none" : "auto" }}>
            {!playing && (
              <svg viewBox="0 0 24 24" fill="#fff" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
            )}
          </button>
          <div className="tc-chip">{fmt(playT)}</div>
          <div className="timecode">{tc(playT)}</div>
        </div>
      ) : (
        <div className="player-frame" style={{ minHeight: 200, display: "grid", placeItems: "center" }}>
          <span className="muted" style={{ fontSize: ".8rem", padding: 12, textAlign: "center" }}>
            Video preview needs a backend job — this is a client-side demo report.
          </span>
        </div>
      )}

      <div className="transport">
        <button className="tp-btn" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
          {playing ? "❚❚" : "▶"}
        </button>
        <span className="tc">{fmt(playT)} / {fmt(dur)}</span>
        <input
          type="range" min={0} max={Math.max(dur - 0.05, 0.05)} step={0.05}
          value={Math.min(playT, dur)}
          onChange={(e) => onScrub(parseFloat(e.target.value))}
          aria-label="Scrub the reel"
        />
      </div>
      <div className="vid-stats">
        <span>Attention <strong>{Math.round(att)}</strong></span>
        <span>Still here <strong>{still}%</strong></span>
        {report.demo && <span><strong style={{ color: "var(--blue)" }}>Neutral</strong> · Scene {Math.min(Math.floor(playT / 4) + 1, 5)}</span>}
      </div>
    </div>
  );
}

/* ══════════════ demo-data cards ══════════════ */

function EmotionsCard({ report }: { report: Report }) {
  if (!report.demo) {
    return (
      <div className="card a-emotions">
        <h2>Emotions</h2>
        <p className="muted" style={{ fontSize: ".88rem" }}>
          Emotion analysis isn't part of this build yet — retention and attention are.
        </p>
      </div>
    );
  }
  const mix = [
    { n: "Admiration", p: 50, c: DEMO_EMOS[0].color },
    { n: "Sorrow", p: 23, c: DEMO_EMOS[1].color },
    { n: "Romance", p: 14, c: DEMO_EMOS[2].color },
    { n: "No clear emoti…", p: 13, c: DEMO_EMOS[3].color },
  ];
  return (
    <div className="card a-emotions">
      <span className="pill pill-yellow">Demo data</span>
      <h2 className="mt-8" style={{ marginBottom: 2 }}>Emotions</h2>
      <p className="muted" style={{ fontSize: ".82rem", margin: "0 0 10px" }}>Predicted for an average viewer, not a measured mood.</p>
      <div className="emotion-stack">
        {mix.map((m) => <i key={m.n} style={{ width: `${m.p}%`, background: m.c }} />)}
      </div>
      <div className="chart-legend">
        {mix.map((m) => <span key={m.n}><i style={{ display: "inline-block", width: 9, height: 9, background: m.c, border: "1px solid var(--ink)", marginRight: 4 }} />{m.n}</span>)}
      </div>
      <div className="strength-box">
        <span className="n">71</span>
        <span style={{ fontSize: ".82rem" }}><strong>Emotional strength.</strong> Emotional peaks 5s of {Math.round(report.duration_s)}s (23%). No audible laughter in the audio.</span>
      </div>
      <p style={{ fontSize: ".85rem", margin: "8px 0" }}>Emotional peaks</p>
      <div>
        <span className="peak-chip">0:10–0:11 Sorrow</span>
        <span className="peak-chip">0:18–0:22 Sorrow</span>
      </div>
      <p style={{ fontSize: ".85rem", margin: "10px 0 4px" }}>Mix</p>
      {mix.map((m) => (
        <div key={m.n} className="mix-row">
          <span className="lbl">{m.n}</span>
          <span className="bar"><i style={{ width: `${m.p}%`, background: m.c }} /></span>
          <span className="pct">{m.p}%</span>
        </div>
      ))}
      <p style={{ fontSize: ".82rem", margin: "8px 0 0" }}><strong>Sequence.</strong> Admiration → Sorrow → Romance</p>
      <p style={{ fontSize: ".82rem", margin: "4px 0 0" }}><strong>Arc.</strong> Fall · Confidence. Low</p>
    </div>
  );
}

function CaptionFitCard({ report }: { report: Report }) {
  const hashtags = report.caption?.match(/#[\w]+/g) ?? [];
  const mentions = report.caption?.match(/@[\w]+/g) ?? [];
  const verdict = !report.caption ? "No caption" : hashtags.length >= 2 ? "Fits the video" : "Mostly fits";
  return (
    <div className="card a-caption">
      <h2>Caption fit <span className="pill pill-lime" style={{ marginLeft: 8 }}>{verdict}</span></h2>
      {report.caption
        ? <div className="caption-box">{report.caption}</div>
        : <p className="muted" style={{ fontSize: ".88rem" }}>No caption was provided — add one to check the fit.</p>}
      {(hashtags.length > 0 || mentions.length > 0) && (
        <div className="tag-row">{[...hashtags, ...mentions].map((t, i) => <span key={i} className="pill pill-blue">{t}</span>)}</div>
      )}
      {report.demo && report.caption && (
        <>
          <div className="fit-row"><span>Caption and visuals</span><span className="pill pill-lime" style={{ marginLeft: "auto" }}>Match</span></div>
          <div style={{ fontSize: ".82rem", color: "var(--muted)" }}>The caption frames the same story the visuals tell.</div>
          <div className="fit-row"><span>Caption and speech</span><span className="pill pill-lime" style={{ marginLeft: "auto" }}>Match</span></div>
          <div style={{ fontSize: ".82rem", color: "var(--muted)" }}>The caption echoes the spoken hook.</div>
          <div className="fit-row"><span>Hashtags and mentions</span><span className="pill pill-lime" style={{ marginLeft: "auto" }}>Match</span></div>
          <div style={{ fontSize: ".82rem", color: "var(--muted)" }}>Tags fit the tone of the reel.</div>
        </>
      )}
    </div>
  );
}

function TranscriptCard({ report }: { report: Report }) {
  return (
    <div className="card a-transcript">
      <h2>Transcript used for judging</h2>
      {report.demo ? (
        <>
          <p className="muted" style={{ fontSize: ".8rem", margin: "-4px 0 6px" }}>Demo data · Hinglish · detected en · 41 words</p>
          <div className="tr-row"><span className="t">0:00</span><span>You are looking very nice. To main thik karta hoon na tumko.</span></div>
          <div className="tr-row"><span className="t">0:05</span><span>Achha, you mean fushiya pink? Vahi.</span></div>
        </>
      ) : (
        <p className="muted" style={{ fontSize: ".88rem" }}>
          Transcription isn't part of this build yet — the reel is judged on video and audio.
        </p>
      )}
    </div>
  );
}

/* ══════════════ export modal (PDF) ══════════════ */

function ExportModal({ report, onClose }: { report: Report; onClose: () => void }) {
  const [tab, setTab] = useState<"json" | "md">("json");
  const md = useMemo(() => {
    const s = report.scores;
    return [
      `# Phoenix report — ${report.video}`, "",
      `- Viral potential: ${s.virality} (${s.virality_tag}) — ${report.method}`,
      `- Hook: ${s.hook_pct}% · Hold: ${s.hold_rate_pct}% · Finish: ${s.finish_pct}%`, "",
      "## Drop-off zones", ...report.moments.drop_zones_s.map(([a, b]) => `- ${fmt(a)}–${fmt(b)}`), "",
      "## Hooks", ...report.moments.hooks_s.map(([a, b]) => `- ${fmt(a)}–${fmt(b)}`), "",
      "## Editing suggestions",
      ...report.suggestions.map((x) => `- ${x.at}: ${x.title} (est. viewers lost ${x.est_lost_pct}%)`),
    ].join("\n");
  }, [report]);
  const text = tab === "json" ? JSON.stringify(report, null, 2) : md;
  const download = () => {
    const blob = new Blob([text], { type: tab === "json" ? "application/json" : "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `phoenix-report.${tab === "json" ? "json" : "md"}`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Export report</h2>
          <button className="x" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="export-row">
          <button className="btn btn-white" onClick={() => window.print()}>PDF</button>
          <button className="btn btn-lime" onClick={() => setTab("json")}>JSON</button>
          <button className="btn btn-white" onClick={() => setTab("md")}>Markdown</button>
        </div>
        <div className="export-preview">{text}</div>
        <button className="btn btn-blue" onClick={download}>↓ Download phoenix-report.{tab === "json" ? "json" : "md"}</button>
        <p className="export-note">Full data: timeline per second, zones, hooks, suggestions and caption.</p>
      </div>
    </div>
  );
}

/* ══════════════ main flow ══════════════ */

export default function HomePage() {
  const [sampleIdx, setSampleIdx] = useState(0);
  const [fileName, setFileName] = useState("");
  const [caption, setCaption] = useState(DEFAULT_CAPTION);
  const [phase, setPhase] = useState<"upload" | "processing" | "report">("upload");
  const [stage, setStage] = useState(0);
  const [statusDetail, setStatusDetail] = useState("");
  const [overall, setOverall] = useState(0);
  const [t0, setT0] = useState(0);
  const [report, setReport] = useState<Report | null>(null);
  const [jobId, setJobId] = useState("");
  const [off, setOff] = useState<Record<string, boolean>>({});
  const [playT, setPlayT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [layers, setLayers] = useState<Record<string, boolean>>({ retention: true, zones: true, cuts: false, emotion: false });
  const [showExport, setShowExport] = useState(false);

  const hashtags = useMemo(() => caption.match(/#[\w]+/g) ?? [], [caption]);
  const mentions = useMemo(() => caption.match(/@[\w]+/g) ?? [], [caption]);

  // chart scrub → seek the video; video time → playhead (both directions)
  const seek = (t: number) => {
    setPlayT(t);
    const v = videoRef.current;
    if (v && Math.abs(v.currentTime - t) > 0.02) v.currentTime = t;
  };

  // per-frame tracking while playing (rAF beats onTimeUpdate's ~4Hz)
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) setPlayT(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const analyze = async () => {
    setPhase("processing");
    setStage(0);
    setOverall(2);
    setStatusDetail("Uploading…");
    setT0(Date.now());
    setReport(null);
    try {
      const body = new FormData();
      body.append("sample", SAMPLES[sampleIdx].name);
      body.append("caption", caption);
      const res = await fetch(`${API}/api/analyze`, { method: "POST", body });
      const { job_id }: { job_id: string } = await res.json();
      setJobId(job_id);

      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
      for (let poll = 0; poll < 600; poll++) {
        await sleep(2000);
        const r = await fetch(`${API}/api/report/${job_id}`);
        const data = await r.json();
        if (data.series) {
          setStage(STAGES.length - 1);
          setReport(data as Report);
          setPhase("report");
          return;
        }
        if (data.status === "failed") throw new Error(data.detail || "Something went wrong. Please try again.");
        if (data.status === "queued") continue;
        setStage(Math.min(data.stage_index ?? 0, STAGES.length - 1));
        setOverall(data.progress ?? 0);
        setStatusDetail(data.detail || "");
      }
      throw new Error("This is taking unusually long — showing a demo report instead.");
    } catch (err) {
      setStatusDetail(String((err as Error).message || err));
      const T = SAMPLES[sampleIdx].durS;
      const retention: number[] = [];
      for (let i = 0; i < T; i++) {
        retention.push(Math.min(100, Math.max(4, (88 * Math.exp(-0.035 * i) + 12 + Math.sin(i / 3) * 3 + (i % 7 === 3 ? 8 : 0)) / 100)));
      }
      const mean = retention.reduce((a, b) => a + b, 0) / T;
      const sd = Math.sqrt(retention.reduce((s, v) => s + (v - mean) ** 2, 0) / T) || 1;
      const attention = retention.map((r) => Math.max(0, Math.min(100, 50 + 15 * (r - mean) / sd)));
      const hooks: Array<[number, number]> = [];
      const drops: Array<[number, number]> = [];
      let i = 1;
      while (i < T) {
        if (retention[i] > retention[i - 1]) { const s = i; while (i < T && retention[i] > retention[i - 1]) i++; if (i - s >= 2) hooks.push([s, i]); }
        else if (retention[i] < retention[i - 1]) { const s = i; while (i < T && retention[i] < retention[i - 1]) i++; if (i - s >= 2) drops.push([s, i]); }
        else i++;
      }
      const hook = retention.slice(0, 3).reduce((a, b) => a + b, 0) / 3 * 100;
      const finish = retention[T - 1] * 100;
      setTimeout(() => {
        setReport({
          job_id: "demo-local", video: SAMPLES[sampleIdx].name + ".mp4", caption, duration_s: T, demo: true,
          method: "heuristic, within-video normalised, uncalibrated",
          scores: { hook_pct: hook, hook_band: "Strong", hold_rate_pct: Math.min(100, finish / Math.max(hook, 1) * 100), finish_pct: finish, virality: 71, virality_band: "Strong", virality_tag: "Popular" },
          virality_rows: [
            { key: "hook", label: "Hook (first 3s)", value: hook, points: hook * 0.3, vs_typical: 0.3 * (hook - 50), weight: 0.3, explain: `Attention in the first 3s scores ${Math.round(hook)}/100.` },
            { key: "drop_free", label: "Few drop-offs", value: 79, points: 19.8, vs_typical: 7.3, weight: 0.25, explain: "79% of the reel stays out of drop zones." },
            { key: "strong_moments", label: "Strong-attention moments", value: 43, points: 12.9, vs_typical: -2.1, weight: 0.3, explain: "21% of seconds at attention 70+." },
            { key: "ending", label: "Ending", value: Math.round(finish), points: 9, vs_typical: -6.1, weight: 0.15, explain: `Attention in the last 3s averages ${Math.round(finish)}/100.` },
          ],
          series: { retention, attention },
          moments: { peak_attention_at_s: attention.indexOf(Math.max(...attention)), hooks_s: hooks, drop_zones_s: drops },
          suggestions: drops.slice(0, 4).map(([a, b]) => ({ at: `${fmt(a)}–${fmt(b)}`, title: `Cut ${fmt(a)}–${fmt(b)} (${b - a}s of weak pull).`, why: `Retention falls here — viewers leave. Add a cut, zoom or re-hook.`, est_lost_pct: Math.round((retention[a] - retention[b - 1]) * 100) })),
          stages: STAGES,
        });
        setPhase("report");
      }, 350);
    }
  };

  /* ── processing view ── */
  if (phase === "processing") {
    const pct = Math.max(2, Math.min(100, overall));
    const elapsed = t0 ? Math.floor((Date.now() - t0) / 1000) : 0;
    return (
      <>
        <TopBar onExport={null} />
        <div className="wrap">
          <span className="pill pill-yellow">Job {jobId || "…"}</span>
          <h1 className="mt-8">Analysing your reel</h1>
          <p className="lead mono" style={{ fontSize: ".9rem" }}>{fileName || SAMPLES[sampleIdx].name + ".mp4"}</p>
          <div className="progress-meta"><span>Stage {stage + 1} of {STAGES.length}</span><span>{pct}% complete</span></div>
          <div className="progress-track"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
          {statusDetail && <p className="muted" style={{ fontSize: ".88rem", margin: "10px 0 0" }}>{statusDetail}</p>}
          <p className="muted" style={{ fontSize: ".8rem", margin: "4px 0 0" }}>
            {elapsed > 0 && <>Elapsed {fmt(elapsed)} · </>}Short reels usually finish within a couple of minutes.
          </p>
          <div className="mt-16">
            {STAGES.map((s, i) => (
              <div key={s} className={"stage-row" + (i === stage ? " running" : "")}>
                <span className={"stage-dot" + (i < stage ? " done" : i === stage ? " running" : "")}>
                  {i < stage ? "✓" : i + 1}
                </span>
                {s}
                <span className="state">{i < stage ? "Done" : i === stage ? "Running" : "Queued"}</span>
              </div>
            ))}
          </div>
          <button className="btn btn-white mt-16" onClick={() => setPhase("upload")}>Cancel</button>
        </div>
      </>
    );
  }

  /* ── report view ── */
  if (phase === "report" && report) {
    const rows = report.virality_rows.filter((r) => !off[r.key]);
    const base = rows.reduce((s, r) => s + r.weight * 50, 0);
    const total = rows.reduce((s, r) => s + r.weight * r.value, 0);
    const boosts = [...report.virality_rows].filter((r) => !off[r.key]).sort((a, b) => b.vs_typical - a.vs_typical);
    const best = boosts[0], worst = boosts[boosts.length - 1];

    return (
      <>
        <TopBar onExport={() => setShowExport(true)} />
        <div className="wrap-wide">
          <div className="report-grid">

            <PlayerCard report={report} playT={playT} onScrub={seek} videoRef={videoRef} onPlayingChange={setPlaying} playing={playing} />

            <ChartCard report={report} playT={playT} onScrub={seek} layers={layers} setLayers={setLayers} />

            {/* middle column: score → why viral → caption fit */}
            <div className="a-mid">
            <div className="card score-tile">
              <div className="score-copy">
                <div><span className="pill pill-yellow">{report.demo ? "Demo data" : "Model output"}</span></div>
                <div className="title mt-8">{SAMPLES[sampleIdx].label}</div>
                <div className="sub">{report.video.split("/").pop()} · {fmt(report.duration_s)}</div>
                <p className="boost">
                  Biggest boost: <strong>{best?.label ?? "—"}</strong>{best ? `, +${best.vs_typical.toFixed(1)} points vs a typical reel` : ""}.<br />
                  Biggest drag: <strong>{worst?.label ?? "—"}</strong>{worst ? `, ${worst.vs_typical.toFixed(1)} points` : ""}.
                </p>
                <button className="btn btn-lime" style={{ fontSize: ".85rem", padding: "10px 12px" }}
                  onClick={() => document.getElementById("suggestions")?.scrollIntoView({ behavior: "smooth" })}>
                  Edit video with suggested fixes
                </button>
              </div>
              <div className="score-num">
                <div className="big">{Math.round(report.scores.virality)}</div>
                <div className="tagline">{report.scores.virality_tag}</div>
              </div>
            </div>

            {/* why viral potential */}
            <div className="card a-why">
              <h2 style={{ display: "flex", alignItems: "center", gap: 8 }}>
                Why viral potential <span className="pill pill-blue">{Math.round(report.scores.virality)} · {report.scores.virality_tag}</span>
              </h2>
              <p className="muted" style={{ margin: "-6px 0 8px", fontSize: ".85rem" }}>Switch a factor off to see the score without it.</p>
              {report.virality_rows.map((r) => {
                const on = !off[r.key];
                return (
                  <div key={r.key} className="factor-row">
                    <div className="factor-head">
                      <span className={"switch" + (on ? "" : " off")} role="switch" aria-checked={on} aria-label={r.label}
                        onClick={() => setOff((o) => ({ ...o, [r.key]: on }))} />
                      <span className="name">{r.label}</span>
                      <span className="pts">{on ? (r.vs_typical >= 0 ? "+" : "") + r.vs_typical.toFixed(1) : "0.0"}</span>
                    </div>
                    <div className="factor-bar">
                      <i style={{ width: `${Math.max(0, Math.min(100, on ? r.value : 0))}%`, background: r.value >= 55 ? "var(--lime)" : r.value >= 40 ? "var(--yellow)" : "var(--red-soft)", borderRight: "2px solid var(--line)" }} />
                    </div>
                    <div className="factor-why">{r.explain} <code>×{r.weight} · {r.points} pts · vs a typical reel (factor at 50)</code></div>
                  </div>
                );
              })}
              <div className="factor-row" style={{ background: "var(--bg)" }}>
                <div style={{ font: "700 .92rem var(--font-sans)", marginBottom: 6 }}>How the score is calculated</div>
                <div className="mono" style={{ fontSize: ".82rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}><span>Attention base</span><span>{base.toFixed(1)}</span></div>
                  {rows.filter((r) => r.key !== "hook").map((r) => (
                    <div key={r.key} style={{ display: "flex", justifyContent: "space-between" }}>
                      <span>+ {r.label}</span><span>{r.vs_typical >= 0 ? "+" : ""}{r.vs_typical.toFixed(1)}</span>
                    </div>
                  ))}
                  <div style={{ display: "flex", justifyContent: "space-between", borderTop: "2px solid var(--line)", marginTop: 4, paddingTop: 4, fontWeight: 700 }}>
                    <span>Viral potential</span><span>{total.toFixed(1)} → {Math.round(total)}</span>
                  </div>
                </div>
              </div>
              <p className="chart-note">Uncalibrated heuristic. Weights are set by hand until fitted on real view counts. It judges the video only, never the account.</p>
            </div>

            <CaptionFitCard report={report} />
            </div>{/* /a-mid */}

            {/* right column: emotions → suggestions → transcript */}
            <div className="a-right">
            <EmotionsCard report={report} />
            <div className="card a-suggest" id="suggestions">
              <h2>Editing suggestions</h2>
              <p className="muted" style={{ margin: "-6px 0 10px", fontSize: ".88rem" }}>Ranked by viewers lost.</p>
              <button className="btn btn-lime" style={{ fontSize: ".88rem", padding: "10px 12px" }}>Edit video with these suggestions</button>
              <button className="btn btn-white" style={{ fontSize: ".85rem", padding: "9px 12px", marginTop: 8 }}
                onClick={(e) => { const b = e.currentTarget; b.textContent = "Model check isn't part of this demo build"; setTimeout(() => { b.textContent = "Verify top 3 edits (about 2–3 min each)"; }, 2200); }}>
                Verify top 3 edits (about 2–3 min each)
              </button>
              {report.suggestions.length === 0 && <p className="muted mt-8">No edits suggested for this version.</p>}
              {report.suggestions.map((s, i) => (
                <div key={i} className="suggestion-card">
                  <span className="at">{s.at}</span>
                  <span className="pill pill-soft" style={{ marginLeft: 6 }}>Not model-checked yet</span>
                  <div className="title" style={{ marginTop: 6 }}>{s.title}</div>
                  <div className="why">{s.why}</div>
                  <div style={{ fontSize: ".82rem", marginTop: 4 }}>Est. viewers lost here: <strong>{s.est_lost_pct}%</strong> <span className="muted">(model estimate)</span></div>
                </div>
              ))}
            </div>

            <TranscriptCard report={report} />
            </div>{/* /a-right */}
          </div>

          <p className="note mt-24" style={{ fontSize: ".8rem" }}>{report.method} · judges the video only, never the account.</p>
          <div style={{ display: "flex", gap: 12 }}>
            <button className="btn btn-white mt-16" onClick={() => { setPhase("upload"); setReport(null); }}>New analysis</button>
            <a className="btn btn-lime mt-16" href="/reports">All reports</a>
          </div>
        </div>
        {showExport && <ExportModal report={report} onClose={() => setShowExport(false)} />}
      </>
    );
  }

  /* ── upload view ── */
  return (
    <>
      <TopBar onExport={null} />
      <div className="wrap">
        <h1>Check your reel before you post it.</h1>
        <p className="lead">Add the video and your caption. Phoenix shows where viewers leave, what they feel and what to fix.</p>

        <section className="card">
          <h2>1. Video</h2>
          <label className="btn btn-blue" style={{ cursor: "pointer" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
            {fileName ? "Choose a different video" : "Choose a video"}
            <input type="file" accept="video/*" style={{ display: "none" }}
              onChange={(e) => setFileName(e.target.files?.[0]?.name ?? "")} />
          </label>
          <div className="file-row">
            <span>{fileName || SAMPLES[sampleIdx].name + ".mp4"}</span>
            <span className="pill pill-lime">Sample reel</span>
          </div>
          <p className="muted" style={{ margin: "10px 0 0", fontSize: ".9rem" }}>No file handy? Load a sample reel.</p>
          {SAMPLES.map((s, i) => (
            <button key={s.name} className={"sample-btn" + (sampleIdx === i ? " selected" : "")} onClick={() => { setSampleIdx(i); setFileName(""); }}>
              <span>{s.label} · {s.dur}</span>
              <span className="dur">{s.dur}</span>
            </button>
          ))}
        </section>

        <section className="card">
          <h2>2. Caption</h2>
          <textarea value={caption} onChange={(e) => setCaption(e.target.value)}
            placeholder="Write the caption you will post, with #hashtags and @mentions." />
          <p className="count-note">{hashtags.length} hashtags · {mentions.length} mention{mentions.length === 1 ? "" : "s"}</p>
          {(hashtags.length > 0 || mentions.length > 0) && (
            <div className="tag-row">
              {hashtags.map((t, i) => <span key={`h${i}`} className="pill pill-blue">{t}</span>)}
              {mentions.map((t, i) => <span key={`m${i}`} className="pill pill-lime">{t}</span>)}
            </div>
          )}
        </section>

        <button className="btn btn-lime mt-8" onClick={analyze}>Analyze</button>
        <p className="note">Demo build. Reports use sample data.</p>
      </div>
    </>
  );
}

function TopBar({ onExport }: { onExport: (() => void) | null }) {
  return (
    <header className="topbar">
      <span className="logo" aria-hidden="true">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3c1 3-2 4-2 7a4 4 0 0 0 8 .5c2 1.5 3 3.5 3 5.5a7 7 0 0 1-14 0c0-5 4-6 5-13z" />
        </svg>
      </span>
      <span className="brand">Project Phoenix</span>
      <nav>
        <a href="/">New</a>
        {onExport
          ? <a href="#export" className="accent" onClick={(e) => { e.preventDefault(); onExport(); }}>Export</a>
          : <a href="/reports" className="accent">Reports</a>}
      </nav>
    </header>
  );
}
