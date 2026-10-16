"use client";

import { useEffect, useState } from "react";

type Item = {
  job_id: string;
  video: string;
  duration_s: number;
  virality: number | null;
  virality_tag: string;
  demo: boolean;
};

export default function ReportsPage() {
  const [items, setItems] = useState<Item[] | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/reports")
      .then((r) => r.json())
      .then(setItems)
      .catch(() => setItems([]));
  }, []);

  return (
    <>
      <header className="topbar">
        <span className="logo" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3c1 3-2 4-2 7a4 4 0 0 0 8 .5c2 1.5 3 3.5 3 5.5a7 7 0 0 1-14 0c0-5 4-6 5-13z" />
          </svg>
        </span>
        <span className="brand">Project Phoenix</span>
        <nav>
          <a href="/">New</a>
          <a href="/reports" className="accent">Reports</a>
        </nav>
      </header>

      <div className="wrap">
        <h1>Reports</h1>
        <p className="lead">Every reel you have analysed, newest first.</p>

        {items === null && <p className="muted">Loading…</p>}
        {items !== null && items.length === 0 && (
          <div className="card"><p style={{ margin: 0 }}>No reports yet — analyse a reel from the <a href="/">New</a> page.</p></div>
        )}
        {items !== null && items.map((it) => (
          <div key={it.job_id} className="card" style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <div style={{ font: "700 1rem var(--font-display)" }}>{it.video || "untitled"}</div>
              {it.demo && <span className="pill pill-yellow">Demo data</span>}
              <span className="pill pill-soft mono">{it.duration_s ?? "?"}s</span>
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
                {it.virality != null && <span className="pill pill-blue">{Math.round(it.virality)} · {it.virality_tag}</span>}
                <span className="pill pill-soft mono">{it.job_id}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
