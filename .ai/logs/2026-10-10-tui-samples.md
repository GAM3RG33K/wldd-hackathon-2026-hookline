# 2026-10-10 — TUI + samples corpus decision
- Corpus found: `samples/{face,faceless,outliers}/` = 50+50+10 reels + metadata.csv (video_id, handle, follower_band, views, page_type), ~633MB, 116 files. Redemption of the "110 reels" PENDING item.
- Built `tui.py` (stdlib only): random pick, preview/Space analyze via backend, saved-report review, health. Verified piped: review renders + random FC013 → job 60d65bcb with timestamped recs.
- Backend restarted: SPACE_ID now wlddhq/hookline-tribe. Space RUNNING (traceback commit).
- Storage: created private `wlddhq/hookline-samples`, uploaded 114 files (~502MB, .DS_Store excluded), commit 49794d4. Verified listing: 50 FC + 50 FL + 10 OL + 3 CSVs + README.
- 110-reel corpus PENDING resolved (was missing at harness time; now local + Hub).
- Harness note: metadata handle/views = validation context only, never model input — TUI labels it as such.
