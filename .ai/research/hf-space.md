# HF Space — GPU backend (decisions 2026-10-10/11, replaces Colab)

- Space `wlddhq/hookline-tribe`, Docker + FastAPI (not Gradio), private. Hardware a100-large ($2.50/hr), sleep 1800s. t4-small took ~20 min for a 28s reel; A100 ~2.5–2.8s of GPU per video-second end to end.
- Speed work: fp16 autocast, resident models (`KEEP_GB=4`, OOM → free + retry), 10s chunks (Wav2Vec-BERT OOM), transcribe before chunking (upstream duplicated words n-fold), cooperative cancel, per-step timing and chunk-based ETA.
- Pins that matter: torch 2.5.1 (transformers ≥5 needs torch ≥2.6 → pin `<5`; `.bin` weights refused → safetensors-only models), torchvision imported on the main thread (circular import), non-root HOME/caches in `/tmp`, spaCy model installed at build.
- Emotion models warm at boot (`boot.emotion_models`); face channel adds OpenCV + YuNet ONNX (<1s per reel on GPU, 88 frames in 5.7s on a Mac CPU).
- Deploy: `hf upload wlddhq/hookline-tribe hf_space . --repo-type space --exclude "__pycache__/*"`; ~10–15 min build; wait for `RUNNING` (`RUNNING_BUILDING` = old container still serving).
- Secrets: `HF_TOKEN` (Llama-3.2-3B access). Local token via `hf auth token`, never committed.
