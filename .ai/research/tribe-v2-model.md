# TRIBE v2 — model notes (source: `tribe/repo/README.md`, `tribe/tribe_analyze.py`)

- Foundation multimodal brain-encoding model: video+audio+text → predicted fMRI on fsaverage5 cortical mesh (~20k verts; here `preds (T,20484)`).
- Entry: `TribeModel.from_pretrained("facebook/tribev2", cache_folder, device, config_update)` then `get_events_dataframe(video_path)` → `predict(events)` → `(preds, segments)` (`tribe_analyze.py:56-73`).
- Encoders (per `prefetch.py` + notebook): `w2v-bert-2.0` (audio), `vjepa2-vitg-fpc64-256` (video), `Llama-3.2-3B` (text, gated — needs `HF_TOKEN`), brain head = Transformer `FmriEncoder`.
- Predictions are average-subject, offset −5s for hemodynamic lag (repo README).
- Features configurable per-device: `data.{text_feature,audio_feature,video_feature.image,image_feature.image}.device` (`tribe_analyze.py:54`).
- Transcript: `.tsv` written next to extracted wav; word-level `{text,start,duration,sequence_id,sentence}`. Local default `TRIBE_WHISPER_MODEL=small`; Colab patches in faster-whisper in-process path + whisperx `uvx --python 3.12`, `int8` on CPU.
- Atlas: nilearn Destrieux surf; groups in `hookline_report.py:37-49`. Vert-count mismatch or no network → fallback whole-cortex norm + stderr warn, never crash (`hookline_report.py:148-165`).
- License: CC-BY-NC-4.0 (repo README). Citation: d'Ascoli et al. 2026, arXiv:2605.04326.
