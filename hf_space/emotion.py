"""Per-second emotion cues for a reel (GPU side; fusion happens in hookline_report).

Predicts the emotion the video EVOKES for an average viewer, not any viewer's actual mood.
Channels (each optional, all per second):
  visual  CLIP zero-shot on 1 fps frames against emotion prompts
  voice   wav2vec2 arousal / valence regression on seconds with speech
  text    GoEmotions classifier on transcript segments (English-trained; Hinglish is partial)
  face    YuNet face detection + ViT expression classifier on 1 fps frames (emotion portrayed on screen)
Sounds (laughter, cheering, gasps...) come from the AST pass in the speech gate.
"""
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# safetensors only: transformers refuses .bin weights on torch < 2.6 (CVE), and TRIBE pins torch 2.5.1.
# openai/clip-vit-base-patch32 ships .bin only; LAION's B/32 (MIT) ships safetensors.
CLIP_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
VOICE_REPO = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # CC-BY-NC-SA: prototype only
TEXT_REPO = "SamLowe/roberta-base-go_emotions"
MUSIC_REPO = "laion/clap-htsat-fused"  # Apache-2.0, safetensors (the music-only CLAPs are .bin)
FACE_REPO = "trpakov/vit-face-expression"  # Apache-2.0, safetensors; FER2013 labels: angry disgust fear happy neutral sad surprise
YUNET = Path(__file__).with_name("models") / "face_detection_yunet_2023mar.onnx"  # OpenCV face detector (Apache-2.0)
GALLERY = Path(__file__).with_name("caption_gallery.npy")  # CLIP frames of other reels (tribe/build_caption_gallery.py)
EMBED_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Apache-2.0, 50+ languages incl. Hindi
# Music mood prompts (two per mood); per-second softmax over moods from 4s windows.

# Two CLIP prompts per emotion plus a neutral anchor; scores are averaged per emotion.
_M = {}


def _frames(video, out_dir, fps=1, width=640):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},scale={width}:-2",
                    str(Path(out_dir) / "f%05d.jpg")], check=True)
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in sorted(Path(out_dir).glob("f*.jpg"))]


def _clip(frames, device):
    """Raw cosine similarity per frame to every taxonomy visual prompt (mapping to emotions is local)."""
    import taxonomy as TX
    from transformers import CLIPModel, CLIPProcessor
    if "clip" not in _M:
        _M["clip"] = (CLIPModel.from_pretrained(CLIP_REPO, torch_dtype=torch.float32, use_safetensors=True)
                      .eval().to(device), CLIPProcessor.from_pretrained(CLIP_REPO))
    model, proc = _M["clip"]
    prompts = list(TX.VISUAL_PROMPTS)
    with torch.inference_mode():
        tin = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        tf = model.get_text_features(**tin)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        out = []
        for i in range(0, len(frames), 32):
            iin = proc(images=frames[i:i + 32], return_tensors="pt").to(device)
            imf = model.get_image_features(**iin)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            out.append((imf @ tf.T).float().cpu().numpy())
    p = np.concatenate(out) if out else np.zeros((0, len(prompts)))
    return {"prompt_sim": {k: [round(float(x), 4) for x in p[:, j]] for j, k in enumerate(prompts)}}


def _voice(audio, sr, speech_prob, device):
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class EmotionModel(Wav2Vec2PreTrainedModel):  # architecture from the model card
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            return self.classifier(torch.mean(self.wav2vec2(input_values)[0], dim=1))

    if "voice" not in _M:
        _M["voice"] = (EmotionModel.from_pretrained(VOICE_REPO, torch_dtype=torch.float32, use_safetensors=True)
                       .eval().to(device),
                       Wav2Vec2Processor.from_pretrained(VOICE_REPO))
    model, proc = _M["voice"]
    n = len(speech_prob)
    arousal, valence = [None] * n, [None] * n
    for i in range(n):
        if speech_prob[i] < 0.4:
            continue
        seg = audio[max(0, (i - 0) * sr):(i + 2) * sr]
        if len(seg) < sr // 2:
            continue
        x = proc(seg, sampling_rate=sr, return_tensors="pt")["input_values"].to(device)
        with torch.inference_mode():
            a, _d, v = model(x)[0].float().cpu().numpy()  # arousal, dominance, valence (~0..1)
        arousal[i], valence[i] = round(float(a), 3), round(float(v), 3)
    return {"arousal": arousal, "valence": valence}


def _music(audio16k, n, device):
    """Raw CLAP similarity per second (4s window) to every taxonomy music prompt."""
    import taxonomy as TX
    from transformers import ClapModel, ClapProcessor
    import torchaudio.functional as AF
    if "music" not in _M:
        _M["music"] = (ClapModel.from_pretrained(MUSIC_REPO, torch_dtype=torch.float32, use_safetensors=True)
                       .eval().to(device), ClapProcessor.from_pretrained(MUSIC_REPO))
    model, proc = _M["music"]
    prompts = list(TX.MUSIC_PROMPTS)
    y = AF.resample(torch.from_numpy(np.asarray(audio16k, dtype=np.float32)), 16000, 48000).numpy()
    sr = 48000
    with torch.inference_mode():
        tin = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        tf = model.get_text_features(**tin)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        clips = [y[max(0, int((t - 1.5) * sr)):int((t + 2.5) * sr)] for t in range(n)]
        clips = [c if len(c) >= sr else np.pad(c, (0, sr - len(c))) for c in clips]
        out = []
        for i in range(0, n, 16):
            ain = proc(audios=clips[i:i + 16], sampling_rate=sr, return_tensors="pt").to(device)
            af = model.get_audio_features(**ain)
            af = af / af.norm(dim=-1, keepdim=True)
            out.append((af @ tf.T).float().cpu().numpy())
    p = np.concatenate(out) if out else np.zeros((0, len(prompts)))
    return {"prompt_sim": {k: [round(float(x), 4) for x in p[:, j]] for j, k in enumerate(prompts)}}


def _face(frames, device):
    """Facial expression of the biggest face per frame (1 fps): the emotion PORTRAYED on screen.
    Raw label probabilities only; mapping to the taxonomy and smoothing happen locally."""
    import cv2
    from transformers import pipeline
    if "face" not in _M:
        _M["face"] = (cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.7, 0.3, 5000),
                      pipeline("image-classification", model=FACE_REPO, top_k=None,
                               device=0 if device == "cuda" else -1, model_kwargs={"use_safetensors": True}))
    det, clf = _M["face"]
    probs, count, area, crops, idx = [None] * len(frames), [0] * len(frames), [0.0] * len(frames), [], []
    for t, im in enumerate(frames):
        img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is None or not len(faces):
            continue
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])[:4]
        count[t], area[t] = int(len(faces)), round(float(fw * fh / (w * h)), 4)
        m = 0.2  # margin around the box: the classifier was trained on loosely cropped faces
        x0, y0 = int(max(0, x - m * fw)), int(max(0, y - m * fh))
        x1, y1 = int(min(w, x + fw * (1 + m))), int(min(h, y + fh * (1 + m)))
        if x1 - x0 < 24 or y1 - y0 < 24:
            continue
        crops.append(im.crop((x0, y0, x1, y1)))
        idx.append(t)
    for i in range(0, len(crops), 32):
        for t, res in zip(idx[i:i + 32], clf(crops[i:i + 32])):
            probs[t] = {d["label"]: round(float(d["score"]), 3) for d in res}
    return {"probs": probs, "faces": count, "area": area}


def _embed(texts, device):
    """Mean-pooled, L2-normalised multilingual sentence embeddings (no sentence-transformers dependency)."""
    from transformers import AutoModel, AutoTokenizer
    if "embed" not in _M:
        _M["embed"] = (AutoModel.from_pretrained(EMBED_REPO, use_safetensors=True).eval().to(device),
                       AutoTokenizer.from_pretrained(EMBED_REPO))
    model, tok = _M["embed"]
    with torch.inference_mode():
        x = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = model(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()


def _caption(caption, frames, transcript, device):
    """Raw relevance evidence for a creator caption: similarity of the caption AND of fixed unrelated
    control captions to every frame (CLIP) and to the transcript (multilingual embeddings), plus the
    caption's own GoEmotions labels. Deciding relevance and effect happens locally."""
    import taxonomy as TX
    texts = [caption] + list(TX.CAPTION_CONTROLS)
    out = {"text": caption, "controls": len(TX.CAPTION_CONTROLS)}
    model, proc = _M["clip"]
    with torch.inference_mode():
        tin = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        tf = model.get_text_features(**tin)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        sims = []
        for i in range(0, len(frames), 32):
            iin = proc(images=frames[i:i + 32], return_tensors="pt").to(device)
            imf = model.get_image_features(**iin)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            sims.append((imf @ tf.T).float().cpu().numpy())
    S = np.concatenate(sims) if sims else np.zeros((0, len(texts)))
    out["frame_sim"] = [round(float(x), 4) for x in S[:, 0]]              # caption vs each frame
    if GALLERY.exists():  # caption vs other reels: mean of each reel's top 3 of 8 frames (specificity)
        G = np.load(GALLERY).astype(np.float32)
        g = np.einsum("rfd,d->rf", G, tf[0].float().cpu().numpy())
        out["gallery_sim"] = [round(float(np.sort(r[~np.isnan(r)])[-3:].mean()), 4) for r in g]
    out["frame_control_sim"] = [[round(float(x), 4) for x in row] for row in S[:, 1:]]
    if transcript and transcript.strip():
        E = _embed([transcript[:2000]] + texts, device)
        cos = E[1:] @ E[0]
        out["transcript_sim"] = round(float(cos[0]), 4)
        out["transcript_control_sim"] = [round(float(x), 4) for x in cos[1:]]
    out["emotion"] = _text([{"start": 0, "end": 0, "text": caption}], 1, device)["probs"][0]
    return out


def _text(segments, n, device):
    from transformers import pipeline
    if "text" not in _M:
        _M["text"] = pipeline("text-classification", model=TEXT_REPO, top_k=None,
                              device=0 if device == "cuda" else -1, model_kwargs={"use_safetensors": True})
    clf = _M["text"]
    per_sec = [None] * n
    if not segments:
        return {"probs": per_sec}
    outs = clf([s["text"][:512] for s in segments])
    for seg, scores in zip(segments, outs):
        d = {x["label"]: round(float(x["score"]), 3) for x in scores}
        for t in range(int(seg["start"]), min(n, int(np.ceil(seg["end"])) + 1)):
            per_sec[t] = d
    return {"probs": per_sec}


def warm(device):
    """Load each model once at startup (independently: one failure does not block the others).
    Returns {channel: "ok" | error}."""
    import numpy as np
    from PIL import Image
    status = {}
    for name, fn in (("visual", lambda: _clip([Image.new("RGB", (64, 64))], device)),
                     ("face", lambda: _face([Image.new("RGB", (64, 64))], device)),
                     ("embed", lambda: _embed(["warm up"], device)),
                     ("voice", lambda: _voice(np.zeros(32000, dtype=np.float32), 16000, [1.0], device)),
                     ("text", lambda: _text([{"start": 0, "end": 1, "text": "warm up"}], 1, device)),
                     ("music", lambda: _music(np.zeros(32000, dtype=np.float32), 2, device))):
        try:
            fn()
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001
            status[name] = f"{type(e).__name__}: {e}"[:200]
    return status


def emotion_cues(video, audio, sr, speech_prob, segments, device, log=print, caption="", transcript=""):
    """Run every channel; a failing channel is skipped (logged), never fatal."""
    out, n = {}, len(speech_prob)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        frames = _frames(video, tmp)
        log(f"emotion: {len(frames)} frames in {time.time() - t0:.1f}s")
        chans = [("visual", lambda: _clip(frames, device)), ("face", lambda: _face(frames, device))]
        if caption:
            chans.append(("caption", lambda: _caption(caption, frames, transcript, device)))
        for name, fn in chans:
            t0 = time.time()
            try:
                out[name] = fn()
                log(f"emotion/{name}: {time.time() - t0:.1f}s")
            except Exception as e:  # noqa: BLE001
                log(f"emotion/{name} skipped: {type(e).__name__}: {e}")
    for name, fn in (("voice", lambda: _voice(audio, sr, speech_prob, device)),
                     ("text", lambda: _text(segments, n, device)),
                     ("music", lambda: _music(audio, n, device))):
        t0 = time.time()
        try:
            out[name] = fn()
            log(f"emotion/{name}: {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            log(f"emotion/{name} skipped: {type(e).__name__}: {e}")
    return out
