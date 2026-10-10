"""Per-second emotion cues for a reel (GPU side; fusion happens in hookline_report).

Predicts the emotion the video EVOKES for an average viewer, not any viewer's actual mood.
Channels (each optional, all per second):
  visual  CLIP zero-shot on 1 fps frames against emotion prompts
  voice   wav2vec2 arousal / valence regression on seconds with speech
  text    GoEmotions classifier on transcript segments (English-trained; Hinglish is partial)
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
                     ("voice", lambda: _voice(np.zeros(32000, dtype=np.float32), 16000, [1.0], device)),
                     ("text", lambda: _text([{"start": 0, "end": 1, "text": "warm up"}], 1, device)),
                     ("music", lambda: _music(np.zeros(32000, dtype=np.float32), 2, device))):
        try:
            fn()
            status[name] = "ok"
        except Exception as e:  # noqa: BLE001
            status[name] = f"{type(e).__name__}: {e}"[:200]
    return status


def emotion_cues(video, audio, sr, speech_prob, segments, device, log=print):
    """Run every channel; a failing channel is skipped (logged), never fatal."""
    out, n = {}, len(speech_prob)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        frames = _frames(video, tmp)
        log(f"emotion: {len(frames)} frames in {time.time() - t0:.1f}s")
        for name, fn in (("visual", lambda: _clip(frames, device)),):
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
