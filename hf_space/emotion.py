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

CLIP_REPO = "openai/clip-vit-base-patch32"
VOICE_REPO = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # CC-BY-NC-SA: prototype only
TEXT_REPO = "SamLowe/roberta-base-go_emotions"

EMOTIONS = ["amusement", "awe", "surprise", "curiosity", "tension", "outrage", "warmth", "sadness"]
# Two CLIP prompts per emotion plus a neutral anchor; scores are averaged per emotion.
CLIP_PROMPTS = {
    "amusement": ["a funny, hilarious moment", "people laughing at a joke"],
    "awe": ["an awe-inspiring, amazing sight", "an impressive, breathtaking moment"],
    "surprise": ["a shocking, surprising moment", "an unexpected twist"],
    "curiosity": ["a mysterious, intriguing scene", "someone revealing a secret"],
    "tension": ["a tense, scary moment", "a dangerous, nerve-wracking situation"],
    "outrage": ["an angry, outrageous scene", "people arguing furiously"],
    "warmth": ["a cute, heartwarming moment", "a loving, wholesome scene"],
    "sadness": ["a sad, emotional moment", "someone crying"],
    "neutral": ["an ordinary everyday scene", "a plain product shot"],
}
_M = {}


def _frames(video, out_dir, fps=1, width=640):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},scale={width}:-2",
                    str(Path(out_dir) / "f%05d.jpg")], check=True)
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in sorted(Path(out_dir).glob("f*.jpg"))]


def _clip(frames, device):
    from transformers import CLIPModel, CLIPProcessor
    if "clip" not in _M:
        _M["clip"] = (CLIPModel.from_pretrained(CLIP_REPO, torch_dtype=torch.float32).eval().to(device),
                      CLIPProcessor.from_pretrained(CLIP_REPO))
    model, proc = _M["clip"]
    labels = list(CLIP_PROMPTS)
    texts = [t for k in labels for t in CLIP_PROMPTS[k]]
    with torch.inference_mode():
        tin = proc(text=texts, return_tensors="pt", padding=True).to(device)
        tf = model.get_text_features(**tin)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        out = []
        for i in range(0, len(frames), 32):
            iin = proc(images=frames[i:i + 32], return_tensors="pt").to(device)
            imf = model.get_image_features(**iin)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            sim = (100.0 * imf @ tf.T).reshape(len(frames[i:i + 32]), len(labels), 2).mean(-1)
            out.append(torch.softmax(sim, dim=-1).cpu().numpy())
    p = np.concatenate(out) if out else np.zeros((0, len(labels)))
    return {k: [round(float(x), 3) for x in p[:, j]] for j, k in enumerate(labels)}


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
        _M["voice"] = (EmotionModel.from_pretrained(VOICE_REPO, torch_dtype=torch.float32).eval().to(device),
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


def _text(segments, n, device):
    from transformers import pipeline
    if "text" not in _M:
        _M["text"] = pipeline("text-classification", model=TEXT_REPO, top_k=None,
                              device=0 if device == "cuda" else -1)
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
    """Load the models once at startup so the first job does not pay the downloads."""
    import numpy as np
    from PIL import Image
    _clip([Image.new("RGB", (64, 64))], device)
    _voice(np.zeros(32000, dtype=np.float32), 16000, [1.0], device)
    _text([{"start": 0, "end": 1, "text": "warm up"}], 1, device)


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
                     ("text", lambda: _text(segments, n, device))):
        t0 = time.time()
        try:
            out[name] = fn()
            log(f"emotion/{name}: {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            log(f"emotion/{name} skipped: {type(e).__name__}: {e}")
    return out
