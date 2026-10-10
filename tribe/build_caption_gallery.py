#!/usr/bin/env python
"""Build hf_space/caption_gallery.npy: CLIP image embeddings of 8 frames from ~55 sample reels.

The Space scores a creator caption against these reels as well as against the reel itself: a true
caption fits its own reel better than other reels; a generic one ("follow for more #viral") fits all
reels about equally. Embeddings only, no video. Rebuild after the sample set changes:
  uv run --python 3.12 --with "transformers<5" --with torch --with pillow --with "numpy<2" python tribe/build_caption_gallery.py
"""
import glob
import io
import subprocess
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model = CLIPModel.from_pretrained(REPO, use_safetensors=True).eval(); proc = CLIPProcessor.from_pretrained(REPO)
ROOT = Path(__file__).resolve().parent.parent
def frames(video, k=8):
    d = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",video],capture_output=True,text=True).stdout or 0)
    out = []
    for t in np.linspace(0.5, max(0.6, d - 0.5), k):
        b = subprocess.run(["ffmpeg","-loglevel","error","-ss",f"{t:.2f}","-i",video,"-frames:v","1","-vf","scale=320:-2","-f","image2pipe","-vcodec","png","-"],capture_output=True).stdout
        if b:
            out.append(Image.open(io.BytesIO(b)).convert("RGB"))
    return out
def img_emb(ims):
    with torch.inference_mode():
        e = model.get_image_features(**proc(images=ims, return_tensors="pt"))
    return (e / e.norm(dim=-1, keepdim=True)).numpy()
vids = sorted(glob.glob(str(ROOT / "samples/*/videos/*.mp4")))
vids = [v for i, v in enumerate(vids) if i % 2 == 0][:56]
G = []
for v in vids:
    ims = frames(v)
    if len(ims) >= 4: G.append(np.pad(img_emb(ims), ((0, 8 - len(ims)), (0, 0)), constant_values=np.nan))
G = np.stack(G).astype(np.float16)
np.save(ROOT / "hf_space" / "caption_gallery.npy", G)
print("gallery", G.shape, "from", len(vids), "reels")
