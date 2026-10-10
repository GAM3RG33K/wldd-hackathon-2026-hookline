FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TRIBE_WHISPER_MODEL=small \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    HF_HUB_VERBOSITY=warning

# Layer 1: OS deps (stable — cached across builds)
RUN apt-get update -qq && apt-get install -y -qq git ffmpeg unzip wget curl > /dev/null \
 && rm -rf /var/lib/apt/lists/*

# Layer 2: punkt tokenizer (stable — cached)
RUN mkdir -p /root/nltk_data/tokenizers \
 && cd /root/nltk_data/tokenizers \
 && wget -q https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip \
 && unzip -qo punkt_tab.zip && rm punkt_tab.zip

WORKDIR /app
COPY requirements.txt .

# Layer 3: torch trio pinned (stable — cached; touch only to change CUDA/torch)
RUN pip install --no-cache-dir -q torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Layer 4: app deps + tribev2 at pinned SHA; transformers<5 because 5.x needs torch>=2.6
# (AutoModel import fails on torch 2.5.1) (re-runs only when requirements.txt
# or the SHA below changes; pip keeps layer-3 torch since it already satisfies)
RUN pip install --no-cache-dir -q -r requirements.txt \
 && pip install --no-cache-dir -q "git+https://github.com/facebookresearch/tribev2.git@af58661791a351a448a489042a28f6c37e1c14b7" faster-whisper \
    "transformers>=4.46,<5" \
 && python -c "import torch, torchaudio, torchvision; import transformers; from transformers import AutoModel; print('torch', torch.__version__, '| ta', torchaudio.__version__, '| tv', torchvision.__version__, '| transformers', transformers.__version__)"

# Runtime tuning, kept below the pip layers so changing it doesn't reinstall torch.
# Whisper base: ~3x faster than small, multilingual (reels may be Hindi/Hinglish).
# 10s chunks + expandable segments: fixes the T4 OOM on a 28s reel.
ENV TRIBE_WHISPER_MODEL=base \
    HOOKLINE_CHUNK_S=10 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Some hardware tiers (seen on a100-large) run the container as a non-root user with
# HOME=/, so every cache write hit "/.cache: Permission denied". Point HOME and the
# caches at a writable path, and let any user read the punkt data under /root and
# patch tribev2 at startup (faster-whisper transcription patch); tribev2 caches features
# in ./cache, i.e. /app/cache. neuralset pip-installs the spaCy model on first use, which
# fails for a non-root user, so it is installed here.
ENV HOME=/tmp/home \
    HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    TORCH_HOME=/tmp/torch \
    NLTK_DATA=/root/nltk_data
RUN mkdir -p /tmp/home /tmp/hf /tmp/cache /tmp/torch && chmod -R a+rwX /tmp \
 && chmod a+rx /root && chmod -R a+rX /root/nltk_data \
 && chmod -R a+rwX "$(python -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("tribev2").origin))')" \
 && mkdir -p /app/cache && chmod -R a+rwX /app \
 && python -m spacy download en_core_web_lg -q

COPY tribe_analyze.py hookline_report.py app.py ./

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
