FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TRIBE_WHISPER_MODEL=small \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    HF_HUB_VERBOSITY=warning

RUN apt-get update -qq && apt-get install -y -qq git ffmpeg unzip wget curl > /dev/null \
 && rm -rf /var/lib/apt/lists/*

# punkt tokenizer (tribev2 event transforms need it)
RUN mkdir -p /root/nltk_data/tokenizers \
 && cd /root/nltk_data/tokenizers \
 && wget -q https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip \
 && unzip -qo punkt_tab.zip && rm punkt_tab.zip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir -q -r requirements.txt \
 && pip install --no-cache-dir -q "git+https://github.com/facebookresearch/tribev2.git" faster-whisper

COPY tribe_analyze.py hookline_report.py app.py ./

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
