# fast in-process transcription (faster-whisper) instead of a whisperx subprocess per video
!pip -q install faster-whisper
import pathlib
p = pathlib.Path("/content/tribev2/tribev2/eventstransforms.py")
s = p.read_text()
if "_FW_MODEL" not in s:
    marker = '        device = "cuda" if torch.cuda.is_available() else "cpu"\n        compute_type'
    fast = """        try:
            global _FW_MODEL
            from faster_whisper import WhisperModel
            if "_FW_MODEL" not in globals():
                _FW_MODEL = WhisperModel(os.environ.get("TRIBE_WHISPER_MODEL", "small"),
                                         device="cuda" if torch.cuda.is_available() else "cpu",
                                         compute_type="float16" if torch.cuda.is_available() else "int8")
            import subprocess as _sp, numpy as _np
            _raw = _sp.run(['ffmpeg','-loglevel','error','-i',str(wav_filename),'-f','f32le','-ac','1','-ar','16000','-'], capture_output=True, check=True).stdout
            _audio = _np.frombuffer(_raw, dtype=_np.float32)
            segs, _ = _FW_MODEL.transcribe(_audio, language=language_codes[language], word_timestamps=True)
            words = []
            for i, seg in enumerate(segs):
                for w in (seg.words or []):
                    words.append({"text": w.word.replace('"', "").strip(), "start": w.start,
                                  "duration": w.end - w.start, "sequence_id": i,
                                  "sentence": seg.text.replace('"', "")})
            return pd.DataFrame(words)
        except ImportError:
            pass
"""
    assert marker in s
    s = s.replace(marker, fast + marker, 1)
    if "\nimport os" not in s: s = "import os\n" + s
    p.write_text(s)
print("fast transcript patch applied")