# qwen3-subtitles-pipeline

A subtitle pipeline workspace with two practical paths:

- **Qwen3 path**: Silero VAD → Qwen3-ASR → Qwen3-ForcedAligner → subtitle re-chunking
- **FunASR / SenseVoice path**: chunked transcription for long media, then Gemini CLI for bilingual SRT generation

## What this repo supports

### 1. Qwen3 alignment pipeline
Useful when you want to experiment with:

- compute-time chunking via VAD
- Qwen3-ASR transcription
- Qwen3-ForcedAligner timing
- subtitle re-chunking after alignment

Main script:

- `qwen3_subtitles.py`

### 2. FunASR / SenseVoice transcription pipeline
Useful when you want faster batch transcription for long videos, especially when Qwen3-ASR is too slow.

Scripts:

- `sensevoice_srt.py` — generate coarse monolingual SRT from long media with SenseVoice
- `translate_srt_gemini.py` — translate a monolingual Japanese SRT into bilingual JA-ZH SRT using Gemini CLI in one pass

This mirrors the practical workflow used in the local `asr-funasr` skill.

## Requirements

- Python 3.12+
- `ffmpeg`
- `uv`
- `gemini` CLI (for bilingual translation)

## Setup

### Qwen3 path

```bash
uv venv
source .venv/bin/activate
uv pip install qwen-asr silero-vad soundfile
```

### FunASR / SenseVoice path

```bash
uv venv
source .venv/bin/activate
uv pip install funasr modelscope soundfile
```

If you want both paths in one environment:

```bash
uv venv
source .venv/bin/activate
uv pip install qwen-asr silero-vad soundfile funasr modelscope
```

## Usage

## Qwen3 path

```bash
source .venv/bin/activate
python qwen3_subtitles.py input.mp4 \
  --output output.srt \
  --device cpu \
  --dtype float32 \
  --language Japanese
```

## SenseVoice path

Generate monolingual SRT first:

```bash
source .venv/bin/activate
python sensevoice_srt.py input.mp4 \
  --output input.ja.srt \
  --language ja \
  --chunk-size 20
```

Then translate the whole SRT into bilingual Japanese + Chinese SRT:

```bash
python translate_srt_gemini.py input.ja.srt \
  --output input.ja-zh.srt
```

## Recommended practical workflow

For long Japanese videos:

1. Use `sensevoice_srt.py` to get a fast first-pass Japanese SRT
2. Use `translate_srt_gemini.py` to generate a bilingual JA-ZH SRT in one pass
3. If timing quality is not good enough, optionally revisit alignment with a dedicated aligner

## Notes

- Qwen3 path is better for alignment experiments; it is slower on CPU for long media.
- SenseVoice path is better for batch transcription throughput.
- Gemini translation works best when it sees the whole SRT at once instead of translating subtitle lines one by one.
