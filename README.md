# qwen3-subtitles-pipeline

A unified subtitle pipeline with one main entry script:

- `qwen3_subtitles.py`

The pipeline keeps the same overall structure:

1. Silero VAD for compute-time chunking
2. selectable ASR backend
3. optional alignment
4. subtitle rechunking
5. optional Gemini bilingual translation

## Supported ASR backends

### 1. `qwen3`
Good for alignment experiments:

- Qwen3-ASR transcription
- Qwen3-ForcedAligner timing
- subtitle re-chunking after alignment

### 2. `funasr`
Good for faster long-media transcription:

- FunASR / SenseVoiceSmall transcription
- keeps the same VAD + rechunk main flow
- can skip alignment and still output SRT quickly

## Requirements

- Python 3.12+
- `ffmpeg`
- `uv`
- `gemini` CLI (optional, for bilingual translation)

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install qwen-asr silero-vad soundfile funasr modelscope
```

If you only want the Qwen3 backend, `funasr` and `modelscope` are optional.

## Usage

## Qwen3 backend

```bash
python qwen3_subtitles.py input.mp4 \
  --language Japanese \
  --asr-engine qwen3 \
  --output input.ja.srt
```

## FunASR / SenseVoice backend

```bash
python qwen3_subtitles.py input.mp4 \
  --language Japanese \
  --asr-engine funasr \
  --no-align \
  --output input.ja.srt
```

## Generate bilingual JA-ZH SRT in the same command

```bash
python qwen3_subtitles.py input.mp4 \
  --language Japanese \
  --asr-engine funasr \
  --no-align \
  --bilingual \
  --output input.ja.srt
```

This will also produce a bilingual file like:

- `input.ja-zh.srt`

## Recommended practical workflow

For long Japanese videos on CPU:

```bash
python qwen3_subtitles.py input.mp4 \
  --language Japanese \
  --asr-engine funasr \
  --no-align \
  --bilingual
```

For alignment experiments:

```bash
python qwen3_subtitles.py input.mp4 \
  --language Japanese \
  --asr-engine qwen3
```

## Notes

- The main pipeline stays unified; only the ASR engine changes.
- VAD and subtitle rechunking remain in the same script.
- On CPU, Qwen3-ASR is significantly slower than FunASR / SenseVoice for long videos.
- Gemini translation works best when it sees the whole SRT at once instead of translating subtitle lines one by one.
