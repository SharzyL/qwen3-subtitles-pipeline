# qwen3-subtitles-pipeline

A subtitle pipeline using:

- Silero VAD for compute-time chunking
- Qwen3-ASR for transcription
- Qwen3-ForcedAligner for alignment
- subtitle re-chunking for SRT generation

## Features

- Uses VAD only for compute chunking, not final subtitle boundaries
- Preserves ASR transcript text before alignment
- Re-chunks subtitles after alignment using punctuation, silence gaps, max duration, and max line width
- Outputs `.srt` and debug `.json`

## Requirements

- Python 3.12+
- `ffmpeg`
- `uv`

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install qwen-asr silero-vad soundfile
```

## Usage

```bash
source .venv/bin/activate
python qwen3_subtitles.py input.mp4 \
  --output output.srt \
  --device cpu \
  --dtype float32 \
  --language Chinese
```

## Notes

- VAD is used only to keep ASR/alignment chunks manageable.
- Final subtitle segmentation is performed after alignment.
- On CPU, the script loads ASR first, frees it, then loads the aligner to reduce memory pressure.
