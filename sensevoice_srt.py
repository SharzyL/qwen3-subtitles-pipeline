#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000):
    run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), str(dst)])


def ffprobe_duration(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], capture=True).stdout.strip()
    return float(out)


def clean_text(text: str) -> str:
    for tag in [
        "<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", "<|yue|>",
        "<|NEUTRAL|>", "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>",
        "<|Speech|>", "<|woitn|>",
    ]:
        text = text.replace(tag, "")
    return " ".join(text.split()).strip()


@dataclass
class SubtitleItem:
    start: float
    end: float
    text: str


def srt_ts(sec: float) -> str:
    sec = max(sec, 0.0)
    ms = round(sec * 1000)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(items: list[SubtitleItem], path: Path):
    with path.open('w', encoding='utf-8') as f:
        for i, item in enumerate(items, 1):
            f.write(f"{i}\n{srt_ts(item.start)} --> {srt_ts(item.end)}\n{item.text}\n\n")


def main():
    ap = argparse.ArgumentParser(description='Generate coarse SRT with FunASR/SenseVoice by chunked transcription')
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--workdir', type=Path, default=Path('tmp/sensevoice_srt'))
    ap.add_argument('-l', '--language', default='ja', choices=['auto', 'zh', 'en', 'ja', 'ko', 'yue'])
    ap.add_argument('--chunk-size', type=int, default=20)
    args = ap.parse_args()

    from funasr import AutoModel

    src = args.input.resolve()
    out = args.output.resolve() if args.output else src.with_suffix('.sensevoice.srt')
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    audio = workdir / f'{src.stem}.16k.wav'

    print(f'[1/4] extracting audio: {src} -> {audio}', flush=True)
    extract_audio(src, audio)
    duration = ffprobe_duration(audio)
    num_chunks = math.ceil(duration / args.chunk_size)

    print('[2/4] loading SenseVoice model...', flush=True)
    model = AutoModel(model='iic/SenseVoiceSmall', device='cpu', disable_update=True)

    print(f'[3/4] transcribing {num_chunks} chunks...', flush=True)
    items: list[SubtitleItem] = []
    for i in range(num_chunks):
        start = i * args.chunk_size
        end = min((i + 1) * args.chunk_size, duration)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            run([
                'ffmpeg', '-y', '-ss', str(start), '-to', str(end), '-i', str(audio),
                '-ac', '1', '-ar', '16000', str(tmp_path)
            ])
            kwargs = {'input': str(tmp_path)}
            if args.language != 'auto':
                kwargs['language'] = args.language
            result = model.generate(**kwargs)
            text = clean_text(result[0]['text'] if result else '')
            if text:
                items.append(SubtitleItem(start, end, text))
            print(f'  chunk {i+1}/{num_chunks}: {start:.1f}-{end:.1f}s', flush=True)
        finally:
            tmp_path.unlink(missing_ok=True)

    print(f'[4/4] writing srt -> {out}', flush=True)
    write_srt(items, out)
    print(f'done: {out}')


if __name__ == '__main__':
    main()
