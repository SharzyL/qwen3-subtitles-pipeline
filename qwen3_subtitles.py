#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000):
    run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), str(dst)])


def trim_audio(src: Path, dst: Path, start: float, end: float, sample_rate: int = 16000):
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate), str(dst)
    ])


@dataclass
class SpeechChunk:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TokenStamp:
    text: str
    start: float
    end: float


@dataclass
class SubtitleItem:
    start: float
    end: float
    text: str


def detect_compute_chunks(audio_path: Path, threshold: float, min_silence_ms: int,
                          min_speech_ms: int, speech_pad_ms: int,
                          max_chunk_s: float | None = 90.0,
                          hard_max_chunk_s: float | None = None) -> list[SpeechChunk]:
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    wav_np, sr = sf.read(str(audio_path), dtype='float32')
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f'expected 16k audio after extraction, got {sr}')
    wav = torch.from_numpy(np.asarray(wav_np))
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=16000,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    if not ts:
        return []

    chunks: list[SpeechChunk] = []
    cur_start = float(ts[0]['start'])
    cur_end = float(ts[0]['end'])
    for item in ts[1:]:
        seg_start = float(item['start'])
        seg_end = float(item['end'])
        proposed_duration = seg_end - cur_start
        if max_chunk_s is not None and proposed_duration > max_chunk_s:
            chunks.append(SpeechChunk(cur_start, cur_end))
            cur_start, cur_end = seg_start, seg_end
        else:
            cur_end = seg_end
    chunks.append(SpeechChunk(cur_start, cur_end))

    if hard_max_chunk_s is None:
        return chunks

    split_chunks: list[SpeechChunk] = []
    for c in chunks:
        if c.duration <= hard_max_chunk_s:
            split_chunks.append(c)
            continue
        start = c.start
        while start < c.end:
            end = min(start + hard_max_chunk_s, c.end)
            split_chunks.append(SpeechChunk(start, end))
            start = end
    return split_chunks


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_sensevoice_text(text: str) -> str:
    for tag in [
        "<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", "<|yue|>",
        "<|NEUTRAL|>", "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>",
        "<|Speech|>", "<|woitn|>",
    ]:
        text = text.replace(tag, "")
    return normalize_text(text)


def init_qwen_asr(model_id: str, device: str, dtype_name: str):
    import torch
    from qwen_asr import Qwen3ASRModel
    dtype = getattr(torch, dtype_name)
    kwargs = dict(dtype=dtype, max_inference_batch_size=1, max_new_tokens=2048)
    kwargs['device_map'] = 'cpu' if device == 'cpu' else device
    return Qwen3ASRModel.from_pretrained(model_id, **kwargs)


def init_aligner(model_id: str, device: str, dtype_name: str):
    import torch
    from qwen_asr import Qwen3ForcedAligner
    dtype = getattr(torch, dtype_name)
    kwargs = dict(dtype=dtype)
    kwargs['device_map'] = 'cpu' if device == 'cpu' else device
    return Qwen3ForcedAligner.from_pretrained(model_id, **kwargs)


def init_funasr():
    from funasr import AutoModel
    return AutoModel(model='iic/SenseVoiceSmall', device='cpu', disable_update=True)


def transcribe_chunk(asr_engine: str, model, audio_path: Path, language: str | None) -> str:
    if asr_engine == 'qwen3':
        results = model.transcribe(audio=str(audio_path), language=language)
        return normalize_text(results[0].text)
    if asr_engine == 'funasr':
        kwargs = {'input': str(audio_path)}
        lang_map = {
            'Chinese': 'zh', 'English': 'en', 'Japanese': 'ja', 'Korean': 'ko',
            'Cantonese': 'yue', 'auto': 'auto', 'zh': 'zh', 'en': 'en', 'ja': 'ja', 'ko': 'ko', 'yue': 'yue'
        }
        mapped = lang_map.get(language or 'auto', 'auto')
        if mapped != 'auto':
            kwargs['language'] = mapped
        result = model.generate(**kwargs)
        return clean_sensevoice_text(result[0]['text'] if result else '')
    raise ValueError(f'unknown asr engine: {asr_engine}')


def force_align_chunk(aligner, audio_path: Path, text: str, language: str | None) -> list[TokenStamp]:
    aligned = aligner.align(audio=str(audio_path), text=text, language=language)[0]
    out: list[TokenStamp] = []
    for tok in aligned:
        out.append(TokenStamp(
            text=getattr(tok, 'text', ''),
            start=float(getattr(tok, 'start_time')),
            end=float(getattr(tok, 'end_time')),
        ))
    return out


def contains_cjk(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


def split_long_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ''
    for ch in text:
        if len(cur) >= max_chars:
            parts.append(cur)
            cur = ''
        cur += ch
    if cur:
        parts.append(cur)
    return parts


def merge_close(items: list[SubtitleItem], gap_s: float = 0.12, max_chars: int = 24,
                max_duration: float = 6.5) -> list[SubtitleItem]:
    if not items:
        return []
    merged = [items[0]]
    for item in items[1:]:
        prev = merged[-1]
        prev_end_char = prev.text[-1] if prev.text else ''
        if (
            item.start - prev.end <= gap_s
            and len(prev.text + item.text) <= max_chars
            and (item.end - prev.start) <= max_duration
            and prev_end_char not in '。！？!?；;'
        ):
            prev.end = item.end
            prev.text += item.text
        else:
            merged.append(item)
    return merged


def rechunk_tokens(tokens: list[TokenStamp], max_chars: int, max_duration: float,
                   min_duration: float = 0.9, silence_gap_s: float = 0.55) -> list[SubtitleItem]:
    if not tokens:
        return []
    hard_punct = set('。！？!?；;')
    soft_punct = set('，、：,:')
    items: list[SubtitleItem] = []
    buf: list[TokenStamp] = []

    def flush(force: bool = False):
        nonlocal buf
        if not buf:
            return
        raw_text = ''.join(t.text for t in buf).strip()
        if not raw_text:
            buf = []
            return
        start = buf[0].start
        end = max(buf[-1].end, start + min_duration)
        if (not force) and len(raw_text) > max_chars * 1.6:
            subparts = split_long_text(raw_text, max_chars)
            span = max(end - start, min_duration)
            step = span / len(subparts)
            cur_start = start
            for i, part in enumerate(subparts):
                cur_end = end if i == len(subparts) - 1 else cur_start + step
                items.append(SubtitleItem(cur_start, cur_end, part))
                cur_start = cur_end
        else:
            items.append(SubtitleItem(start, end, raw_text))
        buf = []

    for tok in tokens:
        if not tok.text.strip():
            continue
        gap = max(0.0, tok.start - buf[-1].end) if buf else 0.0
        if buf and gap >= silence_gap_s:
            flush()
        buf.append(tok)
        text_now = ''.join(t.text for t in buf).strip()
        duration_now = buf[-1].end - buf[0].start
        last_char = buf[-1].text[-1] if buf[-1].text else ''
        if (
            last_char in hard_punct
            or (last_char in soft_punct and len(text_now) >= max(8, max_chars // 2))
            or duration_now >= max_duration
            or (len(text_now) >= max_chars and contains_cjk(text_now))
        ):
            flush(force=True)
    flush(force=True)
    return merge_close(items, gap_s=0.12, max_chars=max_chars, max_duration=max_duration + 0.8)


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
        for idx, item in enumerate(items, 1):
            f.write(f"{idx}\n{srt_ts(item.start)} --> {srt_ts(item.end)}\n{item.text}\n\n")


def translate_srt_with_gemini(src: Path, out: Path, source_lang: str = 'Japanese', target_lang: str = 'Chinese'):
    prompt = (
        f"你是专业的{source_lang}字幕本地化编辑。输入是一份{source_lang} SRT 字幕。"
        f"请在保持每条字幕的编号和时间轴完全不变的前提下，输出一份双语 SRT："
        f"每条字幕第一行保留原文，第二行给出自然、口语化、忠实上下文的简体中文翻译。"
        f"要求：1) 只输出纯 SRT，不要 Markdown，不要代码块，不要解释；"
        f"2) 统一上下文中的称谓和语气；3) 可以根据上下文轻微修正明显的 ASR 误识别，但不要胡编；"
        f"4) 不要增删字幕条目，不要修改时间轴。\n\n以下是需要处理的 SRT：\n\n"
    )
    srt = src.read_text(encoding='utf-8')
    result = subprocess.run(['gemini', prompt + srt], check=True, text=True, capture_output=True)
    out.write_text(result.stdout, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Integrated subtitle pipeline: VAD -> ASR backend -> optional align -> rechunk -> optional Gemini bilingual translation')
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--workdir', type=Path, default=Path('tmp/qwen3_subtitles'))
    ap.add_argument('--language', default='Chinese')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
    ap.add_argument('--asr-engine', default='qwen3', choices=['qwen3', 'funasr'])
    ap.add_argument('--asr-model', default='Qwen/Qwen3-ASR-0.6B')
    ap.add_argument('--aligner-model', default='Qwen/Qwen3-ForcedAligner-0.6B')
    ap.add_argument('--no-align', action='store_true')
    ap.add_argument('--bilingual', action='store_true')
    ap.add_argument('--translate-with', default='gemini', choices=['gemini'])
    ap.add_argument('--bilingual-output', type=Path, default=None)
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--min-silence-ms', type=int, default=600)
    ap.add_argument('--min-speech-ms', type=int, default=250)
    ap.add_argument('--speech-pad-ms', type=int, default=100)
    ap.add_argument('--compute-max-chunk-s', type=float, default=90.0)
    ap.add_argument('--hard-max-chunk-s', type=float, default=None)
    ap.add_argument('--max-sub-chars', type=int, default=22)
    ap.add_argument('--max-sub-duration', type=float, default=5.5)
    ap.add_argument('--silence-gap-s', type=float, default=0.55)
    ap.add_argument('--keep-temp', action='store_true')
    args = ap.parse_args()

    src = args.input.resolve()
    out = args.output.resolve() if args.output else src.with_suffix('.srt')
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    audio = workdir / f'{src.stem}.16k.wav'
    chunks_dir = workdir / 'chunks'
    chunks_dir.mkdir(exist_ok=True)

    print(f'[1/6] extracting audio: {src} -> {audio}', flush=True)
    extract_audio(src, audio)

    print('[2/6] compute chunks via silero vad...', flush=True)
    chunks = detect_compute_chunks(audio, args.threshold, args.min_silence_ms, args.min_speech_ms,
                                   args.speech_pad_ms, args.compute_max_chunk_s, args.hard_max_chunk_s)
    if not chunks:
        print('No speech detected', file=sys.stderr)
        sys.exit(2)
    print(f'  detected {len(chunks)} compute chunks', flush=True)

    chunk_infos = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_path = chunks_dir / f'chunk_{idx:04d}.wav'
        trim_audio(audio, chunk_path, chunk.start, chunk.end)
        chunk_infos.append((idx, chunk, chunk_path))

    print(f'[3/6] loading ASR engine: {args.asr_engine}...', flush=True)
    asr_model = init_qwen_asr(args.asr_model, args.device, args.dtype) if args.asr_engine == 'qwen3' else init_funasr()
    transcripts: dict[int, str] = {}
    for idx, chunk, chunk_path in chunk_infos:
        print(f'[4/6] asr compute chunk {idx}/{len(chunk_infos)} {chunk.start:.2f}-{chunk.end:.2f}s', flush=True)
        try:
            transcripts[idx] = transcribe_chunk(args.asr_engine, asr_model, chunk_path, args.language)
        except Exception as e:
            print(f'  !! asr chunk {idx} failed: {e}', file=sys.stderr, flush=True)
    del asr_model
    import gc
    gc.collect()

    all_items: list[SubtitleItem] = []
    debug = []

    if args.no_align or args.asr_engine == 'funasr':
        print('[5/6] alignment skipped; using coarse chunk timings + rechunk-by-text', flush=True)
        for idx, chunk, _chunk_path in chunk_infos:
            text = transcripts.get(idx, '').strip()
            if not text:
                continue
            parts = split_long_text(text, args.max_sub_chars)
            span = max(chunk.end - chunk.start, 0.8)
            step = span / max(len(parts), 1)
            cur = chunk.start
            subs = []
            for i, part in enumerate(parts):
                end = chunk.end if i == len(parts) - 1 else cur + step
                sub = SubtitleItem(cur, end, part)
                subs.append(sub)
                all_items.append(sub)
                cur = end
            debug.append({'chunk_index': idx, 'start': chunk.start, 'end': chunk.end, 'transcript': text, 'subtitles': [asdict(s) for s in subs]})
    else:
        print('[5/6] loading qwen forced aligner...', flush=True)
        aligner = init_aligner(args.aligner_model, args.device, args.dtype)
        for idx, chunk, chunk_path in chunk_infos:
            text = transcripts.get(idx, '').strip()
            print(f'[6/6] align compute chunk {idx}/{len(chunk_infos)} {chunk.start:.2f}-{chunk.end:.2f}s', flush=True)
            if not text:
                continue
            try:
                tokens = force_align_chunk(aligner, chunk_path, text, args.language)
            except Exception as e:
                print(f'  !! align chunk {idx} failed: {e}', file=sys.stderr, flush=True)
                continue
            subs = rechunk_tokens(tokens, args.max_sub_chars, args.max_sub_duration, silence_gap_s=args.silence_gap_s)
            for s in subs:
                s.start += chunk.start
                s.end += chunk.start
                all_items.append(s)
            debug.append({'chunk_index': idx, 'start': chunk.start, 'end': chunk.end, 'transcript': text, 'tokens': [asdict(t) for t in tokens], 'subtitles': [asdict(s) for s in subs]})

    all_items.sort(key=lambda x: x.start)
    print(f'writing srt -> {out}', flush=True)
    write_srt(all_items, out)
    debug_path = out.with_suffix('.debug.json')
    debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'done: {out}')
    print(f'debug: {debug_path}')

    if args.bilingual:
        bilingual_out = args.bilingual_output.resolve() if args.bilingual_output else out.with_name(out.stem + '.ja-zh.srt')
        print(f'translating with {args.translate_with} -> {bilingual_out}', flush=True)
        translate_srt_with_gemini(out, bilingual_out, source_lang=args.language)
        print(f'bilingual done: {bilingual_out}', flush=True)

    if not args.keep_temp:
        shutil.rmtree(chunks_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
