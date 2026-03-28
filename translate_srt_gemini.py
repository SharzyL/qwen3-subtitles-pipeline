#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROMPT = """你是专业的日语字幕本地化编辑。输入是一份日语 SRT 字幕。请在保持每条字幕的编号和时间轴完全不变的前提下，输出一份双语 SRT：每条字幕第一行保留原日文，第二行给出自然、口语化、忠实上下文的简体中文翻译。要求：1) 只输出纯 SRT，不要 Markdown，不要代码块，不要解释；2) 统一上下文中的称谓和语气；3) 可以根据上下文轻微修正明显的 ASR 误识别，但不要胡编；4) 不要增删字幕条目，不要修改时间轴。\n\n以下是需要处理的 SRT：\n\n"""


def main():
    ap = argparse.ArgumentParser(description='Translate monolingual SRT into bilingual JA-ZH SRT with Gemini CLI')
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--model', default=None)
    args = ap.parse_args()

    src = args.input.resolve()
    out = args.output.resolve() if args.output else src.with_name(src.stem + '.ja-zh.srt')
    srt = src.read_text(encoding='utf-8')
    cmd = ['gemini']
    if args.model:
        cmd += ['--model', args.model]
    cmd += [PROMPT + srt]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    out.write_text(result.stdout, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
