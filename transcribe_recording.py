#!/usr/bin/env python3
"""
Transcribe a FLAC (or any ffmpeg-readable) recording.

Backends:
  gemini  - Gemini 3.5 Transcribe via the Interactions API (needs GEMINI_API_KEY)
  whisper - local faster-whisper (no network; pip install faster-whisper)

Setup: ./setup_env.sh && source .venv/bin/activate   (needs ffmpeg)

Examples:
  transcribe.py talk.flac --lang pl-PL
  transcribe.py meeting.flac --diarize --timestamps
  transcribe.py lecture.flac --smart --vocab terms.txt
  transcribe.py interview.flac --backend whisper --lang pl --timestamps
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GEMINI_MODEL = "gemini-3.5-transcribe"
LIMIT_PLAIN = 55 * 60       # documented limit: 60 min per request
LIMIT_ANNOTATED = 28 * 60   # documented limit: 30 min with diarization / word timestamps
SILENCE_SEARCH = 120        # seconds to look back from a boundary for a silence
RETRYABLE = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------- utilities

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def hms(t):
    t = int(t or 0)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def duration_s(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)]).stdout
    return float(out.strip())


def normalise(src, dst):
    """Mono, 16 kHz, lossless FLAC."""
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dst)])


def silence_midpoints(path, noise_db=-35, min_dur=0.4):
    """Midpoints (s) of silent intervals; silencedetect reports on stderr."""
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
               "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
               "-f", "null", "-"]).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", err)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def plan_chunks(total, limit, silences):
    """[(start, end), ...], cutting at the latest silence before each limit."""
    cuts = [0.0]
    while total - cuts[-1] > limit:
        target = cuts[-1] + limit
        near = [s for s in silences if target - SILENCE_SEARCH <= s <= target]
        cuts.append(max(near) if near else target)   # hard cut as fallback
    cuts.append(total)
    return list(zip(cuts[:-1], cuts[1:]))


def extract(src, start, end, dst):
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
         "-c:a", "flac", str(dst)])


def turns(words):
    """Group consecutive words by speaker label."""
    out, cur = [], None
    for w in words:
        if cur is None or w["speaker"] != cur["speaker"]:
            cur = {"speaker": w["speaker"], "start": w["start"], "words": []}
            out.append(cur)
        cur["words"].append(w["text"])
    # without --timestamps there are no offsets, so omit the time prefix
    return "\n".join(("" if t["start"] is None else f"[{hms(t['start'])}] ")
                     + f"{t['speaker'] or '-'}: {' '.join(t['words'])}"
                     for t in out)


# ---------------------------------------------------------------- gemini

def transcription_config(args, vocab):
    cfg = {}
    if args.lang:
        cfg["language_codes"] = [args.lang]
    if vocab:
        cfg["custom_vocabulary"] = vocab
    if args.smart:
        cfg["mode"] = "smart"
    elif args.diarize or args.timestamps:
        mode = {"type": "verbatim"}
        if args.diarize:
            mode["diarization_mode"] = "speaker"
        if args.timestamps:
            mode["timestamp_granularities"] = ["word"]
        cfg["mode"] = mode
    return cfg


def offset_s(x):
    return float(str(x).rstrip("s")) if x not in (None, "") else None


def transcript_text(interaction):
    """One line per text block (≈ utterance); SDK's output_text glues them with ''."""
    blocks = [c.text.strip()
              for step in getattr(interaction, "steps", None) or []
              if getattr(step, "type", None) == "model_output"
              for c in getattr(step, "content", None) or []
              if getattr(c, "type", None) == "text" and c.text]
    return "\n".join(b for b in blocks if b)


def word_annotations(interaction):
    for step in getattr(interaction, "steps", None) or []:
        for content in getattr(step, "content", None) or []:
            for a in getattr(content, "annotations", None) or []:
                if getattr(a, "type", None) == "word_info":
                    yield a


def gemini_request(client, path, tcfg, retries=5):
    f = client.files.upload(file=str(path), config={"mime_type": "audio/flac"})
    try:
        while getattr(getattr(f, "state", None), "name", "") == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        for attempt in range(retries):
            try:
                it = client.interactions.create(
                    model=GEMINI_MODEL,
                    input=[{"type": "audio", "uri": f.uri, "mime_type": f.mime_type}],
                    generation_config={"transcription_config": tcfg},
                )
            # interactions raise their own error classes (with .status_code),
            # not google.genai.errors.APIError (with .code)
            except Exception as e:
                code = getattr(e, "status_code", None)
                if code not in RETRYABLE or attempt == retries - 1:
                    raise
                wait = 10 * 2 ** attempt
                print(f"  API error {code}, retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if it.status != "completed":
                print(f"  warning: interaction status {it.status!r}, "
                      "transcript may be truncated", file=sys.stderr)
            return it
    finally:
        client.files.delete(name=f.name)


def transcribe_gemini(src, args, vocab, workdir):
    from google import genai
    client = genai.Client()                       # reads GEMINI_API_KEY
    tcfg = transcription_config(args, vocab)
    limit = LIMIT_ANNOTATED if (args.diarize or args.timestamps) else LIMIT_PLAIN

    total = duration_s(src)
    silences = silence_midpoints(src) if total > limit else []
    chunks = plan_chunks(total, limit, silences)

    texts, words = [], []
    for i, (start, end) in enumerate(chunks):
        print(f"chunk {i + 1}/{len(chunks)}: {hms(start)}-{hms(end)}", file=sys.stderr)
        part = src
        if len(chunks) > 1:
            part = workdir / f"chunk_{i:03d}.flac"
            extract(src, start, end, part)
        it = gemini_request(client, part, tcfg)
        texts.append(transcript_text(it))
        for a in word_annotations(it):
            s = offset_s(getattr(a, "start_offset", None))
            e = offset_s(getattr(a, "end_offset", None))
            spk = getattr(a, "speaker", None)
            words.append({
                "text": a.text,
                # labels are only consistent within a single request
                "speaker": f"c{i}:{spk}" if (spk and len(chunks) > 1) else spk,
                "start": None if s is None else round(s + start, 3),
                "end": None if e is None else round(e + start, 3),
            })
    return "\n\n".join(texts), words


# ---------------------------------------------------------------- whisper

def transcribe_whisper(src, args, vocab):
    from faster_whisper import WhisperModel
    model = WhisperModel(args.whisper_model, device="auto", compute_type="default")
    segments, info = model.transcribe(
        str(src),
        language=args.lang.split("-")[0] if args.lang else None,  # ISO 639-1
        vad_filter=True,                   # skip non-speech regions
        condition_on_previous_text=False,  # limits repetition loops on long audio
        word_timestamps=args.timestamps,
        initial_prompt=", ".join(vocab) if vocab else None,
        beam_size=5,
    )
    print(f"language: {info.language} (p={info.language_probability:.2f})", file=sys.stderr)
    lines, words = [], []
    for seg in segments:                   # decoding happens lazily here
        text = seg.text.strip()
        lines.append(f"[{hms(seg.start)}] {text}" if args.timestamps else text)
        for w in seg.words or []:
            words.append({"text": w.word.strip(), "speaker": None,
                          "start": round(w.start, 3), "end": round(w.end, 3)})
    return "\n".join(lines), words


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", help="output stem (default: input without extension)")
    p.add_argument("--backend", choices=["gemini", "whisper"], default="gemini")
    p.add_argument("--lang", help="e.g. pl-PL; omit for auto-detection")
    p.add_argument("--diarize", action="store_true", help="speaker labels (gemini)")
    p.add_argument("--timestamps", action="store_true", help="word-level timestamps")
    p.add_argument("--smart", action="store_true", help="gemini smart mode (clean, formatted)")
    p.add_argument("--vocab", type=Path, help="file with one term per line")
    p.add_argument("--whisper-model", default="large-v3")
    args = p.parse_args()

    annotated = args.diarize or args.timestamps
    if args.smart and annotated:
        p.error("--smart cannot be combined with --diarize/--timestamps")
    if args.backend == "gemini" and args.vocab and annotated:
        p.error("--vocab cannot be combined with --diarize/--timestamps on gemini")
    if args.backend == "whisper" and (args.diarize or args.smart):
        p.error("--diarize/--smart are gemini-only (for local diarization see WhisperX)")

    vocab = []
    if args.vocab:
        vocab = [t.strip() for t in args.vocab.read_text(encoding="utf-8").splitlines() if t.strip()]
    stem = args.output or str(args.input.with_suffix(""))

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        norm = workdir / "normalised.flac"
        normalise(args.input, norm)
        if args.backend == "gemini":
            text, words = transcribe_gemini(norm, args, vocab, workdir)
        else:
            text, words = transcribe_whisper(norm, args, vocab)

    Path(f"{stem}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    if words:
        Path(f"{stem}.words.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.diarize:
            Path(f"{stem}.turns.txt").write_text(turns(words) + "\n", encoding="utf-8")
    print(f"done: {stem}.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
