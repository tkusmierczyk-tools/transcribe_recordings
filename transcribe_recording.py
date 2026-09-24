#!/usr/bin/env python3
"""
Transcribe a meeting recording (FLAC or any ffmpeg-readable audio).

By default the transcript reads like a conversation: one line per speaker turn,
e.g. "[00:01:23] [Speaker1]: ...". Long recordings are sent in overlapping parts
and speakers are matched on the overlap, so Speaker1 stays the same person
throughout. A speaker who may be someone heard earlier is shown with the
candidates, e.g. "[Speaker5 (Speaker1?, Speaker3?)]".
Parts that could not be transcribed are marked NOT TRANSCRIBED in the output
and the script exits with status 1.

Backends:
  gemini  - Gemini 3.5 Transcribe via the Interactions API (needs GEMINI_API_KEY)
  whisper - local faster-whisper (no network, no speaker labels)

Setup: ./setup_env.sh && source .venv/bin/activate   (needs ffmpeg)

Examples:
  transcribe_recording.py meeting.flac                  # speakers + timestamps
  transcribe_recording.py meeting.flac --lang pl-PL
  transcribe_recording.py lecture.flac --smart --vocab terms.txt   # clean text, no speakers
  transcribe_recording.py interview.flac --backend whisper --lang pl
"""
import argparse
import collections
import difflib
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
MISSING = "NOT TRANSCRIBED"  # marks gaps in the output
MIN_OVERLAP = 120           # seconds each part repeats from the previous one, to match speakers;
MAX_OVERLAP = 600           # extended up to this so each recent speaker is heard again
WORDS_TO_HEAR = 10          # words of each recent speaker the overlap should contain
MIN_SHARED_WORDS = 3        # words a speaker must say in the overlap to be matched


# ---------------------------------------------------------------- utilities

T0 = time.monotonic()


def log(msg):
    """Progress message on stderr, prefixed with the time elapsed since start."""
    t = int(time.monotonic() - T0)
    print(f"[{t // 60:3d}m{t % 60:02d}s] {msg}", file=sys.stderr, flush=True)


def mb(path):
    return f"{Path(path).stat().st_size / 1e6:.1f} MB"


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def hms(t):
    t = int(t or 0)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def duration_s(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)]).stdout
    return float(out.strip())


def normalise(src, dst, loudness=False):
    """Mono, 16 kHz, lossless FLAC; optionally with evened-out loudness."""
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         *(["-af", "loudnorm"] if loudness else []),
         "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dst)])


def silence_midpoints(path, noise_db=-35, min_dur=0.4):
    """Midpoints (s) of silent intervals; silencedetect reports on stderr."""
    err = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
               "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
               "-f", "null", "-"]).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", err)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def next_cut(total, target, silences):
    """End of a part that may run until `target`: the latest silence before it."""
    if target >= total:
        return total
    near = [s for s in silences if target - SILENCE_SEARCH <= s <= target]
    return max(near) if near else target          # hard cut as fallback


def overlap_start(prev, cut):
    """Where the part after `cut` should start: early enough that everyone who spoke
    in the last MAX_OVERLAP seconds says WORDS_TO_HEAR words again, and at least
    MIN_OVERLAP back."""
    heard = collections.defaultdict(list)
    for w in prev:
        if w["speaker"] and w["start"] is not None and cut - MAX_OVERLAP <= w["start"] < cut:
            heard[w["speaker"]].append(w["start"])
    need = [sorted(t)[-WORDS_TO_HEAR:][0] for t in heard.values()]
    return max(0.0, min([cut - MIN_OVERLAP, *need]))


def extract(src, start, end, dst):
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
         "-c:a", "flac", str(dst)])


def turns(words, doubts=None):
    """Group consecutive words by speaker label.

    doubts: label -> earlier speakers it may be, shown as "[Speaker5 (Speaker2?)]".
    """
    out, cur = [], None
    for w in words:
        if cur is None or w["speaker"] != cur["speaker"]:
            cur = {"speaker": w["speaker"], "start": w["start"], "words": []}
            out.append(cur)
        cur["words"].append(w["text"])

    def name(s):
        maybe = (doubts or {}).get(s)
        return f"[{s} ({', '.join(m + '?' for m in maybe)})]" if maybe else f"[{s or '-'}]"
    # without --timestamps there are no offsets, so omit the time prefix
    return "\n".join(("" if t["start"] is None else f"[{hms(t['start'])}] ")
                     + f"{name(t['speaker'])}: {' '.join(t['words'])}"
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
    """Plain transcript text, one segment per line."""
    blocks = [c.text.strip()
              for step in getattr(interaction, "steps", None) or []
              if getattr(step, "type", None) == "model_output"
              for c in getattr(step, "content", None) or []
              if getattr(c, "type", None) == "text" and c.text]
    text = "\n".join(b for b in blocks if b)
    # segments come back glued together ("had.So, you"): break the line there
    return re.sub(r"(\w[.?!])(\w)", lambda m: m[1] + "\n" + m[2]
                  if not m[1][0].isupper() and m[2].isupper() else m[0], text)


def link_speakers(prev, new, audio_start, cut, issued):
    """Map this part's speaker labels to recording-wide ones ("Speaker1", ...).

    prev: the previous part's words; new: this part's words, which start at
    audio_start, before `cut`. Both parts transcribed that overlap, so each word
    they agree on is a vote for "this label is that earlier speaker". Unmatched
    labels get a new number. issued (label -> earlier speakers it may be, empty
    if certain) is updated here. Returns the mapping.
    """
    key = lambda w: re.sub(r"\W", "", w["text"].lower())
    a = [w for w in prev if w["start"] is not None and w["start"] >= audio_start]
    b = [w for w in new if w["start"] is not None and w["start"] < cut]
    votes = collections.Counter()
    sm = difflib.SequenceMatcher(None, [key(w) for w in a], [key(w) for w in b], autojunk=False)
    for m in sm.get_matching_blocks():
        for x, y in zip(a[m.a:m.a + m.size], b[m.b:m.b + m.size]):
            if x["speaker"] and y["speaker"] and abs(x["start"] - y["start"]) < 2:
                votes[y["speaker"], x["speaker"]] += 1
    totals = collections.Counter()
    for (label, _), n in votes.items():
        totals[label] += n
    mapping, earlier = {}, list(issued)
    # labels that speak in this part's own time, after the overlap
    speaking = {w["speaker"] for w in new if w["start"] is None or w["start"] >= cut}
    for (label, known), n in votes.most_common():       # strongest agreement first
        if (n >= MIN_SHARED_WORDS and n * 2 > totals[label]
                and label not in mapping and known not in mapping.values()):
            mapping[label] = known
    for w in new:                                        # the rest, in order of appearance
        if w["speaker"] and w["speaker"] not in mapping:
            # a new person, unless an earlier speaker is still unaccounted for; a label
            # heard only in the overlap doesn't count (Gemini may relabel them later)
            accounted = {g for raw, g in mapping.items() if raw in speaking}
            name = f"Speaker{len(issued) + 1}"
            issued[name] = [s for s in earlier if s not in accounted]
            mapping[w["speaker"]] = name
    if mapping:
        log("  speakers: " + ", ".join(
            f"{raw}->{g}" + (f" ({votes[raw, g]} shared words"
                             + ("" if raw in speaking else ", only in the overlap") + ")"
                             if votes[raw, g] else "")
            for raw, g in mapping.items()))
        for g in mapping.values():
            if g not in earlier and issued[g]:
                log(f"  {g} may be {' or '.join(issued[g])} "
                    "(not heard in the overlap with the previous part)")
    return mapping


def word_annotations(interaction):
    for step in getattr(interaction, "steps", None) or []:
        for content in getattr(step, "content", None) or []:
            for a in getattr(content, "annotations", None) or []:
                if getattr(a, "type", None) == "word_info":
                    yield a


def gemini_request(client, path, tcfg, retries=5):
    log(f"  uploading {mb(path)} to Gemini")
    f = client.files.upload(file=str(path), config={"mime_type": "audio/flac"})
    try:
        if getattr(getattr(f, "state", None), "name", "") == "PROCESSING":
            log("  waiting for Gemini to process the upload")
        while getattr(getattr(f, "state", None), "name", "") == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        for attempt in range(retries):
            log("  requesting the transcription (this can take a few minutes)")
            t = time.monotonic()
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
                log(f"  API error {code}, retry in {wait}s")
                time.sleep(wait)
                continue
            log(f"  transcription received after {time.monotonic() - t:.0f}s, status {it.status!r}")
            return it
    finally:
        log("  deleting the uploaded audio from Gemini")
        client.files.delete(name=f.name)


def transcribe_gemini(src, args, vocab, workdir):
    from google import genai
    client = genai.Client()                       # reads GEMINI_API_KEY
    tcfg = transcription_config(args, vocab)
    limit = LIMIT_ANNOTATED if (args.diarize or args.timestamps) else LIMIT_PLAIN

    # speakers are matched across parts on repeated audio, which needs word times
    match = args.diarize and args.timestamps
    log(f"model {GEMINI_MODEL}, request settings: {json.dumps(tcfg)}")

    total = duration_s(src)
    silences = []
    if total > limit:
        log(f"the recording is longer than the {limit // 60} min per request: splitting it "
            "into parts at pauses" + (", each repeating the end of the previous one "
                                      "to match speakers" if match else ""))
        silences = silence_midpoints(src)
        log(f"  found {len(silences)} pauses to cut at")
    else:
        log("the recording fits in one request")

    texts, words, failures = [], [], []
    prev, issued = [], {}                # previous part's words; speakers so far -> doubts
    start, parts = 0.0, 0
    while start < total:                 # each part is planned once the previous one is done
        # later parts start early, repeating the previous part's end to match speakers
        audio_start = overlap_start(prev, start) if (match and start) else start
        end = next_cut(total, audio_start + limit, silences)
        parts += 1
        log(f"part {parts}: {hms(start)}-{hms(end)}" + (
            f" (from {hms(audio_start)} to match speakers)" if audio_start < start else ""))
        part = src
        if (audio_start, end) != (0.0, total):
            log(f"  cutting out the audio {hms(audio_start)}-{hms(end)}")
            part = workdir / f"part_{parts:03d}.flac"
            extract(src, audio_start, end, part)
        try:
            it = gemini_request(client, part, tcfg)
        except Exception as e:           # keep the other parts, mark this one
            log(f"  failed: {e}")
            failures.append(e)
            texts.append(f"[{hms(start)}-{hms(end)} {MISSING}: {e}]")
            prev, start = [], end
            continue
        cw = []
        for a in word_annotations(it):
            s = offset_s(getattr(a, "start_offset", None))
            e = offset_s(getattr(a, "end_offset", None))
            cw.append({
                "text": a.text,
                "speaker": getattr(a, "speaker", None),
                "start": None if s is None else round(s + audio_start, 3),
                "end": None if e is None else round(e + audio_start, 3),
            })
        plain = transcript_text(it)
        log(f"  got {len(plain.split())} words of text, {len(cw)} with per-word details"
            + (f", {len({w['speaker'] for w in cw} - {None})} speakers" if args.diarize else ""))
        # raw labels only hold within one request: map them to recording-wide ones
        mapping = link_speakers(prev, cw, audio_start, start, issued)
        for w in cw:
            w["speaker"] = mapping.get(w["speaker"])
            if issued.get(w["speaker"]):
                w["maybe"] = issued[w["speaker"]]       # earlier speakers this may be
        # drop the repeated overlap: the leading words up to the first run clearly past
        # the cut (by position, so a later word with a garbled timestamp is kept)
        past = lambda w: w["start"] is None or w["start"] >= start
        j = next((k for k in range(len(cw)) if all(map(past, cw[k:k + 3]))), len(cw))
        own = cw[j:]
        if j:
            log(f"  dropping the first {j} words: the previous part already has them")
        words += own
        prev = own
        # speaker turns read like a conversation; use them only if they cover the text
        if any(w["speaker"] for w in own) and len(cw) >= 0.9 * len(plain.split()):
            texts.append(turns(own, issued))
        else:
            if args.diarize:
                log("  note: speaker data incomplete, using plain text for this part")
            texts.append(plain)
        if it.status != "completed":     # e.g. output limit hit: the end may be missing
            log(f"  warning: response status {it.status!r}, the end of this part may be missing")
            texts.append(f"[{hms(start)}-{hms(end)} {MISSING}? "
                         f"response status {it.status!r}, the end of this part may be cut off]")
        start = end
    log(f"transcribed {parts - len(failures)} of {parts} part(s)")
    if len(failures) == parts:           # nothing worth saving
        raise failures[-1]
    return "\n\n".join(texts), words


# ---------------------------------------------------------------- whisper

def transcribe_whisper(src, args, vocab):
    from faster_whisper import WhisperModel
    log(f"loading whisper model {args.whisper_model} (downloaded on first use)")
    model = WhisperModel(args.whisper_model, device="auto", compute_type="default")
    log(f"  model loaded, running on {model.model.device}")
    log("finding the parts with speech" + ("" if args.lang else " and detecting the language"))
    segments, info = model.transcribe(
        str(src),
        language=args.lang.split("-")[0] if args.lang else None,  # ISO 639-1
        vad_filter=True,                   # skip non-speech regions
        condition_on_previous_text=False,  # limits repetition loops on long audio
        word_timestamps=args.timestamps,
        initial_prompt=", ".join(vocab) if vocab else None,
        beam_size=5,
    )
    log(f"  language: {info.language} (p={info.language_probability:.2f}); "
        f"{hms(info.duration_after_vad)} of speech in {hms(info.duration)} of audio")
    log("transcribing")
    lines, words, report = [], [], 300
    for seg in segments:                   # decoding happens lazily here
        text = seg.text.strip()
        lines.append(f"[{hms(seg.start)}] {text}" if args.timestamps else text)
        for w in seg.words or []:
            words.append({"text": w.word.strip(), "speaker": None,
                          "start": round(w.start, 3), "end": round(w.end, 3)})
        if seg.end >= report:              # progress every 5 minutes of audio
            log(f"  transcribed up to {hms(seg.end)} of {hms(info.duration)}")
            report = (seg.end // 300 + 1) * 300
    return "\n".join(lines), words


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", help="output stem (default: input without extension)")
    p.add_argument("--backend", choices=["gemini", "whisper"], default="gemini")
    p.add_argument("--lang", help="e.g. pl-PL; omit for auto-detection")
    p.add_argument("--diarize", action=argparse.BooleanOptionalAction,
                   help="speaker labels (gemini; on by default)")
    p.add_argument("--timestamps", action=argparse.BooleanOptionalAction,
                   help="timestamps (on by default)")
    p.add_argument("--smart", action="store_true",
                   help="gemini smart mode: clean, formatted text without speakers")
    p.add_argument("--vocab", type=Path, help="file with one term per line")
    p.add_argument("--whisper-model", default="large-v3")
    args = p.parse_args()

    if args.backend == "whisper" and (args.diarize or args.smart):
        p.error("--diarize/--smart are gemini-only (for local diarization see WhisperX)")
    # Gemini can't combine --smart/--vocab with speaker labels or timestamps
    plain_only = args.backend == "gemini" and bool(args.smart or args.vocab)
    if plain_only and (args.diarize or args.timestamps):
        p.error("--smart/--vocab cannot be combined with --diarize/--timestamps on gemini")
    if plain_only:
        log("note: --smart/--vocab turn off speaker labels and timestamps")
    # meeting defaults: speakers and timestamps unless turned off or ruled out
    if args.diarize is None:
        args.diarize = args.backend == "gemini" and not plain_only
    if args.timestamps is None:
        args.timestamps = not plain_only
    log(f"input: {args.input} ({mb(args.input)})")
    log(f"backend {args.backend}, language {args.lang or 'auto-detect'}, "
        f"speaker labels {'on' if args.diarize else 'off'}, "
        f"timestamps {'on' if args.timestamps else 'off'}"
        + (", smart mode" if args.smart else "")
        + (f", model {args.whisper_model}" if args.backend == "whisper" else ""))

    vocab = []
    if args.vocab:
        vocab = [t.strip() for t in args.vocab.read_text(encoding="utf-8").splitlines() if t.strip()]
        log(f"vocabulary: {len(vocab)} terms from {args.vocab}")
    stem = args.output or str(args.input.with_suffix(""))

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        norm = workdir / "normalised.flac"
        # whisper's voice detection skips quiet speech; evening out loudness prevents that
        log("converting the audio to 16 kHz mono FLAC"
            + (", evening out loudness" if args.backend == "whisper" else ""))
        normalise(args.input, norm, loudness=args.backend == "whisper")
        log(f"  {hms(duration_s(norm))} of audio, {mb(norm)}")
        if args.backend == "gemini":
            text, words = transcribe_gemini(norm, args, vocab, workdir)
        else:
            text, words = transcribe_whisper(norm, args, vocab)

    log(f"writing {stem}.txt")
    Path(f"{stem}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    if words:
        log(f"writing {stem}.words.json ({len(words)} words)")
        Path(f"{stem}.words.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done: {stem}.txt")
    if MISSING in text:
        log(f"warning: some parts were not transcribed, see '{MISSING}' in {stem}.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
