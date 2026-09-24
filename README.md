# transcribe

Transcribe meeting recordings (FLAC or anything ffmpeg reads) with Gemini 3.5 Transcribe or local Whisper.

## Setup

```sh
sudo apt install ffmpeg          # if missing
./setup_env.sh                   # creates .venv; adds CUDA libs on NVIDIA machines
source .venv/bin/activate
export GEMINI_API_KEY=...        # for the gemini backend
```

## Usage

```sh
python transcribe_recording.py meeting.flac                       # speakers + timestamps
python transcribe_recording.py meeting.flac --lang pl-PL
python transcribe_recording.py meeting.flac --no-diarize --no-timestamps   # plain text only
python transcribe_recording.py meeting.flac --smart               # cleaned-up text, no speakers
python transcribe_recording.py meeting.flac --backend whisper --lang en
```

Outputs next to the input (or `-o STEM`):

- `meeting.txt`: the conversation, one speaker turn per line (`[00:01:23] [Speaker1]: ...`)
- `meeting.words.json`: per-word speaker and time

If a part of the recording can't be transcribed, the rest is still saved, the gap is marked
`NOT TRANSCRIBED` in `meeting.txt`, and the script exits with status 1.

See `--help` for all options.

## How it works

1. ffmpeg converts the audio to 16 kHz mono FLAC (for Whisper it also evens out loudness).
2. **Gemini** accepts 30 min per request with speakers/timestamps, 60 min without. Longer
   recordings are cut at pauses into parts of up to 28 min (55 min without either),
   planned one at a time. Each part after the first starts 2–10 min early, long enough to
   hear everyone who spoke recently again.
3. Each part is uploaded, transcribed, and deleted from Gemini. Gemini returns every word
   with a time and a speaker label, which only holds within that one request.
4. Speakers are matched across parts on the overlap: each word both parts heard at the same
   moment is a vote for "this label is that earlier speaker" (at least 3 words and a clear
   majority). Anyone unmatched gets a new `SpeakerN`; if they might be an earlier speaker,
   every line of theirs lists who: `[Speaker5 (Speaker1?, Speaker3?)]`.
5. Words repeated in the overlap are dropped, by position, so a word with a garbled time
   isn't lost.
6. The script writes `meeting.txt` and `meeting.words.json`, logging every step on stderr
   with the elapsed time.

**Whisper** transcribes all the audio locally, with no speech detection (on meeting
recordings it skipped over half the speech), and gives no speaker labels.

## What it supports

- Backends: Gemini 3.5 Transcribe (cloud, default) and faster-whisper (local).
- Any audio ffmpeg can read, of any length.
- A meeting transcript by default: speakers, timestamps, and speaker names kept consistent
  across parts, with doubtful speakers flagged (`"maybe"` in `meeting.words.json`).
- Options: `--lang`, `--no-diarize`, `--no-timestamps`, `--smart` (cleaned-up text),
  `--vocab` (custom terms), `-o`, `--whisper-model`.
- Failures don't lose work: rate limits and server errors are retried (waits of 10–80 s); a
  part that still fails or comes back cut short is marked `NOT TRANSCRIBED`, the rest is
  saved, and the exit status is 1. If every part fails, nothing is written.
- If a part's per-word data covers under 90% of its text, that part falls back to plain text.
- Plain-text mode splits Gemini's glued-together sentences onto separate lines.

## Limitations

- **Speaker names** are `Speaker1`…`SpeakerN` only; there's no way to name speakers or to
  give the number of people.
- **Speaker identity comes from Gemini's labels, not voices.** Gemini can mix up similar
  voices within one request, or give someone a new label mid-part; the script flags such
  doubts but can't resolve them. Someone who doesn't speak in an overlap can't be linked
  across the cut, and after a failed part the next part's speakers are all doubtful.
  Gemini supports up to 8 speakers; 3 or more is documented as experimental.
- **Gemini:** `--smart` and `--vocab` can't be combined with speakers or timestamps. The API
  occasionally returns wrong word times. Cost is about $0.005 per audio minute on the paid
  tier, plus 5–30% for the overlaps; on the free tier (billing off on the key's project)
  Google may use the audio to improve its products.
- **Whisper:** no speaker labels. Transcribing all the audio is slow on a CPU (about 3× slower
  than with speech detection on, in a test meeting), and `large-v3` downloads ~3 GB on first
  use; an NVIDIA GPU helps a lot.
- **Plain text:** glued sentences are only split where punctuation marks the join; a part
  that falls back to plain text keeps its overlap, so a few minutes repeat.
- There are no automated tests.
