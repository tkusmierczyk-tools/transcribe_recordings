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

- `meeting.txt`: the conversation, one speaker turn per line (`[00:01:23] Speaker1: ...`)
- `meeting.words.json`: per-word speaker and time

If a part of the recording can't be transcribed, the rest is still saved, the gap is marked
`NOT TRANSCRIBED` in `meeting.txt`, and the script exits with status 1.

See `--help` for all options.

## Notes

- Long recordings are split at pauses into parts of up to 28 min with speakers/timestamps
  (default), 55 min without. Each part repeats the end of the previous one (2–10 min, long
  enough to hear everyone who spoke recently), and speakers are matched on the words both
  parts heard there, so `Speaker1` stays the same person. Someone who can't be matched gets
  a new label; if they might be an earlier speaker it has a `?`, and the transcript says who
  (`[Speaker5? may be Speaker1 or Speaker3 ...]`). The overlap adds roughly 5–30% more audio
  to transcribe.
- Gemini: free tier if billing is off on the key's project, but Google may use the audio to
  improve its products. Paid tier costs about $0.005 per audio minute.
- Whisper runs locally, with no speaker labels. The first run downloads the model (~3 GB for
  `large-v3`), and it's slow without an NVIDIA GPU.
