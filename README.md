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
python transcribe_recording.py meeting.flac --lang en-US --diarize --timestamps
python transcribe_recording.py meeting.flac --smart               # cleaned-up text
python transcribe_recording.py meeting.flac --backend whisper --lang en
```

Outputs next to the input (or `-o STEM`):

- `meeting.txt`: the transcript, one utterance per line
- `meeting.turns.txt`: who said what (`--diarize`)
- `meeting.words.json`: per-word speaker and time (`--diarize` / `--timestamps`)

See `--help` for all options.

## Notes

- Long recordings are split at pauses: 55 min chunks, or 28 min with `--diarize`/`--timestamps`.
  Speaker labels don't carry across chunks (`c0:spk_1` and `c1:spk_1` may be different people).
- Gemini: free tier if billing is off on the key's project, but Google may use the audio to
  improve its products. Paid tier costs about $0.005 per audio minute.
- Whisper runs locally, with no speaker labels. The first run downloads the model (~3 GB for
  `large-v3`), and it's slow without an NVIDIA GPU.
