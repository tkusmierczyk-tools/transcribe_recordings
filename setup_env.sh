#!/usr/bin/env bash
# Prepare a virtualenv for transcribe_recording.py.
#
#   ./setup_env.sh                     # creates .venv next to this script
#   PYTHON=python3.12 ./setup_env.sh   # use a specific interpreter
#   source .venv/bin/activate          # then run the transcription script
#
# Safe to re-run: it reuses the existing .venv and only installs what's missing.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}

# ffmpeg/ffprobe do the normalising, probing and chunking
missing=()
for tool in ffmpeg ffprobe; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if ((${#missing[@]})); then
    echo "error: missing ${missing[*]} -- install it with: sudo apt install ffmpeg" >&2
    exit 1
fi

[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt

# NVIDIA GPU: faster-whisper needs cuBLAS 12 on the library path
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA GPU found, installing CUDA libraries for the whisper backend"
    "$VENV/bin/python" -m pip install -r requirements-gpu.txt
    lib=$(find "$VENV" -path '*/nvidia/cublas/lib/libcublas.so.12' -print -quit)
    marker="# transcribe_recording: cuBLAS for faster-whisper"
    if [ -n "$lib" ] && ! grep -qF "$marker" "$VENV/bin/activate"; then
        printf '\n%s\nexport LD_LIBRARY_PATH="%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' \
            "$marker" "$(cd "$(dirname "$lib")" && pwd)" >> "$VENV/bin/activate"
    fi
else
    echo "no NVIDIA GPU found: the whisper backend will run on the CPU"
fi

"$VENV/bin/python" -c "import google.genai, faster_whisper" \
    && echo "ok: packages import cleanly"
[ -n "${GEMINI_API_KEY:-}" ] \
    || echo "note: GEMINI_API_KEY is not set (needed for --backend gemini)"
echo
echo "done. activate with:  source $VENV/bin/activate"
