#!/usr/bin/env bash
# Chạy đánh mask Gemini: tự nạp key từ scripts/.gemini_env, kiểm SDK, gọi src/data_prep/gemini_mask.py.
# Usage:
#   bash scripts/data_prep/gemini_mask.sh --limit 30                 # thử nhỏ trước
#   bash scripts/data_prep/gemini_mask.sh --sample 2000 --concurrency 4
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python}"

ENVF="$REPO/scripts/.gemini_env"
[ -f "$ENVF" ] && source "$ENVF" || { echo "Thiếu $ENVF (copy từ .gemini_env.example)"; exit 1; }
[ -n "${GEMINI_API_KEY:-}" ] || { echo "!! GEMINI_API_KEY còn trống — điền vào $ENVF"; exit 1; }

"$PY" -c "import google.genai" 2>/dev/null || {
    echo "Chưa có SDK. Cài: $PY -m pip install google-genai"; exit 1; }

MODEL_ARG=(); [ -n "${GEMINI_MODEL:-}" ] && MODEL_ARG=(--model "$GEMINI_MODEL")
"$PY" -m src.data_prep.gemini_mask "${MODEL_ARG[@]}" "$@"
