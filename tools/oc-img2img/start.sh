#!/usr/bin/env bash
# ComfyUI + OC Studio 웹앱 동시 실행 (Linux / macOS)
# 사용: APP_PASSWORD=비밀번호 ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${APP_PASSWORD:?APP_PASSWORD 환경변수를 지정해주세요 (8자 이상)}"
source ComfyUI/venv/bin/activate
python ComfyUI/main.py --listen 127.0.0.1 --port 8188 &
COMFY=$!
trap 'kill $COMFY 2>/dev/null' EXIT
python server.py --port 8787
