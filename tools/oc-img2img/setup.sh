#!/usr/bin/env bash
# ComfyUI + IP-Adapter(SDXL) 로컬 설치 (Linux / macOS)
# 사용: ./setup.sh            → 전체 설치 + SDXL base 체크포인트
#       SKIP_CKPT=1 ./setup.sh → 체크포인트는 직접 넣을 때
set -euo pipefail
cd "$(dirname "$0")"

HF=https://huggingface.co
[ -d ComfyUI ] || git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
if command -v nvidia-smi >/dev/null; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
  pip install torch torchvision torchaudio
fi
pip install -r requirements.txt

cd custom_nodes
[ -d ComfyUI_IPAdapter_plus ] || git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
[ -d ComfyUI-Manager ] || git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ..

dl() { [ -f "$2" ] || curl -L --fail -o "$2" "$1"; }
mkdir -p models/clip_vision models/ipadapter models/checkpoints
dl "$HF/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
   models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
dl "$HF/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" \
   models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors
dl "$HF/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" \
   models/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors
if [ "${SKIP_CKPT:-0}" != "1" ]; then
  dl "$HF/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
     models/checkpoints/sd_xl_base_1.0.safetensors
fi

echo
echo "설치 완료. 실행: cd ComfyUI && source venv/bin/activate && python main.py"
