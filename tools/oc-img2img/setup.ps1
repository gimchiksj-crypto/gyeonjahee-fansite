# ComfyUI + IP-Adapter(SDXL) 로컬 설치 (Windows PowerShell, NVIDIA GPU 기준)
# 사용: powershell -ExecutionPolicy Bypass -File setup.ps1 [-SkipCkpt]
param([switch]$SkipCkpt)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$HF = "https://huggingface.co"
if (-not (Test-Path ComfyUI)) { git clone https://github.com/comfyanonymous/ComfyUI.git }
Set-Location ComfyUI
python -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

Push-Location custom_nodes
if (-not (Test-Path ComfyUI_IPAdapter_plus)) { git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git }
if (-not (Test-Path ComfyUI-Manager)) { git clone https://github.com/ltdrdata/ComfyUI-Manager.git }
Pop-Location

function Get-Model($url, $dest) {
  if (-not (Test-Path $dest)) { curl.exe -L --fail -o $dest $url }
}
New-Item -ItemType Directory -Force models\clip_vision, models\ipadapter, models\checkpoints | Out-Null
Get-Model "$HF/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" `
  "models\clip_vision\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
Get-Model "$HF/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" `
  "models\ipadapter\ip-adapter-plus_sdxl_vit-h.safetensors"
Get-Model "$HF/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" `
  "models\ipadapter\ip-adapter-plus-face_sdxl_vit-h.safetensors"
if (-not $SkipCkpt) {
  Get-Model "$HF/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" `
    "models\checkpoints\sd_xl_base_1.0.safetensors"
}

Write-Host "`n설치 완료. 실행: cd ComfyUI; .\venv\Scripts\Activate.ps1; python main.py"
