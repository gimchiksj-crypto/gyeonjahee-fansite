# ComfyUI + IP-Adapter(SDXL) 설치 — Windows + NVIDIA (RTX 40 시리즈 기준)
# 더블클릭: setup.bat   /  체크포인트는 직접 넣을 때: setup.bat -SkipCkpt
param([switch]$SkipCkpt)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail($msg) { Write-Host "`n[중단] $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

Step "사전 점검"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Fail "git이 없어요. 'winget install --id Git.Git -e' 로 설치하고 창을 다시 열어주세요."
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  Fail "Python이 없어요. 'winget install --id Python.Python.3.12 -e' 로 설치하고 창을 다시 열어주세요."
}
& py -3.12 -c "import sys"
if ($LASTEXITCODE -ne 0) {
  Fail "Python 3.12가 필요해요. 'winget install --id Python.Python.3.12 -e' 로 설치해주세요. (다른 버전과 같이 있어도 괜찮아요)"
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  Write-Warning "nvidia-smi가 안 보여요. NVIDIA 드라이버를 최신으로 설치해주세요 (GeForce Experience / NVIDIA App)."
}
if ($PSScriptRoot -match '[^\x00-\x7F]' -or $PSScriptRoot -match ' ') {
  Write-Warning "설치 경로에 한글이나 공백이 있어요: $PSScriptRoot"
  Write-Warning "일부 패키지가 깨질 수 있으니 C:\AI\oc-img2img 같은 경로로 옮기는 걸 권장해요."
}
$free = (Get-PSDrive ($PSScriptRoot.Substring(0, 1))).Free / 1GB
if ($free -lt 20) { Write-Warning ("디스크 여유 공간이 {0:N0}GB예요. 20GB 이상 권장해요." -f $free) }

Step "ComfyUI 받기"
if (-not (Test-Path ComfyUI)) { git clone https://github.com/comfyanonymous/ComfyUI.git }
Push-Location ComfyUI\custom_nodes
if (-not (Test-Path ComfyUI_IPAdapter_plus)) { git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git }
if (-not (Test-Path ComfyUI-Manager)) { git clone https://github.com/ltdrdata/ComfyUI-Manager.git }
Pop-Location

Step "Python 가상환경 + PyTorch (CUDA 12.8)"
if (-not (Test-Path ComfyUI\venv\Scripts\python.exe)) { & py -3.12 -m venv ComfyUI\venv }
$py = Join-Path $PSScriptRoot "ComfyUI\venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Fail "PyTorch 설치 실패" }
& $py -m pip install -r ComfyUI\requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "ComfyUI 패키지 설치 실패" }

Step "모델 다운로드"
$HF = "https://huggingface.co"
function Get-Model($url, $dest) {
  if (Test-Path $dest) { Write-Host "  있음: $dest"; return }
  Write-Host "  받는 중: $dest"
  curl.exe -L --fail --retry 3 -o "$dest.part" $url
  if ($LASTEXITCODE -ne 0) { Fail "다운로드 실패: $url" }
  Move-Item "$dest.part" $dest
}
New-Item -ItemType Directory -Force ComfyUI\models\clip_vision, ComfyUI\models\ipadapter, ComfyUI\models\checkpoints | Out-Null
Get-Model "$HF/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" `
  "ComfyUI\models\clip_vision\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
Get-Model "$HF/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" `
  "ComfyUI\models\ipadapter\ip-adapter-plus_sdxl_vit-h.safetensors"
Get-Model "$HF/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" `
  "ComfyUI\models\ipadapter\ip-adapter-plus-face_sdxl_vit-h.safetensors"
if (-not $SkipCkpt) {
  Get-Model "$HF/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" `
    "ComfyUI\models\checkpoints\sd_xl_base_1.0.safetensors"
}

Step "GPU 확인"
& $py -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'GPU 인식 안 됨')"

Write-Host "`n설치 완료. 이제 start.bat 을 실행하세요." -ForegroundColor Green
