# ComfyUI + OC Studio 웹앱 동시 실행 (Windows)
# 사용: $env:APP_PASSWORD="비밀번호"; powershell -ExecutionPolicy Bypass -File start.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $env:APP_PASSWORD) { throw "APP_PASSWORD 환경변수를 지정해주세요 (8자 이상)" }
$py = ".\ComfyUI\venv\Scripts\python.exe"
$comfy = Start-Process $py -ArgumentList "ComfyUI\main.py --listen 127.0.0.1 --port 8188" -PassThru -NoNewWindow
try { & $py server.py --port 8787 } finally { Stop-Process -Id $comfy.Id -ErrorAction SilentlyContinue }
