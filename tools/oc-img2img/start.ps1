# ComfyUI + OC Studio 웹앱 실행 — Windows
# 더블클릭: start.bat  (처음 한 번 폰 앱 비밀번호를 정하면 .app_password 에 저장돼요)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "ComfyUI\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "먼저 setup.bat 을 실행해주세요." -ForegroundColor Red; exit 1 }

if (Get-NetTCPConnection -LocalPort 8787 -State Listen -ErrorAction SilentlyContinue) {
  Write-Host "이미 실행 중이에요 (포트 8787)." -ForegroundColor Yellow; exit 0
}

$pwFile = Join-Path $PSScriptRoot ".app_password"
if (-not $env:APP_PASSWORD) {
  if (Test-Path $pwFile) {
    $env:APP_PASSWORD = (Get-Content $pwFile -Raw -Encoding UTF8).Trim()
  } else {
    do {
      $ss = Read-Host "폰 앱 비밀번호를 정해주세요 (8자 이상)" -AsSecureString
      $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($ss))
    } while ($plain.Length -lt 8)
    Set-Content $pwFile $plain -Encoding UTF8 -NoNewline
    $env:APP_PASSWORD = $plain
    Write-Host "비밀번호를 .app_password 에 저장했어요. 바꾸려면 그 파일을 지우고 다시 실행하세요."
  }
}

$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force logs | Out-Null
$log = Join-Path $PSScriptRoot "logs"
$comfy = Start-Process $py -WindowStyle Hidden -PassThru -WorkingDirectory $PSScriptRoot `
  -ArgumentList "-u", "ComfyUI\main.py", "--listen", "127.0.0.1", "--port", "8188" `
  -RedirectStandardOutput "$log\comfyui.log" -RedirectStandardError "$log\comfyui.err.log"
$app = Start-Process $py -WindowStyle Hidden -PassThru -WorkingDirectory $PSScriptRoot `
  -ArgumentList "-u", "server.py", "--port", "8787" `
  -RedirectStandardOutput "$log\server.log" -RedirectStandardError "$log\server.err.log"

Write-Host "OC Studio 실행 중  →  PC: http://127.0.0.1:8787" -ForegroundColor Green
if (Get-Command tailscale -ErrorAction SilentlyContinue) {
  $url = $null
  try { $url = (tailscale serve status) -join "`n" | Select-String -Pattern 'https://\S+' | ForEach-Object { $_.Matches[0].Value } } catch {}
  if ($url) { Write-Host "폰: $url" -ForegroundColor Green }
  else { Write-Host "폰 접속 주소가 아직 없어요. 한 번만 'tailscale serve --bg 8787' 을 실행해주세요." -ForegroundColor Yellow }
}
Write-Host "로그: logs\  /  이 창을 닫거나 Ctrl+C 하면 종료돼요."

try { Wait-Process -Id $app.Id }
finally {
  foreach ($p in $comfy, $app) { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
}
