# 윈도우 로그인 시 OC Studio 자동 실행 등록 / 해제
# 등록: autostart.bat   해제: autostart.bat -Remove
param([switch]$Remove)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$name = "OC Studio"

if ($Remove) {
  Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "자동 실행을 해제했어요."; exit 0
}
if (-not (Test-Path .app_password)) {
  Write-Host "먼저 start.bat 을 한 번 실행해서 비밀번호를 정해주세요." -ForegroundColor Red; exit 1
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $PSScriptRoot `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSScriptRoot\start.ps1`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "등록 완료: 윈도우에 로그인하면 OC Studio가 백그라운드로 켜져요. (화면 잠금 상태여도 동작)" -ForegroundColor Green

$ans = Read-Host "밖에서 쓰려면 PC가 절전에 들어가면 안 돼요. 전원 연결 시 절전을 끌까요? (y/n)"
if ($ans -eq "y") {
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  Write-Host "전원 연결 시 절전/최대절전을 껐어요. (모니터 끄기는 그대로)"
}
