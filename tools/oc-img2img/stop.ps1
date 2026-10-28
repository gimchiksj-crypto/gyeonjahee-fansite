# OC Studio 종료 (start.bat 창을 그냥 닫았을 때 남은 백그라운드 프로세스 정리)
$here = $PSScriptRoot
$procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -and $_.CommandLine.Contains($here) -or
                 ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($here)) }
if (-not $procs) { Write-Host "실행 중인 OC Studio가 없어요."; exit 0 }
$procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host ("종료했어요 ({0}개 프로세스)." -f @($procs).Count)
