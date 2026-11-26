param([string]$Storage = "D:\SarcasTone")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:SARCASTONE_HOME = $Storage
$env:PIP_CACHE_DIR = Join-Path $Storage "environment\pip-cache"
$env:TMP = Join-Path $Storage "environment\temp"
$env:TEMP = $env:TMP
New-Item $env:TMP -ItemType Directory -Force | Out-Null
$venv = Join-Path $Storage "environment\venv"
if (!(Test-Path "$venv\Scripts\python.exe")) {
    py -3.10 -m venv $venv
    if ($LASTEXITCODE) { throw "Python 3.10 venv creation failed." }
}
& "$venv\Scripts\python.exe" -m pip install -r requirements-cpu.txt
if ($LASTEXITCODE) { throw "Dependency installation failed. Retry this script." }
& "$venv\Scripts\python.exe" -m pip check
if ($LASTEXITCODE) { throw "Dependency verification failed." }
Write-Output "VS Code interpreter: $venv\Scripts\python.exe"
