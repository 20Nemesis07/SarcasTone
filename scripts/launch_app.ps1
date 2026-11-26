param([string]$Storage = "D:\SarcasTone")
Set-Location (Join-Path $PSScriptRoot "..")
$env:SARCASTONE_HOME = $Storage
& "$Storage\environment\venv\Scripts\python.exe" -m streamlit run app.py
exit $LASTEXITCODE
