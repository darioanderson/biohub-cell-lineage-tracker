param([int]$Port = 8501)
$projectRoot = Split-Path -Parent $PSScriptRoot
$sharedPython = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $sharedPython) { $appPython = $sharedPython }
elseif (Test-Path -LiteralPath $localPython) { $appPython = $localPython }
else { throw 'Create an environment and install this project first. See SUBMISSION_GUIDE.md.' }
Set-Location -LiteralPath $projectRoot
& $appPython -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port
