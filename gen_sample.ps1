param(
  [Parameter(Mandatory=$true)][string]$Voice,
  [Parameter(Mandatory=$true)][string]$InputTxt,
  [Parameter(Mandatory=$true)][string]$OutWav
)
# Generate a sample from a registered Orpheus voice via the BookForge CLI.
# Clean param binding avoids the nested-quote path mangling that breaks a raw
# `cmd /c "... --input E:\..."` invocation from WSL.
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\tellt\Projects\bookforge'
$env:PYTHONIOENCODING = 'utf-8'
& 'C:\Users\tellt\AppData\Local\Programs\Python\Python311\python.exe' 'cli\bookforge-tts.py' `
  --tts --engine=orpheus --voice=$Voice --input $InputTxt --out $OutWav
exit $LASTEXITCODE
