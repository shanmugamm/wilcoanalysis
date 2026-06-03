param(
    [string]$Data,
    [string]$DataDir = ".",
    [string]$OutputDir = "reports",
    [int]$Clusters = 8,
    [int]$SampleSegments = 0,
    [switch]$PrimaryOnlyMultiProperty
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$ArgsList = @(
    "scripts/run_owner_pipeline.py",
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir,
    "--clusters", "$Clusters"
)

if ($Data) {
    $ArgsList += @("--data", $Data)
}

if ($SampleSegments -gt 0) {
    $ArgsList += @("--sample-segments", "$SampleSegments")
}

if ($PrimaryOnlyMultiProperty) {
    $ArgsList += "--primary-only-multi-property"
}

Push-Location $RepoRoot
try {
    & $Python @ArgsList
} finally {
    Pop-Location
}
