# Save the current commit as the stable server version.
# Run this whenever the simulation is working well.

$ErrorActionPreference = 'Stop'

$short = git rev-parse --short HEAD
$dirty = git status --porcelain

if ($dirty) {
    Write-Host "WARNING: you have uncommitted changes — committing a snapshot first is safer."
    Write-Host "         Proceed anyway? (y/n)" -NoNewline
    $ans = Read-Host
    if ($ans -ne 'y') { exit 0 }
}

git branch -f stable HEAD
Write-Host "Stable branch updated to $short"
Write-Host "Run start_stable_server.ps1 to serve it on port 8000."
