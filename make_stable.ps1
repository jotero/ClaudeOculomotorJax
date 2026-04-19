$ErrorActionPreference = 'Stop'

$short = git rev-parse --short HEAD
$dirty = git status --porcelain

if ($dirty) {
    Write-Host 'WARNING: uncommitted changes present. Proceed anyway? (y/n)' -NoNewline
    $ans = Read-Host
    if ($ans -ne 'y') { exit 0 }
}

git branch -f stable HEAD
Write-Host "Stable branch updated to $short"
Write-Host 'Run start_stable_server.ps1 to serve it on port 8000'
