$ErrorActionPreference = 'Stop'

$short = git rev-parse --short HEAD
$dirty = git status --porcelain

if ($dirty) {
    Write-Host 'WARNING: uncommitted changes present. Proceed anyway? (y/n)' -NoNewline
    $ans = Read-Host
    if ($ans -ne 'y') { exit 0 }
}

$worktree = 'D:/OneDrive/UC Berkeley/OMlab - JOM/Code/om-stable'

if (Test-Path $worktree) {
    # Branch is checked out in worktree — reset it there directly
    Push-Location $worktree
    git reset --hard "$(git -C "$PSScriptRoot" rev-parse HEAD)"
    Pop-Location
} else {
    git branch -f stable HEAD
}

Write-Host "Stable branch updated to $short"
Write-Host 'Run start_stable_server.ps1 to serve it on port 8000'
