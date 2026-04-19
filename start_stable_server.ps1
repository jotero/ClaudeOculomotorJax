# Start the stable server on port 8000.
# The stable version is whatever was last saved with make_stable.ps1.
# Your dev server can still run on port 8001 from the main working directory.

$ErrorActionPreference = 'Stop'

$root    = $PSScriptRoot
$stable  = Join-Path (Split-Path $root -Parent) 'om-stable'
$python  = Join-Path $root '.venv\Scripts\python.exe'

# Check stable branch exists
$branch = git branch --list stable
if (-not $branch) {
    Write-Host "No stable version saved yet. Run make_stable.ps1 first."
    exit 1
}

# Create or refresh the worktree
if (Test-Path $stable) {
    Write-Host "Updating stable worktree..."
    git -C $stable checkout stable 2>$null
    git -C $stable reset --hard stable 2>$null
} else {
    Write-Host "Creating stable worktree at $stable ..."
    git worktree add $stable stable
}

$ver = git -C $stable rev-parse --short HEAD
Write-Host "Starting stable server (commit $ver) on http://localhost:8000"
Write-Host "Ctrl-C to stop."

& $python -X utf8 (Join-Path $stable 'scripts\server.py') --port 8000
