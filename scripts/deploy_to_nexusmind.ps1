# Deploy a filter from llm-distillery to NexusMind
#
# Usage: .\scripts\deploy_to_nexusmind.ps1 <filter_name> <version> [-Push]
#
# Examples:
#   .\scripts\deploy_to_nexusmind.ps1 uplifting v5
#   .\scripts\deploy_to_nexusmind.ps1 sustainability_technology v2 -Push
#
# What it does:
#   1. Copies filter folder to NexusMind
#   2. Copies filters/common/ (shared utilities)
#   3. Commits changes to NexusMind repo
#   4. Optionally pushes and shows pull commands for servers

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$FilterName,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$Version,

    [switch]$Push
)

$ErrorActionPreference = "Stop"

# Configuration
$DistilleryRoot = "C:\local_dev\llm-distillery"
$NexusMindRoot = "C:\local_dev\NexusMind"

$FilterPath = "filters\$FilterName\$Version"
$SourceDir = Join-Path $DistilleryRoot $FilterPath
$DestDir = Join-Path $NexusMindRoot $FilterPath
$CommonSource = Join-Path $DistilleryRoot "filters\common"
$CommonDest = Join-Path $NexusMindRoot "filters\common"

# Validate source exists
if (-not (Test-Path $SourceDir)) {
    Write-Error "ERROR: Filter not found: $SourceDir"
    exit 1
}

Write-Host "=== Deploying $FilterName $Version to NexusMind ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Copy filter folder
Write-Host "1. Copying filter: $FilterPath" -ForegroundColor Yellow
if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}
Copy-Item -Path "$SourceDir\*" -Destination $DestDir -Recurse -Force
Write-Host "   Copied to: $DestDir" -ForegroundColor Green

# Step 2: Copy common utilities
Write-Host ""
Write-Host "2. Copying common utilities: filters\common\" -ForegroundColor Yellow
if (-not (Test-Path $CommonDest)) {
    New-Item -ItemType Directory -Path $CommonDest -Force | Out-Null
}
Copy-Item -Path "$CommonSource\*" -Destination $CommonDest -Recurse -Force
Write-Host "   Copied to: $CommonDest" -ForegroundColor Green

# Step 3: Git status in NexusMind
Write-Host ""
Write-Host "3. Changes in NexusMind:" -ForegroundColor Yellow
Push-Location $NexusMindRoot
git status --short

# Step 4: Commit
Write-Host ""
Write-Host "4. Committing changes..." -ForegroundColor Yellow
git add -A
$CommitMsg = "Update $FilterName $Version from llm-distillery"
$commitResult = git commit -m $CommitMsg 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   Committed: $CommitMsg" -ForegroundColor Green
} else {
    Write-Host "   (No changes to commit)" -ForegroundColor DarkGray
}

# Step 5: Push if requested
if ($Push) {
    Write-Host ""
    Write-Host "5. Pushing to origin..." -ForegroundColor Yellow
    git push origin main

    Write-Host ""
    Write-Host "=== Deploy commands for servers ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "# Sadalsuud:" -ForegroundColor White
    Write-Host 'ssh user@sadalsuud "cd ~/NexusMind && git pull origin main"' -ForegroundColor Gray
    Write-Host ""
    Write-Host "# llm-distiller:" -ForegroundColor White
    Write-Host 'ssh jeroen@llm-distiller "cd ~/NexusMind && git pull origin main"' -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "5. Skipping push (use -Push flag to push automatically)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "=== Next steps ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "# Push to origin:" -ForegroundColor White
    Write-Host "cd $NexusMindRoot; git push origin main" -ForegroundColor Gray
    Write-Host ""
    Write-Host "# Then pull on servers:" -ForegroundColor White
    Write-Host 'ssh user@sadalsuud "cd ~/NexusMind && git pull origin main"' -ForegroundColor Gray
    Write-Host 'ssh jeroen@llm-distiller "cd ~/NexusMind && git pull origin main"' -ForegroundColor Gray
}

Pop-Location

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
