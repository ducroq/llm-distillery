# Deploy a filter from llm-distillery to NexusMind
#
# Usage: .\scripts\deploy_to_nexusmind.ps1 <filter_name> <version>
#                                           [-Push] [-DryRun]
#                                           [-ForceSkipOwnedDrift]
#
# Examples:
#   .\scripts\deploy_to_nexusmind.ps1 uplifting v5
#   .\scripts\deploy_to_nexusmind.ps1 sustainability_technology v2 -Push
#   .\scripts\deploy_to_nexusmind.ps1 nature_recovery v2 -DryRun
#
# What it does:
#   1. Copies filter folder to NexusMind
#   2. Copies filters/common/ (shared utilities) — honors .nexusmind-owns
#      manifest at repo root: listed files are skipped, and the deploy fails
#      if a listed file has drifted from NexusMind's copy (issue #50).
#   3. Commits changes to NexusMind repo
#   4. Optionally pushes and shows pull commands for servers

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$FilterName,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$Version,

    [switch]$Push,
    [switch]$DryRun,
    [switch]$ForceSkipOwnedDrift
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
# Defensive: strip any trailing backslash so the Substring offset below is
# correct whether the join inserted one or not.
$CommonSource = $CommonSource.TrimEnd('\')
$CommonDest = $CommonDest.TrimEnd('\')

# Validate source exists
if (-not (Test-Path $SourceDir)) {
    Write-Error "ERROR: Filter not found: $SourceDir"
    exit 1
}

Write-Host "=== Deploying $FilterName $Version to NexusMind ===" -ForegroundColor Cyan
Write-Host ""

# Sanity: refuse to ship uncommitted changes (matches bash-side check).
$dirtyUnstaged  = git -C $DistilleryRoot diff --quiet -- $FilterPath; $dirtyUnstagedExit  = $LASTEXITCODE
$dirtyStaged    = git -C $DistilleryRoot diff --cached --quiet -- $FilterPath; $dirtyStagedExit    = $LASTEXITCODE
if ($dirtyUnstagedExit -ne 0 -or $dirtyStagedExit -ne 0) {
    Write-Host "ERROR: uncommitted changes in $FilterPath. Commit first, then re-run." -ForegroundColor Red
    git -C $DistilleryRoot status --short $FilterPath | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

# Step 0: Verify package is internally consistent (issue #44)
# Catches v_new config x v_old weights mismatches before copying.
Write-Host "0. Verifying filter package..." -ForegroundColor Yellow
$savedPythonPath = $env:PYTHONPATH
Push-Location $DistilleryRoot
try {
    $env:PYTHONPATH = "."
    python scripts\deployment\verify_filter_package.py --filter "$FilterPath" --check-hub
    if ($LASTEXITCODE -ne 0) {
        Write-Error "verify_filter_package failed. Aborting deploy. Fix the package (imports / repo_id / Hub upload) before retrying."
        exit 1
    }
}
finally {
    $env:PYTHONPATH = $savedPythonPath
    Pop-Location
}
Write-Host ""

# Step 1: Copy filter folder
Write-Host "1. Copying filter: $FilterPath" -ForegroundColor Yellow
if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}
Copy-Item -Path "$SourceDir\*" -Destination $DestDir -Recurse -Force
Write-Host "   Copied to: $DestDir" -ForegroundColor Green

# Step 2: Copy common utilities, honoring .nexusmind-owns (issue #50).
# Files listed in .nexusmind-owns evolve independently in NexusMind (e.g. the
# BFloat16 .float() cast in filter_base_scorer.py) and must NOT be overwritten
# by a blind sync from this repo.
Write-Host ""
Write-Host "2. Copying common utilities: filters\common\ (honoring .nexusmind-owns)" -ForegroundColor Yellow
if (-not (Test-Path $CommonDest)) {
    New-Item -ItemType Directory -Path $CommonDest -Force | Out-Null
}

$ManifestPath = Join-Path $DistilleryRoot ".nexusmind-owns"
$OwnedPaths = @()
if (Test-Path $ManifestPath) {
    Get-Content $ManifestPath | ForEach-Object {
        $line = ($_ -split '#', 2)[0].Trim()
        if ($line) { $OwnedPaths += $line }
    }
}

# Typo guard: every manifest entry must exist on at least one side.
foreach ($owned in $OwnedPaths) {
    $distSide = Join-Path $DistilleryRoot ($owned -replace '/', '\')
    $nmSide   = Join-Path $NexusMindRoot ($owned -replace '/', '\')
    if (-not (Test-Path $distSide) -and -not (Test-Path $nmSide)) {
        Write-Error "ERROR: .nexusmind-owns entry not found on either side: $owned (fix the typo or remove the line)"
        exit 1
    }
}

$OwnedSet = @{}
foreach ($p in $OwnedPaths) { $OwnedSet[$p] = $true }
$script:DriftFound = $false

# Walk the source tree and copy file-by-file, skipping owned files. Any drift
# between distillery and NexusMind copies of an owned file is collected and
# fails the deploy after the loop (unless -ForceSkipOwnedDrift was passed).
Get-ChildItem -Path $CommonSource -Recurse -File |
    Where-Object { $_.FullName -notmatch '\\__pycache__\\' } |
    ForEach-Object {
        $relInside = $_.FullName.Substring($CommonSource.Length + 1) -replace '\\', '/'
        $relFromRoot = "filters/common/$relInside"

        if ($OwnedSet.ContainsKey($relFromRoot)) {
            $nm = Join-Path $NexusMindRoot ($relFromRoot -replace '/', '\')
            if ((Test-Path $nm) -and (Get-FileHash -Algorithm SHA256 $_.FullName).Hash -ne (Get-FileHash -Algorithm SHA256 $nm).Hash) {
                Write-Host "   DRIFT NexusMind-owned: $relFromRoot" -ForegroundColor Red
                $script:DriftFound = $true
            } else {
                Write-Host "   skip  NexusMind-owned: $relFromRoot" -ForegroundColor DarkGray
            }
            return
        }

        $destFull = Join-Path $CommonDest ($relInside -replace '/', '\')
        $destDir = Split-Path $destFull -Parent
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
        Copy-Item -Path $_.FullName -Destination $destFull -Force
    }

if ($script:DriftFound -and -not $ForceSkipOwnedDrift) {
    Write-Host ""
    Write-Host "ERROR: NexusMind-owned files have drifted from this repo (DRIFT lines above)." -ForegroundColor Red
    Write-Host "       Inspect with: Compare-Object (Get-Content $CommonSource\<file>) (Get-Content $CommonDest\<file>)" -ForegroundColor Red
    Write-Host "       Then either:" -ForegroundColor Red
    Write-Host "         (a) back-port the NexusMind change to this repo and re-run, or" -ForegroundColor Red
    Write-Host "         (b) re-run with -ForceSkipOwnedDrift to keep NexusMind's copy." -ForegroundColor Red
    exit 1
}
Write-Host "   Copied to: $CommonDest" -ForegroundColor Green

# Step 3: Git status in NexusMind
Write-Host ""
Write-Host "3. Changes in NexusMind:" -ForegroundColor Yellow
Push-Location $NexusMindRoot
git status --short

# Step 4: Commit
Write-Host ""
if ($DryRun) {
    Write-Host "4. DRY RUN: skipping git add/commit. Inspect $NexusMindRoot, then revert with" -ForegroundColor DarkGray
    Write-Host "   'git -C $NexusMindRoot checkout -- .' if you do not want to keep the changes." -ForegroundColor DarkGray
} else {
    Write-Host "4. Committing changes..." -ForegroundColor Yellow
    git add -A
    $CommitMsg = "Update $FilterName $Version from llm-distillery"
    $commitResult = git commit -m $CommitMsg 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Committed: $CommitMsg" -ForegroundColor Green
    } else {
        Write-Host "   (No changes to commit)" -ForegroundColor DarkGray
    }
}

# Step 5: Push if requested
if ($DryRun) {
    Write-Host ""
    Write-Host "5. DRY RUN: skipping push." -ForegroundColor DarkGray
} elseif ($Push) {
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
