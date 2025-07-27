#Requires -Version 5.1
<#
.SYNOPSIS
    PowerShell script to run a hyperparameter sweep for a reinforcement learning project.
    This script is a Windows-native replacement for the original SLURM bash script.
    It iterates through a CSV file of parameters and runs the main Python script for each combination.
#>

# --- Configuration ---
$VENV_DIR = ".\venv"
$LOGS_DIR = ".\logs"
$GRID_FILE = ".\rl_grid.csv"

# --- Environment Setup ---
Write-Host "Checking for virtual environment..." -ForegroundColor Cyan

if (-not (Test-Path -Path $VENV_DIR) -or -not (Test-Path -Path "$VENV_DIR\Scripts\Activate.ps1")) {
    Write-Host "[INFO] Virtualenv not found. Creating new environment at $VENV_DIR..." -ForegroundColor Yellow
    python -m venv $VENV_DIR
    
    Write-Host "[INFO] Activating new environment..." -ForegroundColor Cyan
    . "$VENV_DIR\Scripts\Activate.ps1"
    
    Write-Host "[INFO] Installing requirements..." -ForegroundColor Cyan
    pip install --upgrade pip
    if (Test-Path -Path ".\requirements.txt") {
        pip install -r .\requirements.txt
    } else {
        Write-Host "[WARN] No requirements.txt found. Skipping dependency install." -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] Activating existing virtualenv..." -ForegroundColor Cyan
    . "$VENV_DIR\Scripts\Activate.ps1"
}

# --- Create Logs Directory ---
if (-not (Test-Path -Path $LOGS_DIR)) {
    New-Item -ItemType Directory -Path $LOGS_DIR | Out-Null
}

# --- Parameter Sweep ---
if (-not (Test-Path -Path $GRID_FILE)) {
    Write-Host "[ERROR] Grid file not found at $GRID_FILE. Exiting." -ForegroundColor Red
    exit 1
}

# Import the CSV file with all the parameter combinations
$Grid = Import-Csv -Path $GRID_FILE

Write-Host "Starting parameter sweep from '$GRID_FILE'..." -ForegroundColor Green

# Loop through each row in the CSV
foreach ($params in $Grid) {
    
    # === Build Arguments for the Python script ===
    $RL_ARGS = @()
    $RL_ARGS += "--model=$($params.model)"
    $RL_ARGS += "--degree=$($params.degree)"
    $RL_ARGS += "--buffer_size=$($params.buffer)"
    $RL_ARGS += "-v=$($params.verbosity)"
    $RL_ARGS += "-s=$($params.seed)"
    $RL_ARGS += $params.run_mode
    $RL_ARGS += $params.agent

    # === Compose a unique log filename for this run ===
    $logName = "train_$($params.agent)_$($params.model)"
    if ($params.model -eq "poly") {
        $logName += "_d$($params.degree)"
    }
    if ($params.agent -eq "ppo") {
        $logName += "_b$($params.buffer)"
    }
    $logName += "_s$($params.seed)"
    $timestamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
    $logName += "_$timestamp.log"
    $LOGRUN = Join-Path -Path $LOGS_DIR -ChildPath $logName

    # === Thread settings for performance ===
    # Use all available logical processors
    $coreCount = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
    $env:OMP_NUM_THREADS = $coreCount
    $env:MKL_NUM_THREADS = $coreCount
    $env:OPENBLAS_NUM_THREADS = $coreCount
    $env:VECLIB_MAXIMUM_THREADS = $coreCount

    # === Log environment and Launch ===
    # Combine the script name and arguments into a single flat array
    $Full_ARGS = @("src\main.py") + $RL_ARGS
    $CMD = "python.exe $($Full_ARGS -join ' ')"
    
    $logHeader = @"
======================================================================
LAUNCHING JOB
----------------------------------------------------------------------
Timestamp:    $(Get-Date)
Parameters:   model=$($params.model), degree=$($params.degree), buffer=$($params.buffer), seed=$($params.seed), agent=$($params.agent)
Command:      $CMD
Log File:     $LOGRUN
Core Count:   $coreCount
======================================================================
"@
    
    # Write header to log file
    Add-Content -Path $LOGRUN -Value $logHeader
    
    Write-Host "--------------------------------------------------"
    Write-Host "[INFO] Launching run for agent '$($params.agent)' with seed '$($params.seed)'" -ForegroundColor Green
    Write-Host "[INFO] Log will be saved to '$LOGRUN'"
    
    # Execute the command and redirect all output (stdout and stderr) to the log file.
    # We call the executable directly and use the *> operator to merge all output streams.
    # This is the modern PowerShell equivalent and avoids the Start-Process limitation.
    & python.exe $Full_ARGS *> $LOGRUN
    
    Write-Host "[SUCCESS] Run completed." -ForegroundColor Green
}

Write-Host "--------------------------------------------------"
Write-Host "All runs completed." -ForegroundColor Magenta
